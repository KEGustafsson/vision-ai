// The plugin end to end: where Target Fusion links a camera target to one AIS
// vessel and this plugin's own position match picked another, the collision
// assessment must not come from the other vessel's AIS kinematics and be
// raised under the linked vessel's name.

import { afterEach, describe, expect, it } from 'vitest';
import { AddressInfo } from 'net';
import { createServer, Server } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import createPlugin from '../src/index';
import { Delta, Plugin, ServerApp } from '../src/skapp';

const CPA = 'notifications.navigation.closestApproach.';
const VESSEL_A = 'urn:mrn:imo:mmsi:230000001';
const VESSEL_B = 'vessels.urn:mrn:imo:mmsi:230000002';

function until(cond: () => boolean, timeoutMs = 6000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (cond()) return resolve();
      if (Date.now() > deadline) return reject(new Error('condition not met in time'));
      setTimeout(tick, 10);
    };
    tick();
  });
}

const fresh = (value: unknown) => ({ value, timestamp: new Date().toISOString() });

class FakeApp implements ServerApp {
  deltas: Delta[] = [];
  // Target Fusion links a target only once it has read it from the model, a
  // cycle or more after it was first published.
  linkTo: string | null = null;
  handleMessage(_id: string, delta: Delta): void { this.deltas.push(delta); }
  getSelfPath(path: string): any {
    const values: Record<string, unknown> = {
      'navigation.position': { latitude: 60, longitude: 25 },
      'navigation.headingTrue': 0,
      'navigation.speedOverGround': 0,
      'navigation.courseOverGroundTrue': 0,
    };
    return path in values ? fresh(values[path]) : undefined;
  }
  getPath(path: string): any {
    // AIS vessel A sits where the camera sees the boat, closing fast.
    if (path === 'vessels') {
      return {
        [VESSEL_A]: {
          sensors: { ais: { class: { value: 'A' } } },
          navigation: {
            position: fresh({ latitude: 60 + 400 / 111_320, longitude: 25 }),
            speedOverGround: fresh(8),
            courseOverGroundTrue: fresh(Math.PI),
          },
        },
      };
    }
    if (this.linkTo && /^targets\.camera:visionai-.*\.sameAs$/.test(path)) return { value: this.linkTo };
    return undefined;
  }
  // Delivers deltas the plugin subscribed to, as the server would.
  deliver: ((delta: any) => void) | null = null;
  subscriptionmanager = {
    subscribe: (_cmd: unknown, _unsubscribes: Array<() => void>, _err: unknown, cb: (d: any) => void) => {
      this.deliver = cb;
    },
  };
  debug(): void {}
  error(): void {}
  /** Every collision alarm path ever raised. */
  everRaised(): Set<string> {
    const paths = new Set<string>();
    for (const d of this.deltas)
      for (const u of d.updates)
        for (const v of u.values ?? []) if (v.path.startsWith(CPA) && v.value !== null) paths.add(v.path);
    return paths;
  }
  collisionAlarms(): Map<string, any> {
    const latest = new Map<string, any>();
    for (const d of this.deltas)
      for (const u of d.updates)
        for (const v of u.values ?? []) if (v.path.startsWith(CPA)) latest.set(v.path, v.value);
    for (const [k, v] of latest) if (v === null) latest.delete(k);
    return latest;
  }
}

function event(): string {
  return JSON.stringify({
    schema_version: '1.0',
    camera: 'forward',
    timestamp: new Date().toISOString(),
    frame_seq: 1,
    frame_size: { w: 1280, h: 960 },
    horizon_y: 350,
    inference: { backend: 'mock', latency_ms: 1 },
    calibration_status: 'auto',
    targets: [{
      track_id: 11, stable_id: 5, label: 'vessel', coco_class: 8, confidence: 0.9,
      bbox: { x: 10, y: 20, w: 30, h: 40 }, is_person_in_water: false,
      geometry: { relative_bearing_deg: 0, range_m: 400, range_method: 'horizon', range_confidence: 0.7 },
      pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 3, coasting: false,
    }],
  });
}

describe('plugin — fusion link against its own AIS match', () => {
  let plugin: Plugin | null = null;
  let http: Server | null = null;
  let wss: WebSocketServer | null = null;

  afterEach(async () => {
    plugin?.stop();
    for (const c of wss?.clients ?? []) c.terminate();
    await new Promise<void>((r) => (wss ? wss.close(() => r()) : r()));
    await new Promise<void>((r) => (http ? http.close(() => r()) : r()));
    plugin = null;
    http = null;
    wss = null;
  });

  async function run(app: FakeApp, options: Record<string, unknown> = {}): Promise<void> {
    http = createServer((_req, res) => { res.statusCode = 503; res.end(); });
    wss = new WebSocketServer({ server: http, path: '/ws/events' });
    wss.on('connection', (sock: WebSocket) => {
      const timer = setInterval(() => sock.send(event()), 100);
      sock.on('close', () => clearInterval(timer));
    });
    await new Promise<void>((r) => http!.listen(0, '127.0.0.1', r));
    const { port } = http.address() as AddressInfo;
    plugin = createPlugin(app);
    plugin.start({
      containerUrl: `http://127.0.0.1:${port}`,
      enableVisualRadar: true,
      processIntervalMs: 200,
      detectClasses: [],
      ...options,
    });
  }

  it('alarms a closing vessel by its own AIS match when nothing links the target', async () => {
    const app = new FakeApp();
    await run(app);
    await until(() => app.collisionAlarms().size > 0);
    expect([...app.collisionAlarms().keys()]).toEqual([`${CPA}visionai_mmsi_230000001`]);
  }, 10_000);

  it('does not raise the other vessel\'s assessment under the linked vessel\'s name', async () => {
    const app = new FakeApp();
    // No anti-flap hold, so an alarm that no longer stands is gone at once.
    await run(app, { notifyHoldS: 0 });
    // Before any link, this plugin alarms the boat by its own AIS match...
    await until(() => app.collisionAlarms().size > 0);
    expect([...app.collisionAlarms().keys()]).toEqual([`${CPA}visionai_mmsi_230000001`]);
    // ...then Target Fusion says the boat is vessel B.
    app.linkTo = VESSEL_B;
    await new Promise((r) => setTimeout(r, 1000));
    // The camera alone sees a boat holding still, so no collision risk: the
    // closing kinematics were vessel A's, and the boat is vessel B. The alarm
    // for A goes, and none is raised for B.
    expect([...app.collisionAlarms().keys()]).toEqual([]);
  }, 10_000);

  it('keeps the cycle\'s links when another producer\'s alarm re-checks between cycles', async () => {
    const app = new FakeApp();
    // A long cycle, so the re-check below lands well before the next one.
    await run(app, { notifyHoldS: 0, processIntervalMs: 3000 });
    await until(() => app.collisionAlarms().size > 0);
    // Target Fusion links the boat to vessel B after this cycle's CPA was made
    // from vessel A's AIS, and another producer's alarm changes at once.
    app.linkTo = VESSEL_B;
    app.deliver!({
      updates: [{
        $source: 'collision-alerts',
        values: [{ path: `${CPA}urn:mrn:imo:mmsi:244000000`, value: { state: 'alarm' } }],
      }],
    });
    await new Promise((r) => setTimeout(r, 100));
    // The re-check must not raise vessel A's assessment under vessel B's name.
    expect(app.everRaised().has(`${CPA}visionai_mmsi_230000002`)).toBe(false);
  }, 15_000);
});
