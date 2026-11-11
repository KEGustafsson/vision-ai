// The plugin against a real local WebSocket: when the container restarts its
// track serials start over, so the targets of the previous connection must be
// withdrawn (null position) before a new boat can arrive under a reused serial.

import { afterEach, describe, expect, it } from 'vitest';
import { AddressInfo } from 'net';
import { createServer, Server } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import createPlugin from '../src/index';
import { Delta, Plugin, ServerApp } from '../src/skapp';

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

class FakeApp implements ServerApp {
  deltas: Delta[] = [];
  handleMessage(_id: string, delta: Delta): void { this.deltas.push(delta); }
  getSelfPath(path: string): any {
    const now = new Date().toISOString();
    const values: Record<string, unknown> = {
      'navigation.position': { latitude: 60, longitude: 25 },
      'navigation.headingTrue': 0,
      'navigation.speedOverGround': 0,
      'navigation.courseOverGroundTrue': 0,
    };
    return path in values ? { value: values[path], timestamp: now } : undefined;
  }
  getPath(): any { return undefined; }
  debug(): void {}
  error(): void {}
  positions(context: string): unknown[] {
    return this.deltas
      .filter((d) => d.context === context)
      .flatMap((d) => d.updates.flatMap((u) => u.values ?? []))
      .filter((v) => v.path === 'navigation.position')
      .map((v) => v.value);
  }
}

function event(stableId: number, bearingDeg: number): string {
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
      track_id: 11, stable_id: stableId, label: 'vessel', coco_class: 8, confidence: 0.9,
      bbox: { x: 10, y: 20, w: 30, h: 40 }, is_person_in_water: false,
      geometry: { relative_bearing_deg: bearingDeg, range_m: 400, range_method: 'horizon', range_confidence: 0.7 },
      pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 3, coasting: false,
    }],
  });
}

describe('plugin — container restart', () => {
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

  it('withdraws the old session\'s targets before a reused serial names a new boat', async () => {
    http = createServer((_req, res) => { res.statusCode = 503; res.end(); });
    wss = new WebSocketServer({ server: http, path: '/ws/events' });
    let connections = 0;
    wss.on('connection', (sock: WebSocket) => {
      connections++;
      // Session 1 sees a boat dead ahead; after the restart, session 2 hands
      // the same serial to a different boat off the starboard bow.
      sock.send(event(5, connections === 1 ? 0 : 30));
    });
    await new Promise<void>((r) => http!.listen(0, '127.0.0.1', r));
    const { port } = http.address() as AddressInfo;

    const app = new FakeApp();
    plugin = createPlugin(app);
    plugin.start({
      containerUrl: `http://127.0.0.1:${port}`,
      enableVisualRadar: true,
      processIntervalMs: 200,
      detectClasses: [],
    });
    const ctx = 'targets.camera:visionai-forward-5';

    await until(() => app.positions(ctx).length === 1);
    // The container restarts: its connection drops and the plugin reconnects.
    for (const c of wss.clients) c.terminate();
    await until(() => connections === 2 && app.positions(ctx).length >= 3);

    const [first, withdrawn, second] = app.positions(ctx) as Array<{ longitude: number } | null>;
    expect(first).not.toBeNull();
    expect(withdrawn).toBeNull();
    expect(second).not.toBeNull();
    expect(second!.longitude).not.toBeCloseTo(first!.longitude, 6);
  }, 10_000);
});
