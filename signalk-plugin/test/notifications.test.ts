import { describe, it, expect, beforeEach } from 'vitest';
import { NotificationManager, degradedFaultKey } from '../src/notifications';
import { withDefaults } from '../src/config';
import { Delta, ServerApp } from '../src/skapp';
import { EnrichedTarget } from '../src/types';

class FakeApp implements ServerApp {
  deltas: Delta[] = [];
  handleMessage(_id: string, delta: Delta): void { this.deltas.push(delta); }
  getSelfPath(): any { return null; }
  getPath(): any { return null; }
  debug(): void {}
  error(): void {}

  // Latest value emitted for a path (null = cleared), or undefined if never set.
  valueFor(path: string): any {
    let v: any;
    for (const d of this.deltas)
      for (const u of d.updates)
        for (const x of u.values || []) if (x.path === path) v = x.value;
    return v;
  }
}

function mobTarget(lastSeen: number, partial: Partial<EnrichedTarget> = {}): EnrichedTarget {
  return {
    track_id: 7, label: 'person', coco_class: 0, confidence: 0.9,
    bbox: { x: 0, y: 0, w: 1, h: 1 }, is_person_in_water: true,
    geometry: { relative_bearing_deg: 0, range_m: 40, range_method: 'horizon', range_confidence: 0.7 },
    pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 0,
    key: 'aft.7', camera: 'aft', bearingTrue: 0.1,
    position: { latitude: 60, longitude: 25 },
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null,
    cpa: null, tcpa: null, sog: null, cog: null, threatLevel: 'none', lastSeen,
    ...partial,
  };
}

describe('NotificationManager — MOB persistence', () => {
  let app: FakeApp;
  // Opt into the hazardous alert explicitly rather than relying on defaults.
  const cfg = withDefaults({ enableMob: true, mobPersistFrames: 3 });

  beforeEach(() => {
    app = new FakeApp();
  });

  it('does not fire from a single sighting lingering across process cycles', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // One detection frame; the target stays in the map (trackTimeoutS retention)
    // while several process cycles pass without any new sighting.
    const t = mobTarget(1_000);
    for (const now of [1_100, 2_100, 3_100, 4_100]) n.evaluate([t], new Set(), now);
    expect(app.valueFor('notifications.mob')).toBeUndefined();
  });

  it('restarts the count when a track stops qualifying as a MOB candidate', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // Two qualifying sightings...
    n.evaluate([mobTarget(1_000)], new Set(), 1_100);
    n.evaluate([mobTarget(2_000)], new Set(), 2_100);
    // ...then the same track is reclassified out of the water (still tracked).
    n.evaluate([mobTarget(3_000, { is_person_in_water: false })], new Set(), 3_100);
    // A later single qualifying frame must NOT complete the old count.
    n.evaluate([mobTarget(4_000)], new Set(), 4_100);
    expect(app.valueFor('notifications.mob')).toBeUndefined();
  });

  it('fires after mobPersistFrames distinct sightings', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // lastSeen advances each cycle: three genuine re-sightings.
    n.evaluate([mobTarget(1_000)], new Set(), 1_100);
    n.evaluate([mobTarget(2_000)], new Set(), 2_100);
    expect(app.valueFor('notifications.mob')).toBeUndefined();
    n.evaluate([mobTarget(3_000)], new Set(), 3_100);
    const v = app.valueFor('notifications.mob');
    expect(v.state).toBe('emergency');
    expect(v.method).toEqual(['visual', 'sound']);
    expect(v.message).toContain('60.00000, 25.00000');
  });
});

describe('NotificationManager — container health', () => {
  let app: FakeApp;
  const cfg = withDefaults({});

  beforeEach(() => {
    app = new FakeApp();
  });

  it('raises containerDown at alarm/visual and clears it', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.setContainerDown('unreachable');
    const v = app.valueFor('notifications.vision.containerDown');
    expect(v.state).toBe('alarm');
    expect(v.method).toEqual(['visual']);
    expect(v.message).toBe('unreachable');

    n.clearContainerDown();
    expect(app.valueFor('notifications.vision.containerDown')).toBeNull();
  });

  it('raises containerDegraded at warn and clears it', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.setContainerDegraded('forward: no frames for 30s');
    const v = app.valueFor('notifications.vision.containerDegraded');
    expect(v.state).toBe('warn');
    expect(v.message).toContain('no frames');

    n.clearContainerDegraded();
    expect(app.valueFor('notifications.vision.containerDegraded')).toBeNull();
  });

  it('evaluate() does not clear externally-managed container notifications', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.setContainerDown('unreachable');
    n.setContainerDegraded('degraded');
    // A normal evaluate cycle with no targets must leave these untouched (they
    // are owned by the health-poll lifecycle, not the per-cycle target sweep).
    n.evaluate([], new Set(), Date.now());
    expect(app.valueFor('notifications.vision.containerDown')).not.toBeNull();
    expect(app.valueFor('notifications.vision.containerDegraded')).not.toBeNull();
  });

  it('clearAll() clears container notifications', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.setContainerDown('unreachable');
    n.clearAll();
    expect(app.valueFor('notifications.vision.containerDown')).toBeNull();
  });
});

// Messages as the plugin logged them on the boat during a camera dropout test.
describe('degradedFaultKey', () => {
  const at = (s: number) => `forward: no frames for ${s}s (camera/RTSP stalled)`;

  it('is the same while one camera simply stays down', () => {
    // 22 log lines for one 110 s outage came from this counter alone.
    expect(degradedFaultKey(at(6))).toBe(degradedFaultKey(at(10)));
    expect(degradedFaultKey(at(6))).toBe(degradedFaultKey(at(110)));
  });

  it('changes when another camera goes down', () => {
    const both = `${at(5)}; aft: no frames for 5s (camera/RTSP stalled)`;
    expect(degradedFaultKey(both)).not.toBe(degradedFaultKey(at(5)));
  });

  it('changes when the pipeline restarts or the cause changes', () => {
    const stalled = `${at(21)}; pipeline restarted 1x (Could not read from resource.)`;
    const restarted = 'forward: no frames for 6s (camera/RTSP stalled); pipeline restarted 2x (all cameras stalled > 20s)';
    expect(degradedFaultKey(stalled)).not.toBe(degradedFaultKey(restarted));
    expect(degradedFaultKey('forward: GStreamer error: Could not read from resource.'))
      .not.toBe(degradedFaultKey(at(6)));
  });

  it('leaves durations that are not a stall counter alone', () => {
    expect(degradedFaultKey('pipeline restarted 2x (all cameras stalled > 20s)'))
      .toBe('pipeline restarted 2x (all cameras stalled > 20s)');
  });
});
