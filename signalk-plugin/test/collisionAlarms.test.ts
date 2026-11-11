import { describe, it, expect, beforeEach } from 'vitest';
import { AlarmCoordination, NotificationManager } from '../src/notifications';
import { sensorTargetContext, sensorTargetId } from '../src/sensorTargets';
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

  // The notification paths currently raised (latest value non-null).
  raised(prefix = 'notifications.'): Map<string, any> {
    const latest = new Map<string, any>();
    for (const d of this.deltas)
      for (const u of d.updates)
        for (const x of u.values || []) if (x.path.startsWith(prefix)) latest.set(x.path, x.value);
    for (const [k, v] of latest) if (v === null) latest.delete(k);
    return latest;
  }
}

const CPA = 'notifications.navigation.closestApproach.';
const DARK = 'notifications.vision.darkTarget.';

function boat(stableId: number, partial: Partial<EnrichedTarget> = {}): EnrichedTarget {
  const camera = partial.camera ?? 'forward';
  return {
    track_id: stableId % 90 + 10, stable_id: stableId, label: 'vessel', coco_class: 8, confidence: 0.9,
    bbox: { x: 0, y: 0, w: 1, h: 1 }, is_person_in_water: false,
    geometry: { relative_bearing_deg: 0, range_m: 300, range_method: 'horizon', range_confidence: 0.7 },
    pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 0,
    key: `${camera}.${stableId}`, camera, bearingTrue: 0.1,
    position: { latitude: 60.003, longitude: 25 },
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null, aisPosition: null,
    cpa: 20, tcpa: 120, sog: 3, cog: Math.PI,
    approach: {
      range: 330,
      self: { latitude: 60.001, longitude: 25 },
      target: { latitude: 60.0011, longitude: 25 },
    },
    threatLevel: 'high', lastSeen: 1_000,
    ...partial,
  };
}

const ctxOf = (t: EnrichedTarget) => sensorTargetContext(sensorTargetId(t) as string);

/** Every target published to the chart, with the given fusion links. */
function coordination(partial: Partial<AlarmCoordination> & { links?: Record<string, string> } = {}): AlarmCoordination {
  const { links = {}, ...rest } = partial;
  return {
    publishedContext: (t) => ctxOf(t),
    sameAs: (context) => links[context] ?? null,
    othersAlarmed: new Set(),
    ...rest,
  };
}

describe('NotificationManager — collision alarms', () => {
  let app: FakeApp;
  const cfg = withDefaults({ notifyHoldS: 10 });

  beforeEach(() => {
    app = new FakeApp();
  });

  it('raises a closest-approach alarm chartplotters can draw', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const t = boat(137);
    n.evaluate([t], new Set([t.key]), 1_000, coordination());
    const raised = app.raised();
    expect([...raised.keys()]).toEqual([`${CPA}visionai-forward-137`]);
    const v = raised.get(`${CPA}visionai-forward-137`);
    expect(v.state).toBe('alarm');
    expect(v.method).toEqual(['visual', 'sound']);
    expect(v.message).toContain('(no AIS)');
    expect(v.other).toBe('targets.camera:visionai-forward-137');
    expect(v.data).toMatchObject({
      targetRef: 'targets.camera:visionai-forward-137',
      source: 'camera',
      cpa: 20,
      tcpa: 120,
      range: 330,
      targetSog: 3,
      targetCog: Math.PI,
      cpaPositions: { self: { latitude: 60.001, longitude: 25 }, target: { latitude: 60.0011, longitude: 25 } },
    });
    // The old vision-specific path is gone.
    expect(app.raised('notifications.vision.collision.').size).toBe(0);
  });

  it('names no target when nothing is published or matched, keeping the course lines', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.evaluate([boat(5)], new Set(), 1_000, coordination({ publishedContext: () => null }));
    const v = app.raised().get(`${CPA}visionai-forward-5`);
    expect(v.data.targetRef).toBeUndefined();
    expect(v.other).toBeUndefined();
    expect(v.data.cpaPositions).toBeDefined();
  });

  it('alarms once for a boat both cameras see, with the more urgent track', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const bow = boat(3, { threatLevel: 'medium', tcpa: 400 });
    const stern = boat(9, { camera: 'aft', key: 'aft.9', threatLevel: 'high', tcpa: 100 });
    // Target Fusion links the stern camera's track to the bow camera's.
    const links = { [ctxOf(stern)]: ctxOf(bow) };
    n.evaluate([bow, stern], new Set(), 1_000, coordination({ links }));
    const raised = app.raised(CPA);
    expect(raised.size).toBe(1);
    const [path, v] = [...raised][0];
    expect(path).toBe(`${CPA}visionai-forward-3`);
    expect(v.state).toBe('alarm');
    expect(v.data.tcpa).toBe(100);
    expect(v.data.targetRef).toBe(ctxOf(bow));
  });

  it('alarms once when the links between tracks run through an AIS match', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // The bow track matches AIS; fusion links the stern track to the bow's target.
    const bow = boat(3, { aisMmsi: '244060000', aisCorrelated: true });
    const stern = boat(9, { camera: 'aft', key: 'aft.9' });
    const links = { [ctxOf(stern)]: ctxOf(bow) };
    n.evaluate([bow, stern], new Set(), 1_000, coordination({ links }));
    n.evaluate([stern, bow], new Set(), 2_000, coordination({ links }));
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-mmsi-244060000`]);
  });

  it('stands down when another plugin alarms any track of the boat', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const bow = boat(3);
    const stern = boat(9, { camera: 'aft', key: 'aft.9' });
    const links = { [ctxOf(stern)]: ctxOf(bow) };
    // The other plugin names the stern camera's target, not the bow's.
    n.evaluate([bow, stern], new Set(), 1_000,
      coordination({ links, othersAlarmed: new Set([ctxOf(stern)]) }));
    expect(app.raised(CPA).size).toBe(0);
  });

  it('releases its held alarm when another plugin takes over a boat that stopped threatening', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    n.evaluate([boat(3)], new Set(), 1_000, coordination());
    expect(app.raised(CPA).size).toBe(1);
    n.evaluate([boat(3, { threatLevel: 'none' })], new Set(), 2_000,
      coordination({ othersAlarmed: new Set([ctxOf(boat(3))]) }));
    expect(app.raised(CPA).size).toBe(0);
  });

  it('hands its alarm over the moment another producer alarms the boat, and takes it back', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    const t = boat(3);
    n.evaluate([t], new Set([t.key]), 1_000, coordination());
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-forward-3`]);
    // Collision Alerts raises its alarm: ours goes in the same evaluation.
    n.evaluate([t], new Set([t.key]), 2_000, coordination({ othersAlarmed: new Set([ctxOf(t)]) }));
    expect(app.raised('notifications.').size).toBe(0);
    // Its alarm clears while the boat still threatens: ours comes back.
    n.evaluate([t], new Set([t.key]), 3_000, coordination());
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-forward-3`]);
  });

  it('alarms once for tracks matched to the same AIS vessel, named by the vessel', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const a = boat(3, { aisMmsi: '244060000', aisCorrelated: true });
    const b = boat(4, { camera: 'aft', key: 'aft.4', aisMmsi: '244060000', aisCorrelated: true });
    n.evaluate([a, b], new Set(), 1_000, coordination({ publishedContext: () => null }));
    const raised = app.raised(CPA);
    expect([...raised.keys()]).toEqual([`${CPA}visionai-mmsi-244060000`]);
    expect(raised.get(`${CPA}visionai-mmsi-244060000`).data.targetRef)
      .toBe('vessels.urn:mrn:imo:mmsi:244060000');
  });

  it('never writes a path another producer uses for the same boat', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.evaluate([boat(3, { aisMmsi: '244060000' })], new Set(), 1_000, coordination());
    const [path] = [...app.raised(CPA).keys()];
    // Collision Alerts and SK Derived Data key by the bare target id.
    expect(path).not.toBe(`${CPA}urn:mrn:imo:mmsi:244060000`);
    expect(path).not.toBe(`${CPA}camera:visionai-forward-3`);
  });

  it('keeps alarming a risky boat no other producer alarms, whatever plugins run', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // Another producer alarms a different boat only: this one stays ours.
    n.evaluate([boat(3)], new Set(), 1_000,
      coordination({ othersAlarmed: new Set(['vessels.urn:mrn:imo:mmsi:999999999']) }));
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-forward-3`]);
  });

  it('gives boats whose contexts differ only in punctuation alarms of their own', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const a = boat(3);
    const b = boat(4);
    const links = { [ctxOf(a)]: 'targets.radar:a:b', [ctxOf(b)]: 'targets.radar:a_b' };
    n.evaluate([a, b], new Set(), 1_000, coordination({ links }));
    expect(app.raised(CPA).size).toBe(2);
  });

  it('lets a fusion link to an AIS vessel override its own position match', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // Fusion says this track is vessel B; our position match said vessel A.
    const t = boat(3, { aisMmsi: '230000001', aisCorrelated: true });
    const links = { [ctxOf(t)]: 'vessels.urn:mrn:imo:mmsi:230000002' };
    // Another producer alarms vessel A: that must not silence the alarm for B.
    n.evaluate([t], new Set(), 1_000,
      coordination({ links, othersAlarmed: new Set(['vessels.urn:mrn:imo:mmsi:230000001']) }));
    const raised = app.raised(CPA);
    expect([...raised.keys()]).toEqual([`${CPA}visionai-mmsi-230000002`]);
    expect(raised.get(`${CPA}visionai-mmsi-230000002`).data.targetRef)
      .toBe('vessels.urn:mrn:imo:mmsi:230000002');
  });

  it('does not merge a track linked to one vessel with a track matched to another', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const linked = boat(3, { aisMmsi: '230000001' });
    const matched = boat(4, { camera: 'aft', key: 'aft.4', aisMmsi: '230000001' });
    const links = { [ctxOf(linked)]: 'vessels.urn:mrn:imo:mmsi:230000002' };
    n.evaluate([linked, matched], new Set(), 1_000, coordination({ links }));
    expect([...app.raised(CPA).keys()].sort()).toEqual(
      [`${CPA}visionai-mmsi-230000001`, `${CPA}visionai-mmsi-230000002`]);
  });

  it('releases an alarm at once when its boat is renamed, whichever track qualifies', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    const bow = boat(3);
    const stern = boat(9, { camera: 'aft', key: 'aft.9', threatLevel: 'none' });
    const links = { [ctxOf(stern)]: ctxOf(bow) };
    n.evaluate([bow, stern], new Set(), 1_000, coordination({ links }));
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-forward-3`]);
    // A link to an AIS vessel renames the boat, and now the stern track is
    // the one that threatens.
    const links2 = { ...links, [ctxOf(bow)]: 'vessels.urn:mrn:imo:mmsi:230000001' };
    n.evaluate([{ ...bow, threatLevel: 'none' }, { ...stern, threatLevel: 'high' }], new Set(), 2_000,
      coordination({ links: links2 }));
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-mmsi-230000001`]);
  });

  it('stands down at once, without its hold, when another plugin alarms the boat', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    const t = boat(3, { aisMmsi: '244060000' });
    n.evaluate([t], new Set(), 1_000, coordination());
    expect(app.raised(CPA).size).toBe(1);
    n.evaluate([t], new Set(), 2_000,
      coordination({ othersAlarmed: new Set(['vessels.urn:mrn:imo:mmsi:244060000']) }));
    expect(app.raised(CPA).size).toBe(0);
  });

  it('recognises another plugin\'s alarm on its own target or on a fusion link', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const t = boat(3);
    const links = { [ctxOf(t)]: 'targets.radar:nav1-17' };
    n.evaluate([t], new Set(), 1_000,
      coordination({ links, othersAlarmed: new Set(['targets.radar:nav1-17']) }));
    n.evaluate([boat(4)], new Set(), 1_000,
      coordination({ othersAlarmed: new Set([ctxOf(boat(4))]) }));
    expect(app.raised(CPA).size).toBe(0);
  });

  it('moves the alarm to the AIS vessel without alarming twice while it holds', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    n.evaluate([boat(3)], new Set(), 1_000, coordination());
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-forward-3`]);
    // A fusion link to the AIS vessel arrives: the boat is now named by it.
    const links = { [ctxOf(boat(3))]: 'vessels.urn:mrn:imo:mmsi:244060000' };
    n.evaluate([boat(3)], new Set(), 2_000, coordination({ links }));
    expect([...app.raised(CPA).keys()]).toEqual([`${CPA}visionai-mmsi-244060000`]);
  });

  it('still holds an alarm whose boat merely stopped threatening (anti-flap)', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    n.evaluate([boat(3)], new Set(), 1_000, coordination());
    n.evaluate([boat(3, { threatLevel: 'none' })], new Set(), 2_000, coordination());
    expect(app.raised(CPA).size).toBe(1);
    n.evaluate([boat(3, { threatLevel: 'none' })], new Set(), 12_000, coordination());
    expect(app.raised(CPA).size).toBe(0);
  });
});

describe('NotificationManager — dark targets', () => {
  let app: FakeApp;
  const cfg = withDefaults({ notifyHoldS: 10 });

  beforeEach(() => {
    app = new FakeApp();
  });

  it('raises one dark-target alert per boat', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const bow = boat(3, { threatLevel: 'none' });
    const stern = boat(9, { camera: 'aft', key: 'aft.9', threatLevel: 'none' });
    const links = { [ctxOf(stern)]: ctxOf(bow) };
    n.evaluate([bow, stern], new Set([bow.key, stern.key]), 1_000, coordination({ links }));
    expect([...app.raised(DARK).keys()]).toEqual([`${DARK}visionai-forward-3`]);
  });

  it('raises no dark-target alert for a boat under a collision alarm', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const t = boat(3);
    n.evaluate([t], new Set([t.key]), 1_000, coordination());
    expect(app.raised(DARK).size).toBe(0);
    expect(app.raised(CPA).size).toBe(1);
  });

  it('clears a standing dark-target alert at once when the boat becomes a collision risk', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    n.evaluate([boat(3, { threatLevel: 'none' })], new Set(['forward.3']), 1_000, coordination());
    expect(app.raised(DARK).size).toBe(1);
    n.evaluate([boat(3)], new Set(['forward.3']), 2_000, coordination());
    expect(app.raised(DARK).size).toBe(0);
    expect(app.raised(CPA).size).toBe(1);
  });

  it('raises no dark-target alert for a boat another plugin already alarms', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const t = boat(3, { threatLevel: 'none' });
    n.evaluate([t], new Set([t.key]), 1_000, coordination({ othersAlarmed: new Set([ctxOf(t)]) }));
    expect(app.raised('notifications.').size).toBe(0);
  });

  it('raises no dark-target alert for a boat another plugin alarms, with its own collision alarms off', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', withDefaults({ notifyCollision: false }));
    const t = boat(3, { threatLevel: 'none' });
    n.evaluate([t], new Set([t.key]), 1_000, coordination({ othersAlarmed: new Set([ctxOf(t)]) }));
    expect(app.raised('notifications.').size).toBe(0);
  });

  it('clears a dark-target alert at once when a new link names a boat another plugin alarms', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', withDefaults({ notifyCollision: false, notifyHoldS: 10 }));
    const t = boat(3, { threatLevel: 'none' });
    n.evaluate([t], new Set([t.key]), 1_000, coordination());
    expect([...app.raised(DARK).keys()]).toEqual([`${DARK}visionai-forward-3`]);
    // Fusion now links the track to a radar target that another plugin alarms:
    // the boat's name changes, and the alert under its old name must go too.
    const links = { [ctxOf(t)]: 'targets.radar:nav1-17' };
    n.evaluate([t], new Set([t.key]), 2_000,
      coordination({ links, othersAlarmed: new Set(['targets.radar:nav1-17']) }));
    expect(app.raised('notifications.').size).toBe(0);
  });

  it('does not call a boat dark when one of its tracks is an AIS vessel', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    // The bow track matched AIS; the linked stern track, on its own, did not.
    const bow = boat(3, { aisMmsi: '230000001', threatLevel: 'none' });
    const stern = boat(9, { camera: 'aft', key: 'aft.9', threatLevel: 'high' });
    const links = { [ctxOf(stern)]: ctxOf(bow) };
    n.evaluate([bow, stern], new Set([stern.key]), 1_000, coordination({ links }));
    expect(app.raised(DARK).size).toBe(0);
    const [cpa] = [...app.raised(CPA).values()];
    expect(cpa.message).not.toContain('no AIS');
  });

  it('clears a standing dark-target alert at once when its boat matches AIS', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg); // hold 10 s
    const t = boat(3, { threatLevel: 'none' });
    n.evaluate([t], new Set([t.key]), 1_000, coordination());
    expect(app.raised(DARK).size).toBe(1);
    // Its only dark track matches an AIS vessel: no track of the boat is dark
    // any more, and the alert goes without waiting for its hold.
    n.evaluate([{ ...t, aisMmsi: '230000001', aisCorrelated: true }], new Set(), 2_000, coordination());
    expect(app.raised(DARK).size).toBe(0);
  });

  it('does not call a boat dark that fusion links to an AIS vessel', () => {
    const n = new NotificationManager(app, 'signalk-vision-ai', cfg);
    const t = boat(3, { threatLevel: 'none' });
    const links = { [ctxOf(t)]: 'vessels.urn:mrn:imo:mmsi:230000001' };
    n.evaluate([t], new Set([t.key]), 1_000, coordination({ links }));
    expect(app.raised(DARK).size).toBe(0);
  });
});
