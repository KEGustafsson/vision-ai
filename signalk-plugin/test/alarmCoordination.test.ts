import { describe, it, expect } from 'vitest';
import {
  COLLISION_ALERTS_PROPERTY,
  CollisionAlertsWatch,
  othersAlarmedRefs,
} from '../src/alarmCoordination';
import { ServerApp } from '../src/skapp';

type Listener = (history: Array<{ value?: unknown } | undefined>) => void;

// PropertyValues as the server implements them: a history per name that
// starts with undefined, replayed in full to each new subscriber.
class FakePropertyValues {
  history: Array<{ value?: unknown } | undefined> = [undefined];
  listeners = new Set<Listener>();
  subscribed: string[] = [];
  onPropertyValues = (name: string, cb: Listener): (() => void) => {
    this.subscribed.push(name);
    this.listeners.add(cb);
    cb(this.history);
    return () => this.listeners.delete(cb);
  };
  emit(value: unknown): void {
    this.history = [...this.history, { value }];
    for (const cb of this.listeners) cb(this.history);
  }
}

describe('CollisionAlertsWatch', () => {
  it('follows the collisionAlerts announcement, whenever it arrives', () => {
    const pv = new FakePropertyValues();
    pv.emit({ evaluatesTargets: true }); // announced before we subscribe
    const watch = new CollisionAlertsWatch();
    watch.start(pv);
    expect(pv.subscribed).toEqual([COLLISION_ALERTS_PROPERTY]);
    expect(watch.running()).toBe(true);
    pv.emit({ evaluatesTargets: false }); // Collision Alerts stopped
    expect(watch.running()).toBe(false);
    pv.emit({ evaluatesTargets: true }); // and started again
    expect(watch.running()).toBe(true);
  });

  it('keeps its own alarms until something is announced', () => {
    const pv = new FakePropertyValues();
    const watch = new CollisionAlertsWatch();
    watch.start(pv);
    expect(watch.running()).toBe(false);
  });

  it('accepts only the announced shape', () => {
    const pv = new FakePropertyValues();
    const watch = new CollisionAlertsWatch();
    watch.start(pv);
    for (const v of [true, 'yes', { evaluatesTargets: 'true' }, null, () => true]) {
      pv.emit(v);
      expect(watch.running()).toBe(false);
    }
  });

  it('forgets the announcement on stop, so a restart never inherits it', () => {
    const pv = new FakePropertyValues();
    const watch = new CollisionAlertsWatch();
    watch.start(pv);
    pv.emit({ evaluatesTargets: true });
    watch.stop();
    expect(watch.running()).toBe(false);
    expect(pv.listeners.size).toBe(0);
  });

  it('works on a server without PropertyValues', () => {
    const watch = new CollisionAlertsWatch();
    expect(() => watch.start({})).not.toThrow();
    expect(watch.running()).toBe(false);
  });
});

function appWith(tree: unknown): ServerApp {
  return {
    handleMessage: () => {},
    getSelfPath: (path: string) =>
      path === 'notifications.navigation.closestApproach' ? tree : undefined,
    getPath: () => undefined,
    debug: () => {},
    error: () => {},
  };
}

const alarm = (state: string, extra: Record<string, unknown> = {}) => ({ state, method: [], message: '', ...extra });

describe('othersAlarmedRefs', () => {
  it('reads the boat from data.targetRef, from other, or from the path key', () => {
    const refs = othersAlarmedRefs(appWith({
      'urn:mrn:imo:mmsi:244060000': {
        $source: 'collision-alerts',
        value: alarm('alarm', { data: { targetRef: 'vessels.urn:mrn:imo:mmsi:244060000' } }),
      },
      'radar:nav1-17': { $source: 'mayara', value: alarm('warn', { other: 'targets.radar:nav1-17' }) },
      'urn:mrn:imo:mmsi:230000001': { $source: 'derived-data', value: alarm('alarm') },
      'camera:other-3': { $source: 'someplugin', value: alarm('warn') },
    }), 'vision-ai');
    expect([...refs].sort()).toEqual([
      'targets.camera:other-3',
      'targets.radar:nav1-17',
      'vessels.urn:mrn:imo:mmsi:230000001',
      'vessels.urn:mrn:imo:mmsi:244060000',
    ]);
  });

  it('ignores this plugin\'s own alarms and alarms that are not raised', () => {
    const refs = othersAlarmedRefs(appWith({
      'visionai-forward-3': { $source: 'vision-ai', value: alarm('alarm', { data: { targetRef: 'targets.camera:visionai-forward-3' } }) },
      'urn:mrn:imo:mmsi:1': { $source: 'collision-alerts', value: alarm('normal') },
      'urn:mrn:imo:mmsi:2': { $source: 'collision-alerts', value: null },
    }), 'vision-ai');
    expect(refs.size).toBe(0);
  });

  it('reads every source of a path several producers write', () => {
    const refs = othersAlarmedRefs(appWith({
      'urn:mrn:imo:mmsi:5': {
        $source: 'vision-ai',
        value: alarm('alarm'),
        values: {
          'vision-ai': { value: alarm('alarm') },
          notificationApi: { value: alarm('warn') },
        },
      },
    }), 'vision-ai');
    expect([...refs]).toEqual(['vessels.urn:mrn:imo:mmsi:5']);
  });

  it('finds nothing when there is no closest-approach tree', () => {
    expect(othersAlarmedRefs(appWith(undefined), 'vision-ai').size).toBe(0);
    expect(othersAlarmedRefs(appWith(null), 'vision-ai').size).toBe(0);
  });
});
