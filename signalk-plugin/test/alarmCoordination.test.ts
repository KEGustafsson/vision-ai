import { describe, it, expect } from 'vitest';
import {
  ForeignAlarmWatch,
  othersAlarmedRefs,
  touchesForeignAlarm,
} from '../src/alarmCoordination';
import { ServerApp } from '../src/skapp';

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

  it('recognises its own alarms however the server stores the source label', () => {
    const refs = othersAlarmedRefs(appWith({
      // Older servers store a plugin's label with a suffix.
      'urn:mrn:imo:mmsi:7': { $source: 'vision-ai.XX', value: alarm('alarm') },
      // Its own alarms are keyed visionai-…, whatever the source says.
      'visionai_mmsi_8': { $source: 'unexpected', value: alarm('alarm', { other: 'vessels.urn:mrn:imo:mmsi:8' }) },
      // A different plugin whose label merely starts the same way still counts.
      'urn:mrn:imo:mmsi:9': { $source: 'vision-ai-other', value: alarm('alarm') },
    }), 'vision-ai');
    expect([...refs]).toEqual(['vessels.urn:mrn:imo:mmsi:9']);
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

const cpaDelta = (source: string, key: string, state: string | null) => ({
  updates: [{
    $source: source,
    values: [{ path: `notifications.navigation.closestApproach.${key}`, value: state === null ? null : { state } }],
  }],
});

describe('touchesForeignAlarm', () => {
  it('sees another producer clearing an alarm as well as raising one', () => {
    expect(touchesForeignAlarm(cpaDelta('notificationsApi', 'camera:x-3', 'normal'), 'vision-ai')).toBe(true);
    expect(touchesForeignAlarm(cpaDelta('notificationsApi', 'camera:x-3', null), 'vision-ai')).toBe(true);
    expect(touchesForeignAlarm(cpaDelta('notificationsApi', 'camera:x-3', 'alarm'), 'vision-ai')).toBe(true);
    expect(touchesForeignAlarm(cpaDelta('vision-ai.XX', 'urn:mrn:imo:mmsi:1', null), 'vision-ai')).toBe(false);
    expect(touchesForeignAlarm(cpaDelta('x', 'visionai-forward-x-3', null), 'vision-ai')).toBe(false);
  });
});

describe('ForeignAlarmWatch', () => {
  it('subscribes to closest-approach alarms and calls back on a foreign one', () => {
    let deliver: ((delta: any) => void) | null = null;
    let subscribed: any = null;
    const unsubscribed: string[] = [];
    const app = {
      subscriptionmanager: {
        subscribe: (cmd: any, unsubscribes: Array<() => void>, _err: unknown, cb: (d: any) => void) => {
          subscribed = cmd;
          deliver = cb;
          unsubscribes.push(() => unsubscribed.push('x'));
        },
      },
    };
    let calls = 0;
    const watch = new ForeignAlarmWatch();
    watch.start(app, 'vision-ai', () => calls++);
    expect(subscribed.subscribe[0].path).toBe('notifications.navigation.closestApproach.*');
    deliver!(cpaDelta('vision-ai.XX', 'visionai-forward-x-3', 'alarm'));
    expect(calls).toBe(0);
    deliver!(cpaDelta('notificationsApi', 'camera:visionai-forward-x-3', 'alarm'));
    expect(calls).toBe(1);
    deliver!(cpaDelta('notificationsApi', 'camera:visionai-forward-x-3', 'normal'));
    expect(calls).toBe(2);
    watch.stop();
    expect(unsubscribed).toEqual(['x']);
  });

  it('works on a server without delta subscriptions', () => {
    const watch = new ForeignAlarmWatch();
    expect(() => watch.start({}, 'vision-ai', () => undefined)).not.toThrow();
    expect(() => watch.stop()).not.toThrow();
  });
});
