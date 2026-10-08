// IMU horizon compensation, plugin half: reading navigation.attitude safely and
// low-pass filtering it before it is pushed to the container.

import { describe, it, expect } from 'vitest';
import { AttitudeSmoother, AttitudeWatch } from '../src/attitude';
import { withDefaults } from '../src/config';
import { readAttitude } from '../src/nav';
import { ServerApp } from '../src/skapp';

function app(paths: Record<string, unknown>): ServerApp {
  return {
    getSelfPath: (path: string) => paths[path],
    getPath: () => undefined,
    handleMessage: () => undefined,
    debug: () => undefined,
    error: () => undefined,
  } as unknown as ServerApp;
}

const NOW = Date.parse('2026-10-04T12:00:00Z');
const at = (value: unknown, ageS = 0) => ({
  value,
  timestamp: new Date(NOW - ageS * 1000).toISOString(),
});

describe('readAttitude', () => {
  it('reads roll/pitch (rad) from the spec navigation.attitude object', () => {
    const a = app({ 'navigation.attitude': at({ roll: 0.1, pitch: -0.05, yaw: 0.2 }) });
    expect(readAttitude(a, 2, NOW)).toEqual({ pitch: -0.05, roll: 0.1 });
  });

  it('drops attitude older than the max age instead of applying a frozen heel', () => {
    const a = app({ 'navigation.attitude': at({ roll: 0.3, pitch: 0 }, 5) });
    expect(readAttitude(a, 2, NOW)).toBeNull();
    expect(readAttitude(a, 0, NOW)).toEqual({ pitch: 0, roll: 0.3 }); // age check off
  });

  it('returns null, not a level boat, when pitch or roll is missing', () => {
    // n2k-signalk maps an unavailable PGN 127257 field to NaN -> null in JSON.
    expect(readAttitude(app({ 'navigation.attitude': at({ roll: 0.1, pitch: null, yaw: 1 }) }), 2, NOW)).toBeNull();
    expect(readAttitude(app({ 'navigation.attitude': at({ roll: NaN, pitch: 0 }) }), 2, NOW)).toBeNull();
    expect(readAttitude(app({}), 2, NOW)).toBeNull();
  });

  it('refuses values beyond ±90° (degrees published as radians)', () => {
    expect(readAttitude(app({ 'navigation.attitude': at({ roll: 12, pitch: 2 }) }), 2, NOW)).toBeNull();
  });

  it('accepts a delta-only bare value (no timestamp to age)', () => {
    expect(readAttitude(app({ 'navigation.attitude': { roll: 0.2, pitch: 0.01 } }), 2, NOW)).toEqual({
      pitch: 0.01,
      roll: 0.2,
    });
  });
});

describe('AttitudeSmoother', () => {
  it('starts from the first sample and converges with the time constant', () => {
    const s = new AttitudeSmoother(5);
    expect(s.update({ pitch: 0, roll: 0 }, 0)).toEqual({ pitch: 0, roll: 0 });
    // Step to 10° heel: after one time constant ~63% of the step.
    let out = null;
    for (let t = 200; t <= 5000; t += 200) out = s.update({ pitch: 0, roll: 0.2 }, t);
    expect(out!.roll).toBeCloseTo(0.2 * (1 - Math.exp(-1)), 6);
  });

  it('largely ignores wave motion while following the mean heel', () => {
    const s = new AttitudeSmoother(5);
    let maxDev = 0;
    for (let t = 0; t <= 60_000; t += 200) {
      // 15° mean heel + ±5° roll on a 4 s wave.
      const roll = 0.26 + 0.087 * Math.sin((2 * Math.PI * t) / 4000);
      const out = s.update({ pitch: 0, roll }, t)!;
      if (t > 30_000) maxDev = Math.max(maxDev, Math.abs(out.roll - 0.26));
    }
    expect(maxDev).toBeLessThan(0.087 * 0.15);
  });

  it('passes the raw attitude through with a zero time constant', () => {
    const s = new AttitudeSmoother(0);
    s.update({ pitch: 0, roll: 0 }, 0);
    expect(s.update({ pitch: 0.05, roll: -0.1 }, 200)).toEqual({ pitch: 0.05, roll: -0.1 });
  });

  it('restarts from the next valid sample after a gap, never from an old heel', () => {
    const s = new AttitudeSmoother(5);
    s.update({ pitch: 0, roll: 0.3 }, 0);
    expect(s.update(null, 200)).toBeNull();
    expect(s.update({ pitch: 0, roll: -0.2 }, 400)).toEqual({ pitch: 0, roll: -0.2 });
  });
});

describe('attitude config', () => {
  it('is off by default (opt-in after on-board sign check)', () => {
    expect(withDefaults(undefined).enableAttitudeCompensation).toBe(false);
  });

  it('keeps the push cadence inside the container age-out window', () => {
    expect(withDefaults({ attitudeIntervalMs: 5000 }).attitudeIntervalMs).toBe(200);
    expect(withDefaults({ attitudeIntervalMs: 10 }).attitudeIntervalMs).toBe(200);
    expect(withDefaults({ attitudeIntervalMs: 500 }).attitudeIntervalMs).toBe(500);
    expect(withDefaults({ attitudeSmoothingS: -1 }).attitudeSmoothingS).toBe(5);
  });
});

describe('AttitudeWatch', () => {
  const T0 = 1_000_000;
  const navUp = () => true;
  const navDown = () => false;

  it('stays quiet while SignalK is still coming up after a restart', () => {
    const w = new AttitudeWatch(T0, 15_000, 120_000);
    // No nav, no attitude yet: bus still cold, nothing to report.
    for (let t = T0; t < T0 + 60_000; t += 200) expect(w.update(false, navDown, t)).toBeNull();
    // Nav starts flowing; attitude follows a few seconds later.
    expect(w.update(false, navUp, T0 + 60_000)).toBeNull();
    expect(w.update(false, navUp, T0 + 65_000)).toBeNull();
    expect(w.update(true, navUp, T0 + 66_000)).toBe('arrived');
    expect(w.update(true, navUp, T0 + 66_200)).toBeNull();
  });

  it('reports a missing IMU once own-ship nav has flowed for the settle time', () => {
    const w = new AttitudeWatch(T0, 15_000, 120_000);
    expect(w.update(false, navUp, T0 + 1_000)).toBeNull();
    expect(w.update(false, navUp, T0 + 15_999)).toBeNull();
    expect(w.update(false, navUp, T0 + 16_000)).toBe('never-arrived');
    // Once, not per tick.
    expect(w.update(false, navUp, T0 + 16_200)).toBeNull();
    expect(w.update(true, navUp, T0 + 30_000)).toBe('resumed');
  });

  it('still reports after the boot cap if the bus never comes up', () => {
    const w = new AttitudeWatch(T0, 15_000, 120_000);
    expect(w.update(false, navDown, T0 + 119_999)).toBeNull();
    expect(w.update(false, navDown, T0 + 120_000)).toBe('never-arrived');
  });

  it('reports a loss at once after attitude has been seen, and the recovery', () => {
    const w = new AttitudeWatch(T0, 15_000, 120_000);
    expect(w.update(true, navUp, T0)).toBe('arrived');
    expect(w.update(false, navUp, T0 + 200)).toBe('lost');
    expect(w.update(false, navUp, T0 + 400)).toBeNull();
    expect(w.update(true, navUp, T0 + 600)).toBe('resumed');
  });

  it('only probes own-ship nav while warming up', () => {
    const w = new AttitudeWatch(T0);
    let probes = 0;
    const probe = () => (probes++, true);
    w.update(false, probe, T0);
    w.update(true, probe, T0 + 200);
    w.update(false, probe, T0 + 400);
    expect(probes).toBe(1);
  });
});
