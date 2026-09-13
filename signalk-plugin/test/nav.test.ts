// Own-ship heading: many small vessels publish only a magnetic heading, and
// without a true one nothing downstream works at all — no georeferencing, no
// AIS fusion, no CPA, no position on a man-overboard.

import { describe, it, expect } from 'vitest';
import { readOwnShip } from '../src/nav';
import { deg2rad, rad2deg } from '../src/geo';
import { ServerApp } from '../src/skapp';

/** A SignalK stub serving full-model nodes ({ value, timestamp }). */
function app(paths: Record<string, unknown>, ts = new Date().toISOString()): ServerApp {
  return {
    getSelfPath: (path: string) =>
      path in paths ? { value: paths[path], timestamp: ts } : undefined,
    getPath: () => undefined,
    handleMessage: () => undefined,
    debug: () => undefined,
    error: () => undefined,
  } as unknown as ServerApp;
}

const position = { latitude: 60, longitude: 25 };

describe('readOwnShip heading', () => {
  it('prefers a true heading when the vessel publishes one', () => {
    const own = readOwnShip(app({
      'navigation.position': position,
      'navigation.headingTrue': deg2rad(90),
      'navigation.headingMagnetic': deg2rad(80),
      'navigation.magneticVariation': deg2rad(3),
    }), 5);
    expect(rad2deg(own.headingTrue as number)).toBeCloseTo(90, 6);
  });

  it('derives true heading from magnetic plus variation', () => {
    // East variation is added: 80°M with 8°E variation is 88°T.
    const own = readOwnShip(app({
      'navigation.position': position,
      'navigation.headingMagnetic': deg2rad(80),
      'navigation.magneticVariation': deg2rad(8),
    }), 5);
    expect(rad2deg(own.headingTrue as number)).toBeCloseTo(88, 6);
    expect(own.stale).toBe(false);
  });

  it('subtracts a west variation and wraps into [0, 2π)', () => {
    // 5°M with 10°W variation is 355°T, not −5°.
    const own = readOwnShip(app({
      'navigation.position': position,
      'navigation.headingMagnetic': deg2rad(5),
      'navigation.magneticVariation': deg2rad(-10),
    }), 5);
    expect(rad2deg(own.headingTrue as number)).toBeCloseTo(355, 6);
  });

  it('stays null without a variation to convert with', () => {
    // Guessing that magnetic equals true would put every bearing out by the
    // local variation — up to tens of degrees at high latitude.
    const own = readOwnShip(app({
      'navigation.position': position,
      'navigation.headingMagnetic': deg2rad(80),
    }), 5);
    expect(own.headingTrue).toBeNull();
  });

  it('does not fall back to course over ground', () => {
    // Leeway and current make COG the wrong answer for where the cameras point.
    const own = readOwnShip(app({
      'navigation.position': position,
      'navigation.courseOverGroundTrue': deg2rad(120),
    }), 5);
    expect(own.headingTrue).toBeNull();
  });

  it('uses a fresh magnetic heading when the true one has aged out', () => {
    const old = new Date(Date.now() - 60_000).toISOString();
    const fresh = new Date().toISOString();
    const a = {
      getSelfPath: (path: string) => {
        if (path === 'navigation.headingTrue') return { value: deg2rad(90), timestamp: old };
        if (path === 'navigation.headingMagnetic') return { value: deg2rad(80), timestamp: fresh };
        if (path === 'navigation.magneticVariation') return { value: deg2rad(8), timestamp: fresh };
        if (path === 'navigation.position') return { value: position, timestamp: fresh };
        return undefined;
      },
      getPath: () => undefined,
      handleMessage: () => undefined,
      debug: () => undefined,
      error: () => undefined,
    } as unknown as ServerApp;

    const own = readOwnShip(a, 5);
    expect(rad2deg(own.headingTrue as number)).toBeCloseTo(88, 6);
    // The heading in use is fresh, so it must not be reported as stale.
    expect(own.stale).toBe(false);
  });

  it('reports stale when the magnetic source it fell back to is itself old', () => {
    const old = new Date(Date.now() - 60_000).toISOString();
    const own = readOwnShip(app({
      'navigation.position': position,
      'navigation.headingMagnetic': deg2rad(80),
      'navigation.magneticVariation': deg2rad(8),
    }, old), 5);
    expect(own.headingTrue).toBeNull();
    expect(own.stale).toBe(true);
  });

  // The previous test has every path aged out, so `stale` would come back true
  // from the position alone. These two age out ONLY the fallback path, with
  // everything else fresh, so nothing but the heading can raise the flag.
  it('flags stale when only the magnetic heading has aged out', () => {
    const own = readOwnShip(mixedAge({ 'navigation.headingMagnetic': 'old' }), 5);
    expect(own.headingTrue).toBeNull();
    // "Heading unknown because a source went stale" must never read as
    // "nothing was dropped for age" — that is the whole contract of the flag.
    expect(own.stale).toBe(true);
    // ...and the values that are fresh are still delivered.
    expect(own.position).not.toBeNull();
    expect(own.sog).toBeCloseTo(3, 6);
  });

  it('flags stale when only the variation has aged out', () => {
    const own = readOwnShip(mixedAge({ 'navigation.magneticVariation': 'old' }), 5);
    expect(own.headingTrue).toBeNull();
    expect(own.stale).toBe(true);
  });

  it('stays fresh when both fallback paths are current', () => {
    // The same stub with nothing aged out: the flag must not be sticky.
    const own = readOwnShip(mixedAge({}), 5);
    expect(rad2deg(own.headingTrue as number)).toBeCloseTo(88, 6);
    expect(own.stale).toBe(false);
  });
});

/**
 * A stub where each path carries its own timestamp, so a single source can be
 * aged out while the rest stay current. `navigation.headingTrue` is absent
 * throughout, which is the case that matters: with no true heading the read of
 * it reports `stale: false`, and a fallback path that is dropped for age has to
 * raise the flag itself.
 */
function mixedAge(aged: Record<string, 'old'>): ServerApp {
  const old = new Date(Date.now() - 60_000).toISOString();
  const fresh = new Date().toISOString();
  const values: Record<string, unknown> = {
    'navigation.position': position,
    'navigation.speedOverGround': 3,
    'navigation.courseOverGroundTrue': deg2rad(120),
    'navigation.headingMagnetic': deg2rad(80),
    'navigation.magneticVariation': deg2rad(8),
  };
  return {
    getSelfPath: (path: string) =>
      path in values
        ? { value: values[path], timestamp: aged[path] === 'old' ? old : fresh }
        : undefined,
    getPath: () => undefined,
    handleMessage: () => undefined,
    debug: () => undefined,
    error: () => undefined,
  } as unknown as ServerApp;
}
