// Own-ship heading comes from navigation.headingTrue and nothing else. A
// magnetic compass carries an installation deviation SignalK knows nothing
// about — on Arabella it read 33° off the GNSS true heading — and COG differs
// from heading by leeway and current. Either substitute would put every target
// in the wrong place while looking healthy; an unknown heading is safer.

import { describe, it, expect } from 'vitest';
import { readOwnShip } from '../src/nav';
import { deg2rad, rad2deg } from '../src/geo';
import { ServerApp } from '../src/skapp';

type Node = { value: unknown; timestamp: string };

/** A SignalK stub serving full-model nodes, each with its own timestamp. */
function app(paths: Record<string, Node>): ServerApp {
  return {
    getSelfPath: (path: string) => paths[path],
    getPath: () => undefined,
    handleMessage: () => undefined,
    debug: () => undefined,
    error: () => undefined,
  } as unknown as ServerApp;
}

const fresh = (value: unknown): Node => ({ value, timestamp: new Date().toISOString() });
const old = (value: unknown): Node => ({
  value,
  timestamp: new Date(Date.now() - 60_000).toISOString(),
});

const position = { latitude: 60.2785, longitude: 22.2866 };

// Arabella at the dock, 2026-09-13: GNSS compass 86.6° true; fluxgate 110.3°
// magnetic with 9.3° east variation, i.e. 119.6° — 33° off.
const measured = {
  'navigation.position': fresh(position),
  'navigation.speedOverGround': fresh(0.01),
  'navigation.courseOverGroundTrue': fresh(deg2rad(77.3)),
  'navigation.headingMagnetic': fresh(deg2rad(110.3)),
  'navigation.magneticVariation': fresh(deg2rad(9.3)),
};

describe('readOwnShip heading', () => {
  it('uses the true heading as published, whatever the magnetic compass says', () => {
    const own = readOwnShip(app({
      ...measured,
      'navigation.headingTrue': fresh(deg2rad(86.6)),
    }), 5);
    expect(rad2deg(own.headingTrue as number)).toBeCloseTo(86.6, 6);
    expect(own.stale).toBe(false);
  });

  it('never derives a heading from magnetic plus variation', () => {
    // Exactly the pair a fallback would have used, with no true heading at all.
    const own = readOwnShip(app(measured), 5);
    expect(own.headingTrue).toBeNull();
    // Nothing aged out — heading is simply not published — so not stale, and
    // everything that is published still comes through.
    expect(own.stale).toBe(false);
    expect(own.position).not.toBeNull();
    expect(own.sog).toBeCloseTo(0.01, 6);
  });

  it('never falls back to course over ground', () => {
    const own = readOwnShip(app({
      'navigation.position': fresh(position),
      'navigation.courseOverGroundTrue': fresh(deg2rad(120)),
    }), 5);
    expect(own.headingTrue).toBeNull();
  });

  it('drops an aged-out true heading instead of substituting a fresh magnetic one', () => {
    // The GNSS compass goes quiet while the fluxgate keeps talking: the result
    // must be "heading unknown because it went stale", not the compass reading.
    const own = readOwnShip(app({
      ...measured,
      'navigation.headingTrue': old(deg2rad(86.6)),
    }), 5);
    expect(own.headingTrue).toBeNull();
    expect(own.stale).toBe(true);
    expect(own.position).not.toBeNull();
  });
});
