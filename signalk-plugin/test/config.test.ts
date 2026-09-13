// withDefaults must make a bad saved setting harmless — especially on the
// safety paths, where a silently-disabled alert is worse than a wrong number.

import { describe, it, expect } from 'vitest';
import { withDefaults } from '../src/config';

describe('withDefaults clamping', () => {
  const defaults = withDefaults(undefined);

  it('keeps a non-numeric MOB persistence from disabling man-overboard', () => {
    // `count >= NaN` is always false, so MOB would never fire — no error, no log.
    const cfg = withDefaults({ mobPersistFrames: 'three' as unknown as number });
    expect(cfg.mobPersistFrames).toBe(defaults.mobPersistFrames);
  });

  it('requires at least one frame of MOB persistence', () => {
    expect(withDefaults({ mobPersistFrames: 0 }).mobPersistFrames)
      .toBe(defaults.mobPersistFrames);
    expect(withDefaults({ mobPersistFrames: -5 }).mobPersistFrames)
      .toBe(defaults.mobPersistFrames);
  });

  it('rounds a fractional frame count to a whole frame', () => {
    expect(withDefaults({ mobPersistFrames: 2.6 }).mobPersistFrames).toBe(3);
  });

  it('keeps a valid MOB persistence as given', () => {
    expect(withDefaults({ mobPersistFrames: 5 }).mobPersistFrames).toBe(5);
  });

  it('clamps out-of-range MOB confidence and negative collision thresholds', () => {
    expect(withDefaults({ mobMinConfidence: 1.5 }).mobMinConfidence)
      .toBe(defaults.mobMinConfidence);
    expect(withDefaults({ collisionTcpaS: -1 }).collisionTcpaS).toBe(defaults.collisionTcpaS);
    expect(withDefaults({ collisionAlarmTcpaS: NaN }).collisionAlarmTcpaS)
      .toBe(defaults.collisionAlarmTcpaS);
  });
});
