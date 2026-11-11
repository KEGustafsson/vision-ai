import { describe, it, expect } from 'vitest';
import { ContainerSessionWatch } from '../src/containerClient';

describe('ContainerSessionWatch', () => {
  it('records the first session without reporting a restart', () => {
    const w = new ContainerSessionWatch();
    expect(w.update(120, 1_000_000)).toBe(false);
  });

  it('sees the same session while uptime keeps pace with the clock', () => {
    const w = new ContainerSessionWatch();
    w.update(120, 1_000_000);
    // 5 s later, uptime 5 s more; a few seconds of poll jitter either way.
    expect(w.update(125, 1_005_000)).toBe(false);
    expect(w.update(128, 1_010_000)).toBe(false);
    expect(w.update(140, 1_015_000)).toBe(false);
  });

  it('reports a restart once when uptime starts over', () => {
    const w = new ContainerSessionWatch();
    w.update(3600, 10_000_000);
    expect(w.update(4, 10_060_000)).toBe(true);
    // Then follows the new session.
    expect(w.update(9, 10_065_000)).toBe(false);
  });

  it('ignores a container that reports no uptime', () => {
    const w = new ContainerSessionWatch();
    w.update(3600, 10_000_000);
    expect(w.update(undefined, 10_060_000)).toBe(false);
    expect(w.update(Number.NaN, 10_060_000)).toBe(false);
    expect(w.update(-1, 10_060_000)).toBe(false);
  });

  it('forgets the session on reset', () => {
    const w = new ContainerSessionWatch();
    w.update(3600, 10_000_000);
    w.reset();
    expect(w.update(4, 10_060_000)).toBe(false);
  });
});
