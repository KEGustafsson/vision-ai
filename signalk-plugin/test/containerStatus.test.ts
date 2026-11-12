import { describe, expect, it } from 'vitest';
import { ContainerHealth, containerDownMessage, describeError } from '../src/containerStatus';

describe('ContainerHealth', () => {
  it('starts as not checked yet', () => {
    const s = new ContainerHealth('http://localhost:7000').snapshot();
    expect(s).toEqual({
      url: 'http://localhost:7000',
      reachable: null,
      everConnected: false,
      downSince: null,
      downForS: 0,
      lastError: null,
    });
  });

  it('dates an outage from its first failed check, not the latest', () => {
    const h = new ContainerHealth('http://x');
    h.markDown(new Error('fetch failed'), 1_000_000);
    h.markDown(new Error('fetch failed'), 1_005_000);
    const s = h.snapshot(1_031_900);
    expect(s.reachable).toBe(false);
    expect(s.downSince).toBe(new Date(1_000_000).toISOString());
    expect(s.downForS).toBe(31);
    expect(s.everConnected).toBe(false);
  });

  it('remembers that it was connected once, and starts a new outage after recovery', () => {
    const h = new ContainerHealth('http://x');
    h.markDown('boom', 1_000);
    h.markUp();
    let s = h.snapshot(2_000);
    expect(s).toMatchObject({ reachable: true, everConnected: true, downSince: null, downForS: 0, lastError: null });
    h.markDown('again', 50_000);
    s = h.snapshot(60_000);
    expect(s).toMatchObject({ reachable: false, everConnected: true, downForS: 10, lastError: 'again' });
  });
});

describe('describeError', () => {
  it("prefers the cause's code that Node's fetch hides behind 'fetch failed'", () => {
    const e = Object.assign(new TypeError('fetch failed'), { cause: { code: 'ECONNREFUSED', message: 'connect ECONNREFUSED' } });
    expect(describeError(e)).toBe('ECONNREFUSED');
  });
  it('falls back to the cause message, then the error message, then String()', () => {
    expect(describeError(Object.assign(new Error('outer'), { cause: { message: 'inner' } }))).toBe('inner');
    expect(describeError(new Error('outer'))).toBe('outer');
    expect(describeError('plain')).toBe('plain');
  });
});

describe('containerDownMessage', () => {
  it('names the URL, what is lost, and where the setup steps are', () => {
    const m = containerDownMessage('http://localhost:7000/');
    expect(m).toContain('http://localhost:7000/');
    expect(m).toContain('No camera detections, video or camera-based alarms');
    expect(m).toContain('Vision-AI webapp');
  });
});
