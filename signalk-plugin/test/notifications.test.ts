import { describe, it, expect, beforeEach } from 'vitest';
import { NotificationManager } from '../src/notifications';
import { withDefaults } from '../src/config';
import { Delta, ServerApp } from '../src/skapp';

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
