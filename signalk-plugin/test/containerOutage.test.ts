// The whole plugin with no vision container behind it, then with one: the
// admin must show the outage as the plugin's error (not a green "0 targets"
// that reads as an empty sea), the webapp must see the container as down so
// it can open the setup guide, and both must recover when the container
// answers.

import { afterEach, describe, expect, it } from 'vitest';
import { AddressInfo } from 'net';
import { createServer, Server } from 'http';
import createPlugin from '../src/index';
import { Delta, Plugin, ServerApp } from '../src/skapp';

function until(cond: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (cond()) return resolve();
      if (Date.now() > deadline) return reject(new Error('condition not met in time'));
      setTimeout(tick, 20);
    };
    tick();
  });
}

class FakeApp implements ServerApp {
  deltas: Delta[] = [];
  statuses: string[] = [];
  errors: string[] = [];
  handleMessage(_id: string, delta: Delta): void { this.deltas.push(delta); }
  getSelfPath(): any { return undefined; }
  getPath(): any { return undefined; }
  debug(): void {}
  error(): void {}
  setPluginStatus(msg: string): void { this.statuses.push(msg); }
  setPluginError(msg: string): void { this.errors.push(msg); }
  notification(path: string): any {
    for (let i = this.deltas.length - 1; i >= 0; i--) {
      for (const u of this.deltas[i].updates) {
        for (const v of u.values ?? []) if (v.path === path) return v.value;
      }
    }
    return undefined;
  }
}

type Handler = (req: any, res: any) => void;

function routesOf(plugin: Plugin): Map<string, Handler> {
  const routes = new Map<string, Handler>();
  plugin.registerWithRouter!({ use: () => {}, get: (p: string, h: Handler) => routes.set(p, h), post: () => {} });
  return routes;
}

function getJson(routes: Map<string, Handler>, path: string): any {
  let body: any;
  const res = { set: () => res, json: (b: any) => { body = b; }, status: () => res };
  routes.get(path)!({}, res);
  return body;
}

async function freePort(): Promise<number> {
  const s = createServer();
  await new Promise<void>((r) => s.listen(0, '127.0.0.1', () => r()));
  const port = (s.address() as AddressInfo).port;
  await new Promise<void>((r) => s.close(() => r()));
  return port;
}

describe('plugin — vision container not running', () => {
  let plugin: Plugin | null = null;
  let container: Server | null = null;

  afterEach(async () => {
    plugin?.stop();
    plugin = null;
    await new Promise<void>((r) => (container ? container.close(() => r()) : r()));
    container = null;
  });

  it('reports the outage to the admin and the webapp, and recovers', async () => {
    const port = await freePort();
    const app = new FakeApp();
    plugin = createPlugin(app as any);
    const routes = routesOf(plugin);
    plugin.start({ containerUrl: `http://127.0.0.1:${port}`, processIntervalMs: 200 });

    // Down: the admin gets the plugin error, the alarm is raised, the webapp
    // sees an outage of a container that has never answered.
    await until(() => app.errors.length > 0, 3000);
    expect(app.errors[0]).toContain(`Vision container is not running at http://127.0.0.1:${port}/`);
    expect(app.notification('notifications.vision.containerDown')?.state).toBe('alarm');
    const down = getJson(routes, '/targets').container;
    expect(down).toMatchObject({ reachable: false, everConnected: false });
    expect(down.downSince).not.toBeNull();

    // While down, the process cycle must not paint the admin green again.
    const before = app.statuses.length;
    await new Promise((r) => setTimeout(r, 700));
    expect(app.statuses.slice(before).filter((s) => s.includes('targets'))).toEqual([]);

    // Up: a container starts answering /health on that port.
    container = createServer((req, res) => {
      res.writeHead(req.url === '/health' ? 200 : 404, { 'content-type': 'application/json' });
      res.end(req.url === '/health' ? JSON.stringify({ status: 'ok' }) : '{}');
    });
    await new Promise<void>((r) => container!.listen(port, '127.0.0.1', () => r()));

    // The health poll runs every 5 s.
    await until(() => getJson(routes, '/targets').container.reachable === true, 8000);
    expect(getJson(routes, '/targets').container).toMatchObject({ everConnected: true, downSince: null });
    expect(app.notification('notifications.vision.containerDown')).toBeNull();
    await until(() => app.statuses.some((s) => s.includes('targets')), 2000);
  }, 15000);

  it('never serves or reports the password of a container URL with credentials', async () => {
    const port = await freePort();
    const app = new FakeApp();
    plugin = createPlugin(app as any);
    const routes = routesOf(plugin);
    // Node's fetch rejects a URL with credentials, with a message repeating it.
    plugin.start({ containerUrl: `http://vision:s3cr3t-pw@127.0.0.1:${port}`, processIntervalMs: 200 });
    await until(() => getJson(routes, '/targets').container.reachable === false, 3000);
    const served = JSON.stringify([getJson(routes, '/targets'), getJson(routes, '/setup')]);
    expect(served).not.toContain('s3cr3t-pw');
    expect(getJson(routes, '/targets').container.lastError).toBeTruthy();
    expect(JSON.stringify(app.errors)).not.toContain('s3cr3t-pw');
    expect(JSON.stringify(app.deltas)).not.toContain('s3cr3t-pw');
  }, 10000);
});
