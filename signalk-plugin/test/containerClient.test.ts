// The control-plane client must stay bounded on a vessel network: a container
// that accepts a connection and then stalls cannot be allowed to hold a request
// open, and a request outstanding when the plugin stops must never land.

import { afterEach, describe, expect, it } from 'vitest';
import http from 'http';
import { AddressInfo } from 'net';
import { ContainerClient } from '../src/containerClient';

describe('ContainerClient', () => {
  const servers: http.Server[] = [];

  afterEach(async () => {
    for (const s of servers) await new Promise<void>((r) => s.close(() => r()));
    servers.length = 0;
  });

  function serve(handler: http.RequestListener): Promise<string> {
    const server = http.createServer(handler);
    servers.push(server);
    return new Promise((resolve) => {
      server.listen(0, '127.0.0.1', () => {
        const { port } = server.address() as AddressInfo;
        resolve(`http://127.0.0.1:${port}`);
      });
    });
  }

  it('returns the parsed body of a healthy response', async () => {
    const url = await serve((_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', mode: 'mock' }));
    });
    const client = new ContainerClient(url, 1000);
    await expect(client.health()).resolves.toMatchObject({ status: 'ok' });
  });

  it('gives up on a container that stalls mid-body', async () => {
    // Headers sent, body never finished — an unbounded fetch would wait for
    // undici's default header timeout, and the 5 s poll would pile up more.
    const url = await serve((_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json' });
      res.write('{"status":');
    });
    const client = new ContainerClient(url, 150);
    const started = Date.now();
    await expect(client.health()).rejects.toThrow();
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it('surfaces a non-2xx status', async () => {
    const url = await serve((_req, res) => {
      res.writeHead(503);
      res.end();
    });
    const client = new ContainerClient(url, 1000);
    await expect(client.health()).rejects.toThrow(/health 503/);
  });

  it('does not wait on an error response whose body never ends', async () => {
    // fetch resolves on headers; undici holds the connection until the body is
    // consumed or cancelled, so an unfinished error body would leak one
    // connection per poll.
    const url = await serve((_req, res) => {
      res.writeHead(503, { 'content-type': 'application/json' });
      res.write('{"detail":');
    });
    const client = new ContainerClient(url, 10000); // far longer than this should take
    const started = Date.now();
    await expect(client.health()).rejects.toThrow(/health 503/);
    expect(Date.now() - started).toBeLessThan(2000);
  });

  it('stamps every control with this instance\'s ordering token', async () => {
    const bodies: any[] = [];
    const url = await serve((req, res) => {
      let data = '';
      req.on('data', (c) => (data += c));
      req.on('end', () => {
        bodies.push(JSON.parse(data));
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end('{"applied":{}}');
      });
    });
    const client = new ContainerClient(url, 1000, 4242);
    await client.control({ enabled: true });
    await client.control({ enabled: false });
    // The container refuses a body from a superseded plugin instance by this
    // token, so every control must carry it — not just the first.
    expect(bodies.map((b) => b.client_generation)).toEqual([4242, 4242]);
    expect(bodies[0].enabled).toBe(true);
  });

  it('gives each plugin instance a token that rises across restarts', () => {
    const first = new ContainerClient('http://127.0.0.1:1');
    const second = new ContainerClient('http://127.0.0.1:1');
    expect((second as any).generation).toBeGreaterThanOrEqual((first as any).generation);
  });

  it('close() aborts an in-flight request and refuses new ones', async () => {
    let delivered = 0;
    const url = await serve((_req, res) => {
      delivered += 1;
      // Never answer: the request is only resolved by the abort.
      void res;
    });
    const client = new ContainerClient(url, 10000);
    const pending = client.control({ enabled: true });
    await new Promise((r) => setTimeout(r, 100));
    client.close();
    await expect(pending).rejects.toThrow();
    expect(delivered).toBe(1);
    await expect(client.control({ enabled: false })).rejects.toThrow(/client closed/);
    expect(delivered).toBe(1); // nothing new was sent after close()
  });
});
