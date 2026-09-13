// EventStream keepalive + lifecycle against a real local WebSocket server: a
// half-open link (no pongs, no data) must be torn down and re-dialled, and
// stop() must cancel a pending reconnect so a restarted plugin can't be raced
// by a zombie socket.

import { afterEach, describe, expect, it } from 'vitest';
import { AddressInfo } from 'net';
import { WebSocketServer } from 'ws';
import { EventStream } from '../src/eventStream';
import { DetectionEvent } from '../src/types';

const log = { debug: () => undefined, error: () => undefined };

function until(cond: () => boolean, timeoutMs = 3000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (cond()) return resolve();
      if (Date.now() > deadline) return reject(new Error('condition not met in time'));
      setTimeout(tick, 5);
    };
    tick();
  });
}

function sampleEvent(): Record<string, unknown> {
  return {
    schema_version: '1.0',
    camera: 'forward',
    timestamp: new Date().toISOString(),
    frame_seq: 1,
    frame_size: { w: 1280, h: 960 },
    horizon_y: 350,
    inference: { backend: 'mock', latency_ms: 1 },
    calibration_status: 'auto',
    targets: [],
  };
}

describe('EventStream', () => {
  const servers: WebSocketServer[] = [];
  const streams: EventStream[] = [];

  afterEach(async () => {
    for (const s of streams) s.stop();
    for (const wss of servers) {
      for (const c of wss.clients) c.terminate();
      await new Promise<void>((r) => wss.close(() => r()));
    }
    streams.length = 0;
    servers.length = 0;
  });

  async function serve(
    opts: { autoPong?: boolean } = {}
  ): Promise<{ wss: WebSocketServer; url: string }> {
    const wss = new WebSocketServer({ port: 0, host: '127.0.0.1', ...opts });
    servers.push(wss);
    await new Promise<void>((r) => wss.once('listening', r));
    const { port } = wss.address() as AddressInfo;
    return { wss, url: `ws://127.0.0.1:${port}/ws/events` };
  }

  it('delivers events and pings the container on the keepalive interval', async () => {
    const { wss, url } = await serve();
    const got: DetectionEvent[] = [];
    let pings = 0;
    wss.on('connection', (sock) => {
      sock.on('ping', () => pings++);
      sock.send(JSON.stringify(sampleEvent()));
    });
    const stream = new EventStream(url, (ev) => got.push(ev), log, undefined, () => 0, undefined, {
      pingIntervalMs: 30,
      pongTimeoutMs: 200,
    });
    streams.push(stream);
    stream.start();
    await until(() => got.length === 1 && pings >= 2);
    expect(got[0].camera).toBe('forward');
    // Pongs come back, so the healthy link is never torn down.
    expect(wss.clients.size).toBe(1);
  });

  it('re-dials a half-open link that answers neither pongs nor data', async () => {
    // autoPong=false makes the server swallow pings — the client sees a link
    // that is up at the TCP level but dead at the application level.
    const { wss, url } = await serve({ autoPong: false });
    let connections = 0;
    wss.on('connection', () => connections++);
    const stream = new EventStream(url, () => undefined, log, undefined, () => 0, undefined, {
      pingIntervalMs: 30,
      pongTimeoutMs: 60,
    });
    streams.push(stream);
    stream.start();
    // First dial, then a keepalive-triggered terminate + reconnect (backoff 1 s).
    await until(() => connections >= 2, 4000);
    expect(connections).toBeGreaterThanOrEqual(2);
  });

  it('stop() cancels a pending reconnect', async () => {
    const { wss, url } = await serve();
    let connections = 0;
    wss.on('connection', (sock) => {
      connections++;
      sock.close(); // server drops us -> client schedules a reconnect
    });
    const stream = new EventStream(url, () => undefined, log);
    streams.push(stream);
    stream.start();
    await until(() => connections === 1);
    stream.stop();
    await new Promise((r) => setTimeout(r, 1200)); // past the 1 s first backoff
    expect(connections).toBe(1);
  });
});
