import { describe, it, expect } from 'vitest';
import http from 'http';
import { AddressInfo } from 'net';
import { MjpegPartAligner, multipartBoundary, registerRoutes, SharedState } from '../src/router';

// ---- helpers ---------------------------------------------------------------

function part(boundary: string, body: Buffer | string, withLength = true): Buffer {
  const b = Buffer.from(body);
  const headers =
    `--${boundary}\r\nContent-Type: image/jpeg\r\n` +
    (withLength ? `Content-Length: ${b.length}\r\n` : '') +
    '\r\n';
  return Buffer.concat([Buffer.from(headers), b, Buffer.from('\r\n')]);
}

/** Parse a multipart byte stream strictly (as a browser would): every part must
 * start at a boundary and carry its declared Content-Length. Returns the bodies
 * of the complete parts; throws on interior desync (bytes after a completed
 * part that aren't a boundary — the mid-part-splice bug this suite guards
 * against). A part still in flight at the TAIL of the stream is normal TCP
 * chunking on a live connection and simply isn't returned yet. */
function parseStrict(stream: Buffer, boundary: string): string[] {
  const marker = `--${boundary}`;
  const bodies: string[] = [];
  let off = 0;
  while (off < stream.length) {
    // skip inter-part CRLFs
    while (stream[off] === 0x0d || stream[off] === 0x0a) off++;
    if (off >= stream.length) break;
    const head = stream.subarray(off, off + marker.length).toString();
    if (head !== marker) {
      if (marker.startsWith(head)) break; // boundary itself still in flight
      throw new Error(`expected boundary at ${off}`);
    }
    const headerEnd = stream.indexOf('\r\n\r\n', off);
    if (headerEnd < 0) break; // trailing part headers still in flight
    const headers = stream.subarray(off, headerEnd).toString();
    const m = /content-length:\s*(\d+)/i.exec(headers);
    if (!m) throw new Error('part without Content-Length');
    const bodyStart = headerEnd + 4;
    const bodyEnd = bodyStart + Number(m[1]);
    if (bodyEnd > stream.length) break; // trailing part body still in flight
    bodies.push(stream.subarray(bodyStart, bodyEnd).toString());
    off = bodyEnd;
  }
  return bodies;
}

describe('multipartBoundary', () => {
  it('extracts the boundary token', () => {
    expect(multipartBoundary('multipart/x-mixed-replace; boundary=frame')).toBe('frame');
    expect(multipartBoundary('multipart/x-mixed-replace; boundary="quoted"')).toBe('quoted');
  });
  it('falls back to "frame" when absent', () => {
    expect(multipartBoundary(undefined)).toBe('frame');
    expect(multipartBoundary('text/plain')).toBe('frame');
  });
});

describe('MjpegPartAligner', () => {
  it('forwards complete parts and reassembles arbitrarily split chunks', () => {
    const a = new MjpegPartAligner('frame', 'frame');
    const input = Buffer.concat([part('frame', 'AAAA'), part('frame', 'BBBBBB')]);
    const out: Buffer[] = [];
    // Feed one byte at a time — worst-case TCP fragmentation.
    for (let i = 0; i < input.length; i++) out.push(...a.push(input.subarray(i, i + 1)));
    const bodies = parseStrict(Buffer.concat(out), 'frame');
    expect(bodies).toEqual(['AAAA', 'BBBBBB']);
  });

  it('never emits a truncated part (container killed mid-frame)', () => {
    const a = new MjpegPartAligner('frame', 'frame');
    const whole = part('frame', 'GOODFRAME');
    const truncated = part('frame', 'DOOMEDFRAME').subarray(0, 20); // dies mid-part
    const out = [...a.push(whole), ...a.push(truncated)];
    // The truncated tail stays buffered and dies with the aligner; a fresh
    // aligner (new upstream connection) then continues cleanly.
    const b = new MjpegPartAligner('frame', 'frame');
    out.push(...b.push(part('frame', 'AFTERRESTART')));
    const bodies = parseStrict(Buffer.concat(out), 'frame');
    expect(bodies).toEqual(['GOODFRAME', 'AFTERRESTART']);
  });

  it('re-emits parts under the client boundary when a restarted container uses a new one', () => {
    const a = new MjpegPartAligner('other', 'frame');
    const out = a.push(part('other', 'X'));
    expect(parseStrict(Buffer.concat(out), 'frame')).toEqual(['X']);
  });

  it('splits on the next boundary when Content-Length is missing', () => {
    const a = new MjpegPartAligner('frame', 'frame');
    const input = Buffer.concat([part('frame', 'NOLEN1', false), part('frame', 'WITHLEN')]);
    const out = a.push(input);
    expect(out.length).toBe(2);
    expect(out[0].toString()).toContain('NOLEN1');
    expect(out[1].toString()).toContain('WITHLEN');
  });

  it('throws when a part exceeds the buffer cap instead of buffering forever', () => {
    const a = new MjpegPartAligner('frame', 'frame');
    const head = Buffer.from(`--frame\r\nContent-Length: ${64 * 1024 * 1024}\r\n\r\n`);
    expect(() => {
      a.push(head);
      a.push(Buffer.alloc(1024));
    }).toThrow();
  });
});

// ---- proxyStream end-to-end: container restart must not stall the stream ----

type Handler = (req: any, res: any) => void;

function buildRouter(containerUrl: string): Map<string, Handler> {
  const routes = new Map<string, Handler>();
  const router = {
    use: () => {},
    get: (path: string, h: Handler) => routes.set(path, h),
    post: () => {},
  };
  const shared: SharedState = {
    targets: [],
    ownShip: {} as any,
    system: { activeCamera: 'forward', cameras: ['forward'], detectionEnabled: true, maxTargets: 20 },
    setDetection: () => {},
    setMaxTargets: () => {},
    client: () => ({ streamUrl: () => `${containerUrl}/stream/forward.mjpg` }) as any,
  };
  registerRoutes(router, shared, () => ({ containerUrl }) as any);
  return routes;
}

interface FakeContainer extends http.Server {
  /** Destroy all sockets mid-write — the exact failure mode of a container
   * being taken down while streaming (the last part is truncated). */
  killHard(): Promise<void>;
}

function startFakeContainer(port: number, tag: string): Promise<FakeContainer> {
  const sockets = new Set<any>();
  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'multipart/x-mixed-replace; boundary=frame' });
    const iv = setInterval(() => res.write(part('frame', `FRAME-${tag}`)), 25);
    res.on('close', () => clearInterval(iv));
  }) as FakeContainer;
  server.on('connection', (s) => {
    sockets.add(s);
    s.on('close', () => sockets.delete(s));
  });
  server.killHard = () =>
    new Promise<void>((resolve) => {
      server.close(() => resolve());
      for (const s of sockets) s.destroy();
      sockets.clear();
    });
  return new Promise((resolve) => server.listen(port, () => resolve(server)));
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Poll until the condition holds (deterministic under CI load, unlike a fixed
 * sleep) or the timeout elapses — assertions after it then report the miss. */
async function waitFor(cond: () => boolean, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!cond() && Date.now() < deadline) await sleep(25);
}

describe('proxyStream across a container restart', () => {
  it('keeps the browser connection open and resumes with only whole parts', async () => {
    // Ports are OS-assigned (listen on 0) so parallel suites can't collide.
    let container = await startFakeContainer(0, 'GEN1');
    const containerPort = (container.address() as AddressInfo).port;
    const routes = buildRouter(`http://127.0.0.1:${containerPort}`);

    const plugin = http.createServer((req, res) => {
      (res as any).set = (k: string, v: string) => res.setHeader(k, v);
      (res as any).status = (c: number) => ((res.statusCode = c), res);
      (res as any).json = (o: unknown) => res.end(JSON.stringify(o));
      (req as any).params = { camera: 'forward' };
      routes.get('/stream/:camera')!(req, res);
    });
    await new Promise<void>((r) => plugin.listen(0, () => r()));
    const pluginPort = (plugin.address() as AddressInfo).port;

    const received: Buffer[] = [];
    let ended = false;
    const frames = (tag: string) =>
      parseStrict(Buffer.concat(received), 'frame').filter((b) => b === `FRAME-${tag}`).length;
    const clientReq = http.get(`http://127.0.0.1:${pluginPort}/stream/forward`, (res) => {
      res.on('data', (c: Buffer) => received.push(c));
      res.on('end', () => (ended = true));
      res.on('error', () => (ended = true));
    });

    try {
      await waitFor(() => frames('GEN1') > 2, 5000);
      const beforeKill = parseStrict(Buffer.concat(received), 'frame');
      expect(beforeKill.length).toBeGreaterThan(2);
      expect(beforeKill.every((b) => b === 'FRAME-GEN1')).toBe(true);

      await container.killHard();
      await sleep(200); // give a wrongly-ended response time to surface
      expect(ended).toBe(false); // browser connection must stay open

      // Restart on the SAME port (the proxy keeps dialing the original URL).
      container = await startFakeContainer(containerPort, 'GEN2');
      await waitFor(() => frames('GEN2') > 2, 10000); // retry loop re-dials within ~1s

      // parseStrict throws if ANY forwarded part is truncated — this is the
      // regression this test exists for (a mid-part splice permanently desyncs
      // the browser's multipart parser with no error event to recover on).
      expect(frames('GEN2')).toBeGreaterThan(2);
      expect(ended).toBe(false);
    } finally {
      clientReq.destroy();
      await container.killHard();
      await new Promise<void>((r) => plugin.close(() => r()));
    }
  }, 20000);
});
