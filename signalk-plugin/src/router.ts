// registerWithRouter: same-origin REST surface for the captain webapp plus a
// reverse-proxy to the container's MJPEG/snapshot endpoints (so the browser
// uses SignalK's authenticated origin and never touches the container directly).

import http from 'http';
import https from 'https';
import { ContainerClient, ControlBody } from './containerClient';
import { PluginConfig } from './config';
import { EnrichedTarget, OwnShip } from './types';

// Only forward known control fields to the container — never relay an arbitrary
// client body verbatim. The container validates types/ranges too, but the plugin
// must not be a blind passthrough to a looser/older container.
const CONTROL_KEYS: (keyof ControlBody)[] = [
  'active_camera', 'confidence', 'max_targets',
  'min_target_range_m', 'mode_hint', 'labels', 'enabled',
];

function pickControl(body: unknown): ControlBody {
  const out: ControlBody = {};
  if (body && typeof body === 'object') {
    for (const k of CONTROL_KEYS) {
      const v = (body as Record<string, unknown>)[k];
      if (v !== undefined) (out as Record<string, unknown>)[k] = v;
    }
  }
  return out;
}

// Strip any embedded credentials (user:pass@) from a URL before returning it to
// the webapp. containerUrl accepts arbitrary http(s) URLs, so it can carry
// credentials; the webapp reaches the container only through this plugin's
// same-origin proxy routes and never needs them. Mirrors the vision service,
// which redacts camera RTSP URLs from its own /config.
function redactUrl(url: string): string {
  try {
    const u = new URL(url);
    u.username = '';
    u.password = '';
    return u.toString();
  } catch {
    // Fail closed: if it doesn't parse as a URL, still strip any `user:pass@`
    // userinfo rather than returning the raw value (which could leak credentials).
    return url.replace(/^(\w+:\/\/)[^/?#@]*@/, '$1');
  }
}

export interface SharedState {
  readonly targets: EnrichedTarget[];
  readonly ownShip: OwnShip;
  readonly system: {
    activeCamera: string;
    cameras: string[];
    detectionEnabled: boolean;
    maxTargets: number;
  };
  setDetection: (on: boolean) => void;
  setMaxTargets: (value: number) => void;
  client: () => ContainerClient | null;
}

function proxy(targetUrl: string, res: any): void {
  const mod = targetUrl.startsWith('https') ? https : http;
  const upstream = mod.get(targetUrl, (up) => {
    res.writeHead(up.statusCode || 502, {
      'content-type': up.headers['content-type'] || 'application/octet-stream',
      'cache-control': 'no-store, no-cache, must-revalidate, max-age=0',
      pragma: 'no-cache',
      'x-accel-buffering': 'no',
    });
    up.pipe(res);
  });
  // Don't let a slow/hung container hold the connection open forever. MJPEG is
  // a long-lived stream, so only guard the time-to-first-byte (the response
  // headers), not the streaming body.
  upstream.setTimeout(10000, () => upstream.destroy(new Error('upstream timeout')));
  upstream.on('error', () => {
    if (!res.headersSent) res.status(502).json({ error: 'vision container unreachable' });
    else res.end();
  });
  res.on('close', () => upstream.destroy());
}

// Reverse-proxy a long-lived MJPEG stream, transparently reconnecting to the
// container when its connection drops. A browser <img> never reconnects an
// MJPEG stream on its own — and a closed multipart stream fires neither `error`
// nor `load` — so on a container restart the frame just freezes until a manual
// reload. Instead we keep the *browser's* connection open and re-dial the
// container behind it: the client sees a brief pause, then frames resume.
// Upstream statuses worth retrying — the container is up but briefly not ready
// (booting, restarting). Any other non-2xx (401/403/404, ...) is permanent and
// is surfaced to the browser instead of retried.
const RETRYABLE_STATUS = new Set([500, 502, 503, 504]);

// The splice between the dying and the fresh upstream MUST land on a part
// boundary. A container that is killed mid-frame leaves the browser's multipart
// parser waiting for the rest of a truncated Content-Length'd JPEG; raw bytes
// from the new connection are then swallowed as that frame's remainder and
// (verified in Chromium) the parser never resyncs — the video freezes for good,
// with no `error` event to recover on. So instead of piping raw bytes we
// re-frame the upstream through MjpegPartAligner: buffer each multipart part,
// forward it only once complete, and DROP any truncated tail when the upstream
// dies. The browser then only ever sees whole parts, and a reconnect is
// indistinguishable from a slow frame.
const MAX_PART_BYTES = 32 * 1024 * 1024; // way above any real JPEG frame

/** Boundary token of a multipart/x-mixed-replace content-type ("frame" if absent). */
export function multipartBoundary(contentType: string | undefined): string {
  const m = /boundary="?([^";\s]+)"?/i.exec(contentType || '');
  return m ? m[1] : 'frame';
}

/**
 * Re-frames an MJPEG byte stream into complete multipart parts.
 *
 * Feed it raw upstream chunks; it returns only whole parts (boundary line +
 * headers + full body), re-emitted under `clientBoundary` — the boundary the
 * browser was promised in the initial response headers — so a restarted
 * container with a different boundary still splices cleanly. Anything buffered
 * when the upstream dies is simply discarded with the instance (one aligner per
 * upstream connection), which is exactly the mid-part-truncation fix.
 */
export class MjpegPartAligner {
  private buf: Buffer = Buffer.alloc(0);
  private readonly marker: Buffer;
  private readonly clientMarker: Buffer;

  constructor(upstreamBoundary: string, clientBoundary: string) {
    this.marker = Buffer.from(`--${upstreamBoundary}`);
    this.clientMarker = Buffer.from(`--${clientBoundary}`);
  }

  /**
   * Append upstream bytes and return every part that is now complete.
   * Throws if no part completes within MAX_PART_BYTES (malformed upstream);
   * the caller should drop the connection and let the retry loop re-dial.
   */
  push(chunk: Buffer): Buffer[] {
    this.buf = this.buf.length ? Buffer.concat([this.buf, chunk]) : chunk;
    const out: Buffer[] = [];
    for (;;) {
      const start = this.buf.indexOf(this.marker);
      if (start < 0) {
        // No boundary yet: keep only a tail that could still be a marker prefix
        // (start-of-stream preamble or a stray CRLF between parts is dropped).
        if (this.buf.length > this.marker.length) {
          this.buf = this.buf.subarray(this.buf.length - this.marker.length);
        }
        break;
      }
      if (start > 0) this.buf = this.buf.subarray(start); // trim preamble/CRLF
      const headerEnd = this.buf.indexOf('\r\n\r\n');
      if (headerEnd < 0) {
        if (this.buf.length > MAX_PART_BYTES) throw new Error('mjpeg part header too large');
        break; // headers still arriving
      }
      const headers = this.buf.subarray(0, headerEnd).toString('latin1');
      const lenMatch = /content-length:\s*(\d+)/i.exec(headers);
      let partEnd: number;
      if (lenMatch) {
        partEnd = headerEnd + 4 + Number(lenMatch[1]);
        if (this.buf.length < partEnd) {
          if (partEnd > MAX_PART_BYTES) throw new Error('mjpeg part too large');
          break; // body still arriving
        }
      } else {
        // No Content-Length: the part runs to the next boundary marker.
        const next = this.buf.indexOf(this.marker, headerEnd + 4);
        if (next < 0) {
          if (this.buf.length > MAX_PART_BYTES) throw new Error('mjpeg part too large');
          break;
        }
        partEnd = next;
      }
      // Re-emit under the browser's boundary, and always terminate with CRLF
      // (a source CRLF left in the buffer is trimmed as preamble next round).
      out.push(Buffer.concat([
        this.clientMarker,
        this.buf.subarray(this.marker.length, partEnd),
        Buffer.from('\r\n'),
      ]));
      this.buf = this.buf.subarray(partEnd);
    }
    return out;
  }
}

function proxyStream(targetUrl: string, res: any): void {
  const mod = targetUrl.startsWith('https') ? https : http;
  let closed = false;
  let headersWritten = false;
  // Boundary token promised to the browser in the initial response headers;
  // every later upstream's parts are re-emitted under it (see MjpegPartAligner).
  let clientBoundary: string | null = null;
  let current: any = null;
  let retrying = false;
  // The in-flight request, so a browser disconnect can cancel it before its
  // response callback fires and writes onto an already-closed response.
  let req: http.ClientRequest | null = null;

  const scheduleRetry = () => {
    if (closed || retrying) return;
    retrying = true;
    setTimeout(() => {
      retrying = false;
      connect();
    }, 1000);
  };

  const connect = () => {
    if (closed) return;
    const r = mod.get(targetUrl, (up) => {
      req = null; // response received; nothing left to cancel
      current = up;
      const status = up.statusCode || 0;
      if (status < 200 || status >= 300) {
        if (RETRYABLE_STATUS.has(status)) {
          up.resume(); // drain and retry (container booting/restarting)
          scheduleRetry();
        } else {
          // Permanent error (auth/not-found/...): stop retrying and forward it.
          closed = true;
          if (!headersWritten) {
            res.writeHead(status, { 'content-type': up.headers['content-type'] || 'application/json' });
            up.pipe(res); // forward the error body and end the response
          } else {
            up.resume();
            res.end();
          }
        }
        return;
      }
      const upBoundary = multipartBoundary(up.headers['content-type']);
      if (!headersWritten) {
        clientBoundary = upBoundary;
        res.writeHead(up.statusCode, {
          'content-type': up.headers['content-type'] || 'multipart/x-mixed-replace',
          'cache-control': 'no-store, no-cache, must-revalidate, max-age=0',
          pragma: 'no-cache',
          'x-accel-buffering': 'no',
        });
        headersWritten = true;
      }
      // Forward whole parts only (never a truncated one), applying backpressure
      // manually since we no longer pipe. The aligner is per-connection, so a
      // partial part buffered when this upstream dies is discarded with it.
      const aligner = new MjpegPartAligner(upBoundary, clientBoundary || upBoundary);
      up.on('data', (chunk: Buffer) => {
        if (closed || up !== current) return; // stale upstream must not write
        let parts: Buffer[];
        try {
          parts = aligner.push(chunk);
        } catch {
          up.destroy(); // malformed/oversized part: re-dial via onTerminal
          return;
        }
        for (const part of parts) {
          if (!res.write(part)) {
            up.pause();
            res.once('drain', () => up.resume());
          }
        }
      });
      // One terminal event per upstream, and ignore a stale upstream once a
      // newer one is live, so two upstreams can never write into res at once.
      let done = false;
      const onTerminal = () => {
        if (done || up !== current) return;
        done = true;
        scheduleRetry();
      };
      up.on('end', onTerminal);
      up.on('error', onTerminal);
      up.on('close', onTerminal); // belt-and-braces: some teardowns skip end/error
    });
    req = r;
    // Guard only the time-to-first-byte; a healthy stream is long-lived.
    r.setTimeout(10000, () => r.destroy(new Error('upstream timeout')));
    r.on('error', scheduleRetry);
  };

  res.on('close', () => {
    closed = true;
    if (req) req.destroy();
    if (current) current.destroy();
  });
  connect();
}

// Minimal JSON body parser: SignalK mounts plugin routers without guaranteeing
// a JSON body parser, so populate req.body ourselves when it's absent.
function ensureJsonBody(req: any, _res: any, next: () => void): void {
  if (req.body !== undefined || req.method === 'GET' || req.method === 'HEAD') return next();
  let data = '';
  req.on('data', (c: Buffer) => {
    data += c;
    if (data.length > 1e6) req.destroy(); // guard against oversized bodies
  });
  req.on('end', () => {
    try {
      req.body = data ? JSON.parse(data) : {};
    } catch {
      req.body = {};
    }
    next();
  });
  req.on('error', () => next());
}

export function registerRoutes(
  router: any,
  shared: SharedState,
  getCfg: () => PluginConfig
): void {
  router.use(ensureJsonBody);

  const knownCamera = (name: string): boolean =>
    shared.system.cameras.includes(name) || name === 'forward' || name === 'aft';

  router.get('/targets', (_req: any, res: any) => {
    // Live snapshot — never cache it. Without this Express adds a weak ETag and
    // no Cache-Control, so a browser can serve a stale/304 body and the UI
    // appears frozen (own-ship/targets stop updating).
    res.set('Cache-Control', 'no-store');
    res.json({
      ownShip: shared.ownShip,
      system: shared.system,
      targets: shared.targets,
    });
  });

  router.get('/config', (_req: any, res: any) => {
    const c = getCfg();
    res.json({
      containerUrl: redactUrl(c.containerUrl),
      features: {
        visualRadar: c.enableVisualRadar,
        aisFusion: c.enableAisFusion,
        collision: c.enableCollision,
        mob: c.enableMob,
        notifyCollision: c.notifyCollision,
        notifyDarkTarget: c.notifyDarkTarget,
      },
      cameras: shared.system.cameras,
      activeCamera: shared.system.activeCamera,
      detectionEnabled: shared.system.detectionEnabled,
      maxTargets: shared.system.maxTargets,
    });
  });

  // Master on/off for detection. GET reports the live state; POST flips it and
  // pushes the change to the container immediately.
  router.get('/detection', (_req: any, res: any) => {
    res.set('Cache-Control', 'no-store');
    res.json({ enabled: shared.system.detectionEnabled });
  });

  router.post('/detection', (req: any, res: any) => {
    const enabled = req.body?.enabled;
    if (typeof enabled !== 'boolean') {
      return res.status(400).json({ error: 'enabled must be a boolean' });
    }
    shared.setDetection(enabled);
    res.json({ enabled });
  });

  router.post('/target-limit', (req: any, res: any) => {
    const maxTargets = Number(req.body?.maxTargets);
    if (!Number.isInteger(maxTargets) || maxTargets < 1 || maxTargets > 300) {
      return res.status(400).json({ error: 'maxTargets must be an integer from 1 to 300' });
    }
    shared.setMaxTargets(maxTargets);
    res.json({ maxTargets });
  });

  router.post('/control', async (req: any, res: any) => {
    const client = shared.client();
    if (!client) return res.status(503).json({ error: 'not started' });
    try {
      const result = await client.control(pickControl(req.body));
      res.json(result);
    } catch (e) {
      res.status(502).json({ error: String(e) });
    }
  });

  // List of PTZ-capable cameras (so the webapp shows the control pad only when
  // the active camera actually supports PTZ).
  router.get('/ptz', (_req: any, res: any) => {
    const client = shared.client();
    if (!client) return res.status(503).json({ error: 'not started' });
    proxy(client.ptzListUrl(), res);
  });

  router.post('/ptz/:camera', async (req: any, res: any) => {
    const client = shared.client();
    if (!client) return res.status(503).json({ error: 'not started' });
    if (!knownCamera(req.params.camera)) return res.status(404).json({ error: 'unknown camera' });
    try {
      res.json(await client.ptz(req.params.camera, req.body || {}));
    } catch (e) {
      res.status(502).json({ error: String(e) });
    }
  });

  router.get('/stream/:camera', (req: any, res: any) => {
    const client = shared.client();
    if (!client) return res.status(503).json({ error: 'not started' });
    if (!knownCamera(req.params.camera)) return res.status(404).json({ error: 'unknown camera' });
    proxyStream(client.streamUrl(req.params.camera), res);
  });

  router.get('/snapshot/:camera', (req: any, res: any) => {
    const client = shared.client();
    if (!client) return res.status(503).json({ error: 'not started' });
    if (!knownCamera(req.params.camera)) return res.status(404).json({ error: 'unknown camera' });
    proxy(client.snapshotUrl(req.params.camera), res);
  });
}
