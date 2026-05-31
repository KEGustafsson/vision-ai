// registerWithRouter: same-origin REST surface for the captain webapp plus a
// reverse-proxy to the container's MJPEG/snapshot endpoints (so the browser
// uses SignalK's authenticated origin and never touches the container directly).

import http from 'http';
import https from 'https';
import { ContainerClient } from './containerClient';
import { PluginConfig } from './config';
import { EnrichedTarget, OwnShip } from './types';

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
// container behind it: the client sees a brief pause, then frames resume. The
// multipart boundary is identical across reconnects, so the browser resyncs at
// the next `--frame` marker (at most one frame is glitched at the seam).
// Upstream statuses worth retrying — the container is up but briefly not ready
// (booting, restarting). Any other non-2xx (401/403/404, ...) is permanent and
// is surfaced to the browser instead of retried.
const RETRYABLE_STATUS = new Set([500, 502, 503, 504]);

function proxyStream(targetUrl: string, res: any): void {
  const mod = targetUrl.startsWith('https') ? https : http;
  let closed = false;
  let headersWritten = false;
  let current: any = null;
  let retrying = false;
  // The in-flight request, so a browser disconnect can cancel it before its
  // response callback fires and pipes onto an already-closed response.
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
      if (!headersWritten) {
        res.writeHead(up.statusCode, {
          'content-type': up.headers['content-type'] || 'multipart/x-mixed-replace',
          'cache-control': 'no-store, no-cache, must-revalidate, max-age=0',
          pragma: 'no-cache',
          'x-accel-buffering': 'no',
        });
        headersWritten = true;
      }
      up.pipe(res, { end: false }); // keep res open across reconnects
      // One terminal event per upstream, and ignore a stale upstream once a
      // newer one is live, so two upstreams can never pipe into res at once.
      let done = false;
      const onTerminal = () => {
        if (done || up !== current) return;
        done = true;
        scheduleRetry();
      };
      up.on('end', onTerminal);
      up.on('error', onTerminal);
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
      containerUrl: c.containerUrl,
      features: {
        visualRadar: c.enableVisualRadar,
        aisFusion: c.enableAisFusion,
        collision: c.enableCollision,
        mob: c.enableMob,
        notifyCollision: c.notifyCollision,
        notifyDarkTarget: c.notifyDarkTarget,
        aisBlips: c.enableAisBlips,
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
      const result = await client.control(req.body || {});
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
