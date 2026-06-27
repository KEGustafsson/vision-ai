// WebSocket consumer of the container's detection events, with reconnect
// backoff and JSON-Schema validation against the shared contract.

import { readFileSync } from 'fs';
import { join } from 'path';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import WebSocket from 'ws';
import { DetectionEvent } from './types';

type Logger = { debug: (m: string, ...a: unknown[]) => void; error: (m: string) => void };

// Wire-contract major version this plugin understands. The container stamps
// every event with `schema_version` (see vision-service/app/schemas.py); a major
// mismatch means breaking field changes, so we refuse those events and surface a
// notification rather than silently mis-interpreting them.
const SUPPORTED_SCHEMA_MAJOR = '1';
// Don't spam the log on a persistent bad-frame source; warn at most this often.
const WARN_INTERVAL_MS = 30000;

export class EventStream {
  private ws: WebSocket | null = null;
  private closed = false;
  private backoff = 1000;
  private validate: ValidateFunction | null = null;
  private lastValidationWarnAt = 0;
  private lastStaleWarnAt = 0;
  private mismatchedVersion: string | null = null;
  private staleActive = false;

  constructor(
    private wsUrl: string,
    private onEvent: (ev: DetectionEvent) => void,
    private log: Logger,
    // Raised once per distinct incompatible version seen (and cleared when a
    // compatible event arrives) so the plugin can notify the operator.
    private onVersionMismatch?: (version: string | null) => void,
    // Max accepted event age (ms); 0 disables the absolute check. A getter so a
    // changed setting is picked up without rebuilding the stream.
    private getMaxAgeMs: () => number = () => 0,
    // Raised when events start being rejected as stale (and cleared when fresh
    // ones resume) so a frozen/replayed feed surfaces instead of going silently
    // dark — dropping every event without telling anyone is itself a hazard.
    private onStaleEvents?: (stale: boolean) => void
  ) {
    this.loadSchema();
  }

  // Flip the stale state on transition only, notifying + logging once each way.
  private markStale(stale: boolean, ageMs?: number): void {
    if (stale) {
      const now = Date.now();
      if (now - this.lastStaleWarnAt > WARN_INTERVAL_MS) {
        this.lastStaleWarnAt = now;
        this.log.error(
          `vision-ai: dropping detection events as stale` +
          (ageMs !== undefined ? ` (age ${(ageMs / 1000).toFixed(1)}s)` : '') +
          ' — check container/SignalK clock sync and the network'
        );
      }
    }
    if (stale === this.staleActive) return;
    this.staleActive = stale;
    this.onStaleEvents?.(stale);
  }

  private loadSchema(): void {
    try {
      const schemaPath = join(__dirname, '..', 'schema', 'detection-event.schema.json');
      const schema = JSON.parse(readFileSync(schemaPath, 'utf-8'));
      const ajv = new Ajv({ allErrors: false, strict: false });
      addFormats(ajv);
      this.validate = ajv.compile(schema);
    } catch (e) {
      this.log.error(`vision-ai: could not load event schema, validation disabled: ${e}`);
    }
  }

  start(): void {
    this.closed = false;
    this.connect();
  }

  stop(): void {
    this.closed = true;
    if (this.ws) {
      this.ws.removeAllListeners();
      this.ws.close();
      this.ws = null;
    }
  }

  private connect(): void {
    if (this.closed) return;
    this.log.debug(`vision-ai: connecting to ${this.wsUrl}`);
    // Cap inbound frame size so a hostile/buggy container can't pressure memory
    // with an enormous frame (default ws limit is 100 MB).
    const ws = new WebSocket(this.wsUrl, { maxPayload: 4 * 1024 * 1024 });
    this.ws = ws;

    ws.on('open', () => {
      this.backoff = 1000;
      this.log.debug('vision-ai: event stream connected');
    });

    ws.on('message', (data: WebSocket.RawData) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(data.toString());
      } catch {
        return;
      }
      if (this.validate && !this.validate(parsed)) {
        const now = Date.now();
        if (now - this.lastValidationWarnAt > WARN_INTERVAL_MS) {
          this.lastValidationWarnAt = now;
          this.log.error(`vision-ai: event failed schema validation: ${JSON.stringify(this.validate.errors)}`);
        }
        return;
      }
      // Refuse an event whose major schema version we don't understand, and
      // notify once per distinct bad version. Clear the flag when a compatible
      // event arrives again.
      const version = (parsed as { schema_version?: unknown }).schema_version;
      const major = typeof version === 'string' ? version.split('.')[0] : null;
      if (major !== SUPPORTED_SCHEMA_MAJOR) {
        const seen = typeof version === 'string' ? version : 'unknown';
        if (this.mismatchedVersion !== seen) {
          this.mismatchedVersion = seen;
          this.log.error(`vision-ai: incompatible event schema_version ${seen} (need ${SUPPORTED_SCHEMA_MAJOR}.x)`);
          this.onVersionMismatch?.(seen);
        }
        return;
      }
      if (this.mismatchedVersion !== null) {
        this.mismatchedVersion = null;
        this.onVersionMismatch?.(null);
      }
      // Freshness gate: a delayed/buffered/replayed frame must not be treated as
      // live (it would feed a stale position into CPA/fusion history). Reject an
      // unparseable timestamp outright, and — when an age limit is set — anything
      // older than it. The per-camera out-of-order guard lives in the handler
      // (index.ts handleEvent), which holds the last-accepted time per camera.
      const ev = parsed as DetectionEvent;
      const ts = Date.parse(ev.timestamp);
      if (!Number.isFinite(ts)) {
        const now = Date.now();
        if (now - this.lastStaleWarnAt > WARN_INTERVAL_MS) {
          this.lastStaleWarnAt = now;
          this.log.error(`vision-ai: dropping event with invalid timestamp ${JSON.stringify(ev.timestamp)}`);
        }
        return;
      }
      const maxAgeMs = this.getMaxAgeMs();
      if (maxAgeMs > 0) {
        // Reject too-old AND future-dated frames: if the container clock runs
        // ahead of SignalK, a negative age would otherwise pass as "fresh" and
        // push lastEventTsByCamera/lastSeen into the future, starving real-time
        // frames and keeping tracks alive past their timeout.
        const ageMs = Date.now() - ts;
        if (Math.abs(ageMs) > maxAgeMs) {
          this.markStale(true, Math.abs(ageMs));
          return;
        }
      }
      this.markStale(false);
      // A downstream throw must not tear down the socket pipeline.
      try {
        this.onEvent(ev);
      } catch (e) {
        this.log.error(`vision-ai: event handler error: ${e}`);
      }
    });

    ws.on('close', () => this.scheduleReconnect());
    ws.on('error', (err) => {
      this.log.debug(`vision-ai: ws error ${err}`);
      ws.close();
    });
  }

  private scheduleReconnect(): void {
    if (this.closed) return;
    const delay = this.backoff;
    this.backoff = Math.min(this.backoff * 2, 30000);
    setTimeout(() => this.connect(), delay);
  }
}
