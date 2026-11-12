// Reachability of the vision container, as the webapp's setup guide needs it.
//
// The health poll (index.ts checkHealth) already raises and clears the
// notifications.vision.containerDown alarm on each transition. This keeps the
// same facts in a form the captain webapp can read on every /targets poll:
// whether the container answers now, whether it ever answered since the plugin
// started, and since when it has been silent. The webapp decides from these
// when to open the setup guide (see public/app.js), so a container restart or
// a TensorRT engine build does not flash the guide up for a few seconds.

export interface ContainerStatus {
  /** Base URL the plugin polls, credentials removed by the router. */
  url: string;
  /** null until the first health check has finished. */
  reachable: boolean | null;
  /** True once any health check has succeeded since the plugin started. */
  everConnected: boolean;
  /** ISO time of the first failed check of the current outage, else null. */
  downSince: string | null;
  /** Seconds since downSince, rounded down; 0 when reachable or unknown. */
  downForS: number;
  /** The last failure, e.g. "fetch failed", while down; else null. */
  lastError: string | null;
}

export class ContainerHealth {
  private reachable: boolean | null = null;
  private everConnected = false;
  private downSinceMs: number | null = null;
  private lastError: string | null = null;

  constructor(private readonly url: string) {}

  markUp(): void {
    this.reachable = true;
    this.everConnected = true;
    this.downSinceMs = null;
    this.lastError = null;
  }

  markDown(error: unknown, now: number = Date.now()): void {
    if (this.reachable !== false) this.downSinceMs = now;
    this.reachable = false;
    this.lastError = describeError(error);
  }

  snapshot(now: number = Date.now()): ContainerStatus {
    const downForS =
      this.reachable === false && this.downSinceMs !== null
        ? Math.max(0, Math.floor((now - this.downSinceMs) / 1000))
        : 0;
    return {
      url: this.url,
      reachable: this.reachable,
      everConnected: this.everConnected,
      downSince: this.downSinceMs !== null ? new Date(this.downSinceMs).toISOString() : null,
      downForS,
      lastError: this.lastError,
    };
  }
}

/**
 * A short, human-readable reason for a failed health check. Node's fetch
 * reports a refused or unresolvable host as "TypeError: fetch failed" with the
 * useful part in `cause`, so prefer the cause's code or message.
 */
export function describeError(error: unknown): string {
  return redactCredentials(rawDescription(error));
}

function rawDescription(error: unknown): string {
  if (error && typeof error === 'object') {
    const cause = (error as { cause?: unknown }).cause;
    if (cause && typeof cause === 'object') {
      const code = (cause as { code?: unknown }).code;
      if (typeof code === 'string' && code) return code;
      const msg = (cause as { message?: unknown }).message;
      if (typeof msg === 'string' && msg) return msg;
    }
    const msg = (error as { message?: unknown }).message;
    if (typeof msg === 'string' && msg) return msg;
  }
  return String(error);
}

/**
 * Remove `user:password@` from every URL in a piece of text. The container URL
 * may carry credentials, and Node's fetch rejects such a URL with a message
 * that repeats it in full ("Request cannot be constructed from a URL that
 * includes credentials: http://user:secret@…"), so an error description must
 * never be stored, served or logged unredacted.
 *
 * fetch repeats the URL as configured, unencoded, so the userinfo may hold
 * spaces or an `@`: everything up to the last `@` before the path, query,
 * fragment or end of line is removed. On a URL with no path that is followed
 * by prose containing an `@`, this removes more than the userinfo, which errs
 * on the side of hiding a password rather than showing one.
 */
export function redactCredentials(text: string): string {
  return text.replace(/([a-z][a-z0-9+.-]*:\/\/)[^/?#\r\n]*@/gi, '$1');
}

/** The text of the containerDown alarm and of the plugin's error status. */
export function containerDownMessage(url: string): string {
  return (
    `Vision container can't be reached at ${url}. ` +
    'No camera detections, video or camera-based alarms. ' +
    'Open the Vision-AI webapp for setup steps.'
  );
}
