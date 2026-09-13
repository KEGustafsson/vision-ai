// Thin REST client for controlling the vision container.

export interface HealthInfo {
  status: string; // "ok" | "degraded"
  mode: string;
  backend: string;
  cameras: string[];
  model?: string;
  model_labels?: string[];
  detection_enabled: boolean;
  labels: string[] | null;
  // Degradation detail (present on newer containers). camera_errors maps a
  // camera name to its stall/init message; pipeline_restarts is DeepStream's
  // auto-restart count and pipeline_last_error its last restart cause.
  camera_errors?: Record<string, string>;
  pipeline_restarts?: number;
  pipeline_last_error?: string | null;
}

export interface ControlBody {
  active_camera?: string;
  confidence?: number;
  max_targets?: number;
  min_target_range_m?: number; // drop detections closer than this (m) in the container; 0 => off
  mode_hint?: string;
  labels?: string[]; // canonical labels to surface (person | vessel | buoy | debris | kayak | log)
  enabled?: boolean; // master on/off: pause/resume detection in the container
}

export interface PtzBody {
  action?: 'move' | 'stop' | 'home';
  pan?: number;
  tilt?: number;
  zoom?: number;
}

// Bound every control-plane request. On a boat network a half-open link (the
// container host rebooting, a switch power-cycle) makes an unbounded fetch hang
// for the kernel's connect/read timeout — minutes — while the 5 s sync/health
// timers keep starting new ones, piling up pending requests for the whole
// outage. Shorter than the 5 s poll so at most one request per poll is in flight.
const REQUEST_TIMEOUT_MS = 4000;

export class ContainerClient {
  // Abort handles for the requests currently in flight, so close() can drop
  // them all (see close()).
  private inFlight = new Set<AbortController>();
  private closed = false;

  constructor(private baseUrl: string, private timeoutMs: number = REQUEST_TIMEOUT_MS) {}

  private url(path: string): string {
    return `${this.baseUrl.replace(/\/$/, '')}${path}`;
  }

  /**
   * Issue one request and return its parsed JSON body.
   *
   * The timeout covers the body too, not just the response headers: a
   * container that accepts the connection and then stalls mid-body would
   * otherwise hold the request open indefinitely.
   */
  private async requestJson<T>(label: string, path: string, init: RequestInit = {}): Promise<T> {
    if (this.closed) throw new Error(`${label}: client closed`);
    const controller = new AbortController();
    this.inFlight.add(controller);
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const r = await fetch(this.url(path), { ...init, signal: controller.signal });
      if (!r.ok) throw new Error(`${label} ${r.status}`);
      return (await r.json()) as T;
    } finally {
      clearTimeout(timer);
      this.inFlight.delete(controller);
    }
  }

  private postJson<T>(label: string, path: string, body: unknown): Promise<T> {
    return this.requestJson<T>(label, path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  /**
   * Abort everything in flight and refuse further requests.
   *
   * Called when the plugin stops. Without it, a `/control` body composed before
   * the stop can still land on the container after the restarted plugin has
   * pushed its new settings, leaving the container on the old confidence,
   * labels or enabled state until the next sync corrects it. Dropping the
   * request before it is delivered is the client's half of that; a request the
   * container has already accepted is beyond our reach (see the PR discussion
   * on fencing `/control` with a generation token).
   */
  close(): void {
    this.closed = true;
    for (const c of this.inFlight) c.abort();
    this.inFlight.clear();
  }

  health(): Promise<HealthInfo> {
    return this.requestJson<HealthInfo>('health', '/health');
  }

  control(body: ControlBody): Promise<any> {
    return this.postJson<any>('control', '/control', body);
  }

  ptz(camera: string, body: PtzBody): Promise<any> {
    return this.postJson<any>('ptz', `/ptz/${encodeURIComponent(camera)}`, body);
  }

  /** URL of the list of PTZ-capable cameras (used by the proxy). */
  ptzListUrl(): string {
    return this.url('/ptz');
  }

  /** URL of the annotated MJPEG stream for a camera (used by the proxy). */
  streamUrl(camera: string): string {
    return this.url(`/stream/${encodeURIComponent(camera)}.mjpg`);
  }

  snapshotUrl(camera: string): string {
    return this.url(`/snapshot/${encodeURIComponent(camera)}`);
  }

  wsUrl(): string {
    return this.baseUrl.replace(/^http/, 'ws').replace(/\/$/, '') + '/ws/events';
  }
}
