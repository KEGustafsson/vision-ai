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
  constructor(private baseUrl: string, private timeoutMs: number = REQUEST_TIMEOUT_MS) {}

  private url(path: string): string {
    return `${this.baseUrl.replace(/\/$/, '')}${path}`;
  }

  private fetch(path: string, init: RequestInit = {}): Promise<Response> {
    return fetch(this.url(path), { ...init, signal: AbortSignal.timeout(this.timeoutMs) });
  }

  private post(path: string, body: unknown): Promise<Response> {
    return this.fetch(path, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  async health(): Promise<HealthInfo> {
    const r = await this.fetch('/health');
    if (!r.ok) throw new Error(`health ${r.status}`);
    return (await r.json()) as HealthInfo;
  }

  async control(body: ControlBody): Promise<any> {
    const r = await this.post('/control', body);
    if (!r.ok) throw new Error(`control ${r.status}`);
    return r.json();
  }

  async ptz(camera: string, body: PtzBody): Promise<any> {
    const r = await this.post(`/ptz/${encodeURIComponent(camera)}`, body);
    if (!r.ok) throw new Error(`ptz ${r.status}`);
    return r.json();
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
