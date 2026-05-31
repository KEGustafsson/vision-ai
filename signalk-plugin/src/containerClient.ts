// Thin REST client for controlling the vision container.

export interface ControlBody {
  active_camera?: string;
  confidence?: number;
  max_targets?: number;
  mode_hint?: string;
  labels?: string[]; // canonical object types to surface (person | vessel | buoy)
  enabled?: boolean; // master on/off: pause/resume detection in the container
}

export interface PtzBody {
  action?: 'move' | 'stop' | 'home';
  pan?: number;
  tilt?: number;
  zoom?: number;
}

export class ContainerClient {
  constructor(private baseUrl: string) {}

  private url(path: string): string {
    return `${this.baseUrl.replace(/\/$/, '')}${path}`;
  }

  async health(): Promise<any> {
    const r = await fetch(this.url('/health'));
    if (!r.ok) throw new Error(`health ${r.status}`);
    return r.json();
  }

  async cameras(): Promise<string[]> {
    const r = await fetch(this.url('/cameras'));
    if (!r.ok) throw new Error(`cameras ${r.status}`);
    return (await r.json()) as string[];
  }

  async control(body: ControlBody): Promise<any> {
    const r = await fetch(this.url('/control'), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`control ${r.status}`);
    return r.json();
  }

  async ptz(camera: string, body: PtzBody): Promise<any> {
    const r = await fetch(this.url(`/ptz/${encodeURIComponent(camera)}`), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
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
