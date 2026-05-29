// Minimal structural interface for the SignalK ServerAPI surface we use.
// Avoids a hard compile dependency on the exact @signalk/server-api version
// while keeping call sites type-checked.

export interface Delta {
  context?: string;
  updates: Array<{
    source?: { label: string; type?: string };
    timestamp?: string;
    values?: Array<{ path: string; value: unknown }>;
    meta?: Array<{ path: string; value: unknown }>;
  }>;
}

export interface ServerApp {
  handleMessage(pluginId: string, delta: Delta, version?: string): void;
  getSelfPath(path: string): any;
  getPath(path: string): any;
  debug(msg: string, ...args: unknown[]): void;
  error(msg: string | Error): void;
  setPluginStatus?(msg: string): void;
  setPluginError?(msg: string): void;
}

export interface Plugin {
  id: string;
  name: string;
  description?: string;
  schema: () => object;
  uiSchema?: () => object;
  start: (settings: any, restart?: () => void) => void;
  stop: () => void;
  registerWithRouter?: (router: any) => void;
  statusMessage?: () => string;
}
