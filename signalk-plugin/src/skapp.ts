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
  // PropertyValues: the server's channel for plugins to announce themselves to
  // each other (Collision Alerts announces `collisionAlerts`). Optional: older
  // servers lack it, and then this plugin keeps every alarm of its own.
  onPropertyValues?(
    name: string,
    cb: (history: Array<{ value?: unknown } | undefined>) => void
  ): () => void;
  // Delta subscriptions, as a client's. Optional: without it, other
  // producers' alarms are only noticed on the next process cycle.
  subscriptionmanager?: {
    subscribe(
      command: { context: string; subscribe: Array<{ path: string; policy?: string }> },
      unsubscribes: Array<() => void>,
      errorCallback: (err: unknown) => void,
      callback: (delta: Delta & { updates: Array<{ $source?: string }> }) => void
    ): void;
  };
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
