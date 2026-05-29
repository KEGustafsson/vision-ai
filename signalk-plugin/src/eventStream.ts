// WebSocket consumer of the container's detection events, with reconnect
// backoff and JSON-Schema validation against the shared contract.

import { readFileSync } from 'fs';
import { join } from 'path';
import Ajv, { ValidateFunction } from 'ajv';
import addFormats from 'ajv-formats';
import WebSocket from 'ws';
import { DetectionEvent } from './types';

type Logger = { debug: (m: string, ...a: unknown[]) => void; error: (m: string) => void };

export class EventStream {
  private ws: WebSocket | null = null;
  private closed = false;
  private backoff = 1000;
  private validate: ValidateFunction | null = null;
  private validationWarned = false;

  constructor(
    private wsUrl: string,
    private onEvent: (ev: DetectionEvent) => void,
    private log: Logger
  ) {
    this.loadSchema();
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
    const ws = new WebSocket(this.wsUrl);
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
        if (!this.validationWarned) {
          this.validationWarned = true;
          this.log.error(`vision-ai: event failed schema validation: ${JSON.stringify(this.validate.errors)}`);
        }
        return;
      }
      this.onEvent(parsed as DetectionEvent);
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
