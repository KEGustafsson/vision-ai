// Lifecycle management for vision-driven SignalK notifications: MOB, dark
// targets and collision risk. Tracks active notifications so they can be
// cleared (value: null) when the condition resolves, with simple hysteresis to
// avoid flapping.

import { PluginConfig } from './config';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

type State = 'normal' | 'alert' | 'warn' | 'alarm' | 'emergency';

function sanitize(key: string): string {
  return key.replace(/[^a-zA-Z0-9]/g, '_');
}

// Notifications managed outside evaluate() (set/cleared by their own lifecycle),
// so the per-cycle clear loop must not touch them.
const EXTERNAL_PATHS = new Set([
  'notifications.vision.labelMismatch',
  'notifications.vision.schemaMismatch',
  'notifications.vision.staleEvents',
  'notifications.vision.containerDown',
  'notifications.vision.containerDegraded',
]);

const isHoldable = (path: string): boolean =>
  path.startsWith('notifications.vision.collision.') ||
  path.startsWith('notifications.vision.darkTarget.');

export class NotificationManager {
  private active = new Set<string>();
  private mobCounters = new Map<string, number>();
  private mobHoldUntil = new Map<string, number>();
  // Anti-flap hold for collision/dark notifications: keep them up briefly after
  // they would clear so a target hovering around a threshold doesn't toggle the
  // audible alarm every cycle.
  private holdUntil = new Map<string, number>();
  private lastMobMessage: string | null = null;

  constructor(
    private app: ServerApp,
    private pluginId: string,
    private cfg: PluginConfig
  ) {}

  private send(path: string, state: State, message: string, methods: string[]): void {
    this.app.handleMessage(this.pluginId, {
      context: 'vessels.self',
      updates: [
        {
          source: { label: 'vision-ai' },
          timestamp: new Date().toISOString(),
          values: [{ path, value: { state, method: methods, message } }],
        },
      ],
    });
    this.active.add(path);
  }

  private clear(path: string): void {
    if (!this.active.has(path)) return;
    this.app.handleMessage(this.pluginId, {
      context: 'vessels.self',
      updates: [
        {
          source: { label: 'vision-ai' },
          timestamp: new Date().toISOString(),
          values: [{ path, value: null }],
        },
      ],
    });
    this.active.delete(path);
  }

  /** Evaluate the current target set and emit/clear notifications. */
  evaluate(targets: EnrichedTarget[], darkTargetKeys: Set<string>, nowMs: number): void {
    const wantActive = new Set<string>();

    if (this.cfg.enableMob) this.evaluateMob(targets, nowMs, wantActive);
    // Collision/dark notifications need their computation on (they read values
    // it produces) and the notification itself enabled.
    if (this.cfg.enableAisFusion && this.cfg.notifyDarkTarget) {
      this.evaluateDark(targets, darkTargetKeys, wantActive);
    }
    if (this.cfg.enableCollision && this.cfg.notifyCollision) {
      this.evaluateCollision(targets, wantActive);
    }

    this.applyHold(wantActive, nowMs);

    // Clear any previously-active notification no longer wanted. Skip paths owned
    // by a separate lifecycle (label/schema mismatch) so we don't clear them here.
    for (const path of [...this.active]) {
      if (EXTERNAL_PATHS.has(path)) continue;
      if (!wantActive.has(path)) this.clear(path);
    }
  }

  // Refresh the hold for currently-wanted holdable paths, then re-add any held
  // path whose hold hasn't expired so it isn't cleared yet (anti-flap).
  private applyHold(want: Set<string>, nowMs: number): void {
    const holdMs = this.cfg.notifyHoldS * 1000;
    for (const path of want) {
      if (isHoldable(path)) this.holdUntil.set(path, nowMs + holdMs);
    }
    for (const [path, until] of [...this.holdUntil]) {
      if (until <= nowMs) {
        this.holdUntil.delete(path);
      } else if (!want.has(path) && this.active.has(path)) {
        want.add(path);
      }
    }
  }

  private evaluateMob(targets: EnrichedTarget[], nowMs: number, want: Set<string>): void {
    const path = 'notifications.mob';
    let fire = false;

    for (const t of targets) {
      if (!t.is_person_in_water || t.confidence < this.cfg.mobMinConfidence) continue;
      const c = (this.mobCounters.get(t.key) ?? 0) + 1;
      this.mobCounters.set(t.key, c);
      if (c >= this.cfg.mobPersistFrames) {
        fire = true;
        // Hold the alarm for 60s after last sighting (drift / occlusion).
        this.mobHoldUntil.set(t.key, nowMs + 60000);
        // Capture the last-known position/bearing so the message stays useful
        // even if the detection drops out during the hold window.
        if (t.position) {
          this.lastMobMessage =
            `MAN OVERBOARD — person in water at ` +
            `${t.position.latitude.toFixed(5)}, ${t.position.longitude.toFixed(5)}` +
            ` (${t.camera} camera)`;
        } else {
          this.lastMobMessage = `MAN OVERBOARD — person in water, bearing ${
            t.bearingTrue !== null ? ((t.bearingTrue * 180) / Math.PI).toFixed(0) + '°T' : 'unknown'
          } (${t.camera} camera)`;
        }
      }
    }

    // Decay counters for tracks not seen this cycle.
    const seen = new Set(targets.map((t) => t.key));
    for (const key of [...this.mobCounters.keys()]) {
      if (!seen.has(key)) this.mobCounters.delete(key);
    }

    // Honour hold windows even if the detection briefly drops out.
    if (!fire) {
      for (const [, until] of this.mobHoldUntil) {
        if (until > nowMs) {
          fire = true;
          break;
        }
      }
    }
    for (const [key, until] of [...this.mobHoldUntil]) {
      if (until <= nowMs) this.mobHoldUntil.delete(key);
    }

    if (fire) {
      this.send(path, 'emergency',
        this.lastMobMessage ?? 'Man overboard detected by vision system',
        ['visual', 'sound']);
      want.add(path);
    } else {
      this.lastMobMessage = null;
    }
  }

  private evaluateDark(targets: EnrichedTarget[], darkKeys: Set<string>, want: Set<string>): void {
    for (const t of targets) {
      if (!darkKeys.has(t.key)) continue;
      const path = `notifications.vision.darkTarget.${sanitize(t.key)}`;
      const rng = t.geometry.range_m !== null ? `${t.geometry.range_m.toFixed(0)} m` : 'unknown range';
      const brg = t.bearingTrue !== null ? `${((t.bearingTrue * 180) / Math.PI).toFixed(0)}°T` : '';
      this.send(path, 'alert', `Non-AIS ${t.label} detected at ${brg} ${rng}`, ['visual']);
      want.add(path);
    }
  }

  private evaluateCollision(targets: EnrichedTarget[], want: Set<string>): void {
    for (const t of targets) {
      if (t.threatLevel !== 'medium' && t.threatLevel !== 'high') continue;
      const path = `notifications.vision.collision.${sanitize(t.key)}`;
      const state: State = t.threatLevel === 'high' ? 'alarm' : 'warn';
      const tcpa = t.tcpa !== null ? `${(t.tcpa / 60).toFixed(1)} min` : '?';
      const cpa = t.cpa !== null ? `${t.cpa.toFixed(0)} m` : '?';
      const who = t.aisMmsi ? `AIS ${t.aisMmsi}` : t.label;
      this.send(path, state, `Collision risk: ${who} CPA ${cpa} in ${tcpa}`, ['visual', 'sound']);
      want.add(path);
    }
  }

  setLabelMismatch(message: string): void {
    this.send('notifications.vision.labelMismatch', 'warn', message, ['visual']);
  }

  clearLabelMismatch(): void {
    this.clear('notifications.vision.labelMismatch');
  }

  setSchemaMismatch(message: string): void {
    this.send('notifications.vision.schemaMismatch', 'warn', message, ['visual']);
  }

  clearSchemaMismatch(): void {
    this.clear('notifications.vision.schemaMismatch');
  }

  setStaleEvents(message: string): void {
    this.send('notifications.vision.staleEvents', 'warn', message, ['visual']);
  }

  clearStaleEvents(): void {
    this.clear('notifications.vision.staleEvents');
  }

  // Container unreachable: the whole vision sensor is offline (no events, no
  // stream). Raised at `alarm` so it's distinct from the data-quality warnings
  // above, but visual-only to avoid an audible alarm flapping on a routine
  // container restart — the plugin recovers on its own when it comes back.
  setContainerDown(message: string): void {
    this.send('notifications.vision.containerDown', 'alarm', message, ['visual']);
  }

  clearContainerDown(): void {
    this.clear('notifications.vision.containerDown');
  }

  // Container reachable but reporting status="degraded" (camera/RTSP stall or a
  // pipeline auto-restart). The sensor is partially working, so this is a warn.
  setContainerDegraded(message: string): void {
    this.send('notifications.vision.containerDegraded', 'warn', message, ['visual']);
  }

  clearContainerDegraded(): void {
    this.clear('notifications.vision.containerDegraded');
  }

  clearAll(): void {
    for (const path of [...this.active]) this.clear(path);
    this.mobCounters.clear();
    this.mobHoldUntil.clear();
    this.holdUntil.clear();
    this.lastMobMessage = null;
  }
}
