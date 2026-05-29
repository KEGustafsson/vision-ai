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

export class NotificationManager {
  private active = new Set<string>();
  private mobCounters = new Map<string, number>();
  private mobHoldUntil = new Map<string, number>();
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
    if (this.cfg.enableAisFusion) this.evaluateDark(targets, darkTargetKeys, wantActive);
    if (this.cfg.enableCollision) this.evaluateCollision(targets, wantActive);

    // Clear any previously-active notification no longer wanted.
    for (const path of [...this.active]) {
      if (!wantActive.has(path)) this.clear(path);
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

  clearAll(): void {
    for (const path of [...this.active]) this.clear(path);
    this.mobCounters.clear();
    this.mobHoldUntil.clear();
    this.lastMobMessage = null;
  }
}
