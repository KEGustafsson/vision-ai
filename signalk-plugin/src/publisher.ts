// Build and emit SignalK deltas for the novel vision.* tree, plus optional
// synthetic AIS blips. Sends metadata (units/zones) once per concrete path and
// ages out stale tracks by publishing null.

import { PluginConfig } from './config';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

interface Meta {
  units?: string;
  description: string;
  displayName?: string;
  zones?: Array<{ state: string; lower?: number; upper?: number }>;
}

// Every leaf published under vision.targets.<camera>.<trackId>; used both to
// age out departed tracks and to fully retract state on reset().
const TARGET_LEAVES = [
  'label', 'confidence', 'bearingTrue', 'distance', 'position',
  'cpa', 'tcpa', 'threatLevel', 'aisCorrelated', 'aisMmsi', 'rangeMethod',
];

const FIELD_META: Record<string, Meta> = {
  bearingTrue: { units: 'rad', description: 'True bearing to visual target' },
  distance: { units: 'm', description: 'Estimated range to visual target' },
  confidence: { units: 'ratio', description: 'Detection confidence' },
  cpa: { units: 'm', description: 'Closest point of approach' },
  tcpa: { units: 's', description: 'Time to closest point of approach' },
};

export class Publisher {
  private metaSent = new Set<string>();
  private publishedTracks = new Set<string>();
  private publishedBlips = new Set<string>();

  constructor(
    private app: ServerApp,
    private pluginId: string,
    private cfg: PluginConfig
  ) {}

  private emit(values: Array<{ path: string; value: unknown }>, meta?: Array<{ path: string; value: unknown }>): void {
    if (values.length === 0 && (!meta || meta.length === 0)) return;
    this.app.handleMessage(this.pluginId, {
      context: 'vessels.self',
      updates: [
        {
          source: { label: 'vision-ai' },
          timestamp: new Date().toISOString(),
          values,
          ...(meta && meta.length ? { meta } : {}),
        },
      ],
    });
  }

  private metaFor(path: string, field: string): { path: string; value: unknown } | null {
    if (this.metaSent.has(path)) return null;
    const m = FIELD_META[field];
    if (!m) return null;
    this.metaSent.add(path);
    return { path, value: m };
  }

  publishTargets(targets: EnrichedTarget[]): void {
    if (!this.cfg.enableVisualRadar) return;
    const values: Array<{ path: string; value: unknown }> = [];
    const meta: Array<{ path: string; value: unknown }> = [];
    const current = new Set<string>();

    for (const t of targets) {
      if (t.track_id === null) continue; // only persistent tracks go on the radar
      const base = `vision.targets.${t.camera}.${t.track_id}`;
      current.add(base);

      const push = (leaf: string, value: unknown, field?: string) => {
        const path = `${base}.${leaf}`;
        values.push({ path, value });
        if (field) {
          const m = this.metaFor(path, field);
          if (m) meta.push(m);
        }
      };

      // Optional measurements: emit the value, or an explicit null if it has
      // gone away while the track stays live — otherwise SignalK would keep
      // showing the last known value indefinitely.
      const pushOptional = (leaf: string, value: unknown, field?: string) => {
        if (value === null || value === undefined) {
          values.push({ path: `${base}.${leaf}`, value: null });
        } else {
          push(leaf, value, field);
        }
      };

      push('label', t.label);
      push('confidence', t.confidence, 'confidence');
      push('rangeMethod', t.geometry.range_method);
      push('aisCorrelated', t.aisCorrelated);
      push('aisMmsi', t.aisMmsi);
      push('threatLevel', t.threatLevel);
      pushOptional('bearingTrue', t.bearingTrue, 'bearingTrue');
      pushOptional('distance', t.geometry.range_m, 'distance');
      pushOptional('position', t.position);
      pushOptional('cpa', t.cpa, 'cpa');
      pushOptional('tcpa', t.tcpa, 'tcpa');
    }

    // Age out tracks that disappeared.
    for (const base of [...this.publishedTracks]) {
      if (!current.has(base)) {
        for (const leaf of TARGET_LEAVES) {
          values.push({ path: `${base}.${leaf}`, value: null });
        }
      }
    }
    this.publishedTracks = current;

    this.emit(values, meta);
    if (this.cfg.enableAisBlips) this.publishBlips(targets);
  }

  publishFusionSummary(darkCount: number, aisCount: number): void {
    this.emit([
      { path: 'vision.fusion.darkTargetCount', value: darkCount },
      { path: 'vision.fusion.aisCorrelatedCount', value: aisCount },
      { path: 'vision.fusion.lastUpdate', value: new Date().toISOString() },
    ]);
  }

  publishSystem(stats: {
    activeCamera?: string;
    mode?: string;
    backend?: string;
    inferenceFps?: number;
    horizonY?: number | null;
    perCameraCounts?: Record<string, number>;
  }): void {
    const values: Array<{ path: string; value: unknown }> = [];
    const meta: Array<{ path: string; value: unknown }> = [];
    if (stats.activeCamera !== undefined) values.push({ path: 'vision.system.activeCamera', value: stats.activeCamera });
    if (stats.mode !== undefined) values.push({ path: 'vision.system.mode', value: stats.mode });
    if (stats.backend !== undefined) values.push({ path: 'vision.system.backend', value: stats.backend });
    if (stats.horizonY !== undefined) values.push({ path: 'vision.system.horizonY', value: stats.horizonY });
    if (stats.inferenceFps !== undefined) {
      const path = 'vision.system.inferenceFps';
      values.push({ path, value: stats.inferenceFps });
      if (!this.metaSent.has(path)) {
        this.metaSent.add(path);
        meta.push({
          path,
          value: {
            units: 'Hz',
            description: 'Vision inference frame rate',
            zones: [
              { state: 'alarm', upper: 3 },
              { state: 'warn', lower: 3, upper: 6 },
              { state: 'normal', lower: 6 },
            ],
          },
        });
      }
    }
    for (const [cam, count] of Object.entries(stats.perCameraCounts ?? {})) {
      values.push({ path: `vision.${cam}.targetCount`, value: count });
    }
    this.emit(values, meta);
  }

  /** Optional: render high-confidence visual targets as synthetic AIS vessels. */
  private publishBlips(targets: EnrichedTarget[]): void {
    const ts = new Date().toISOString();
    const current = new Set<string>();
    for (const t of targets) {
      if (!t.position || t.track_id === null) continue;
      const uuid = `urn:mrn:signalk:uuid:vision-${t.camera}-${t.track_id}`;
      current.add(uuid);
      this.app.handleMessage(this.pluginId, {
        context: `vessels.${uuid}`,
        updates: [
          {
            source: { label: 'vision-ai' },
            timestamp: ts,
            values: [
              { path: 'navigation.position', value: t.position },
              { path: 'name', value: `VIS-${t.label}-${t.track_id}` },
            ],
          },
        ],
      });
    }
    // Age out departed blips by nulling their position (best-effort removal).
    for (const uuid of this.publishedBlips) {
      if (current.has(uuid)) continue;
      this.app.handleMessage(this.pluginId, {
        context: `vessels.${uuid}`,
        updates: [
          {
            source: { label: 'vision-ai' },
            timestamp: ts,
            values: [{ path: 'navigation.position', value: null }],
          },
        ],
      });
    }
    this.publishedBlips = current;
  }

  /** Retract all previously published state, then clear bookkeeping. Called on
   * plugin stop so stale vision.* leaves and synthetic blips don't linger. */
  reset(): void {
    const ts = new Date().toISOString();
    const values: Array<{ path: string; value: unknown }> = [];
    for (const base of this.publishedTracks) {
      for (const leaf of TARGET_LEAVES) values.push({ path: `${base}.${leaf}`, value: null });
    }
    this.emit(values);
    for (const uuid of this.publishedBlips) {
      this.app.handleMessage(this.pluginId, {
        context: `vessels.${uuid}`,
        updates: [
          { source: { label: 'vision-ai' }, timestamp: ts, values: [{ path: 'navigation.position', value: null }] },
        ],
      });
    }
    this.metaSent.clear();
    this.publishedTracks.clear();
    this.publishedBlips.clear();
  }
}
