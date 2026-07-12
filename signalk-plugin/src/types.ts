// TypeScript mirror of the container's DetectionEvent contract
// (vision-service/app/schemas.py). Kept minimal; validation against the
// generated JSON Schema in eventStream.ts is the real guard against drift.

// Camera name is free-form on the wire ("forward"/"aft" are the defaults, but any
// configured camera name is valid), so don't narrow it to a misleading union.
export type CameraName = string;
export type Backend =
  | 'tensorrt' | 'torch-cuda' | 'torch-cpu' | 'mock' | 'deepstream';
export type RangeMethod = 'horizon' | 'known_size';
export type CalibrationStatus = 'ok' | 'uncalibrated' | 'auto';

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Geometry {
  relative_bearing_deg: number;
  range_m: number | null;
  range_method: RangeMethod | null;
  range_confidence: number;
}

export interface PixelVelocity {
  vx: number;
  vy: number;
}

export interface RawTarget {
  track_id: number | null;
  // Per-session serial (per camera): never recycled, unlike track_id's bounded
  // 10-99 display range where a freed number can later name a DIFFERENT
  // vessel. Preferred for target/blip identity; optional so events from an
  // older container (without the field) still work on track_id.
  stable_id?: number | null;
  label: string;
  coco_class: number;
  confidence: number;
  bbox: BBox;
  is_person_in_water: boolean;
  geometry: Geometry;
  pixel_velocity: PixelVelocity;
  first_seen: string | null;
  age_frames: number;
  // True when the stabilizer is coasting this track across a detector dropout
  // (last detection + velocity) rather than from a fresh detection this frame.
  coasting?: boolean;
}

export interface DetectionEvent {
  schema_version: string;
  camera: string; // free-form per contract; "forward"/"aft" are the defaults
  timestamp: string;
  frame_seq: number;
  frame_size: { w: number; h: number };
  horizon_y: number | null;
  inference: { backend: Backend; latency_ms: number };
  calibration_status: CalibrationStatus;
  targets: RawTarget[];
}

// --- Plugin-internal enriched model ---------------------------------------

export type ThreatLevel = 'none' | 'low' | 'medium' | 'high';

export interface LatLon {
  latitude: number;
  longitude: number;
}

export interface OwnShip {
  position: LatLon | null;
  headingTrue: number | null; // rad
  sog: number | null; // m/s
  cog: number | null; // rad
  // True when at least one nav value was dropped (nulled) because its SignalK
  // timestamp was older than the configured max age (or unparseable). Downstream
  // treats stale/missing own kinematics as unknown — never as a stationary ship —
  // so a frozen GPS/heading feed can't silently georeference targets to an old
  // fix or suppress a CPA by assuming zero own velocity.
  stale: boolean;
}

export interface EnrichedTarget extends RawTarget {
  key: string; // `${camera}.${stable_id ?? track_id}` (see targetKey)
  camera: string;
  bearingTrue: number | null; // rad
  position: LatLon | null;
  aisCorrelated: boolean;
  aisMmsi: string | null;
  aisCog: number | null; // rad, from a correlated AIS contact
  aisSog: number | null; // m/s, from a correlated AIS contact
  cpa: number | null; // m
  tcpa: number | null; // s
  sog: number | null; // m/s, target ground speed (estimated from track or AIS)
  cog: number | null; // rad [0,2π), target ground course
  threatLevel: ThreatLevel;
  lastSeen: number; // epoch ms
}
