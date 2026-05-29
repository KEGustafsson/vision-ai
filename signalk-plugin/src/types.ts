// TypeScript mirror of the container's DetectionEvent contract
// (vision-service/app/schemas.py). Kept minimal; validation against the
// generated JSON Schema in eventStream.ts is the real guard against drift.

export type CameraName = 'forward' | 'aft';
export type Backend = 'tensorrt' | 'torch-cuda' | 'torch-cpu' | 'mock';
export type RangeMethod = 'horizon' | 'known_size';

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
  label: string;
  coco_class: number;
  confidence: number;
  bbox: BBox;
  is_person_in_water: boolean;
  geometry: Geometry;
  pixel_velocity: PixelVelocity;
  first_seen: string | null;
  age_frames: number;
}

export interface DetectionEvent {
  schema_version: string;
  camera: CameraName;
  timestamp: string;
  frame_seq: number;
  frame_size: { w: number; h: number };
  horizon_y: number | null;
  inference: { backend: Backend; latency_ms: number };
  calibration_status: string;
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
}

export interface EnrichedTarget extends RawTarget {
  key: string; // `${camera}.${track_id}`
  camera: CameraName;
  bearingTrue: number | null; // rad
  position: LatLon | null;
  aisCorrelated: boolean;
  aisMmsi: string | null;
  cpa: number | null; // m
  tcpa: number | null; // s
  threatLevel: ThreatLevel;
  lastSeen: number; // epoch ms
}
