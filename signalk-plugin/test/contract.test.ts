// Contract test: a representative event shaped exactly like the container emits
// (vision-service/app/schemas.py) must validate against the SAME generated JSON
// Schema the plugin enforces at runtime (eventStream.ts). This closes the
// producer→consumer loop — the Python side proves it can round-trip its own
// model; this proves the payload still satisfies the schema the TS side checks.

import { readFileSync } from 'fs';
import { join } from 'path';
import { describe, it, expect } from 'vitest';
import Ajv from 'ajv';
import addFormats from 'ajv-formats';

const schema = JSON.parse(
  readFileSync(join(__dirname, '..', 'schema', 'detection-event.schema.json'), 'utf-8')
);
const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);
const validate = ajv.compile(schema);

function baseEvent(): Record<string, unknown> {
  return {
    schema_version: '1.0',
    camera: 'forward',
    timestamp: '2026-06-20T12:00:00Z',
    frame_seq: 1,
    frame_size: { w: 1280, h: 960 },
    horizon_y: 350,
    inference: { backend: 'deepstream', latency_ms: 12.3 },
    calibration_status: 'auto',
    targets: [
      {
        track_id: 11,
        label: 'vessel',
        coco_class: 8,
        confidence: 0.9,
        bbox: { x: 10, y: 20, w: 30, h: 40 },
        is_person_in_water: false,
        geometry: {
          relative_bearing_deg: 5,
          range_m: 400,
          range_method: 'horizon',
          range_confidence: 0.7,
        },
        pixel_velocity: { vx: 0, vy: 0 },
        first_seen: null,
        age_frames: 3,
        coasting: false,
      },
    ],
  };
}

describe('DetectionEvent wire contract', () => {
  it('validates a full event (incl. deepstream backend + coasting)', () => {
    const ok = validate(baseEvent());
    expect(validate.errors).toBeNull();
    expect(ok).toBe(true);
  });

  it('accepts an event that omits the optional targets array', () => {
    const ev = baseEvent();
    delete ev.targets;
    expect(validate(ev)).toBe(true);
  });

  it('every Backend enum value the TS type lists is allowed by the schema', () => {
    for (const backend of ['tensorrt', 'torch-cuda', 'torch-cpu', 'mock', 'deepstream']) {
      const ev = baseEvent();
      (ev.inference as Record<string, unknown>).backend = backend;
      expect(validate(ev), `backend ${backend}`).toBe(true);
    }
  });

  it('rejects a structurally invalid event (confidence out of range)', () => {
    const ev = baseEvent();
    (ev.targets as Array<Record<string, unknown>>)[0].confidence = 5;
    expect(validate(ev)).toBe(false);
  });
});
