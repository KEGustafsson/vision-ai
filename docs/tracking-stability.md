# Tracking stability — keeping targets locked to one id, without flicker

How the vision service keeps a detected target **on screen continuously** and
**under one detection number**, even though the raw per-frame detector output
blinks, jitters, and re-associates. This page maps each layer to the code and
to the multi-object-tracking (MOT) literature it borrows from, and lists the
knobs to turn when a symptom shows up on the water.

All layers run identically on every backend (`mock`, `jetson`/TensorRT,
`deepstream`); state is per camera.

## The stack, bottom to top

```text
YOLO detections (per frame, flickery)
  │
  ▼
Backend tracker              ByteTrack / BoT-SORT (torch, tensorrt)  or  NvDCF (deepstream)
  │   raw track ids            config/trackers/*_marine.yaml · deepstream/nvdcf_config.yml
  ▼
Waterline re-identification  app/detector/tracker.py  VelocityTracker.resolve()
  │   canonical ids            one id per hull, across partial/full box flips & dropouts
  ▼
Alternation merge + split    app/detector/tracker.py  _merge_alternating() / _check_split()
  │   repairs a vessel that already holds two live tracks; reversible
  ▼
Display ids + stable serial  app/detector/tracker.py  (2-digit ids, recycled + quarantined;
  │                           stable_id: per-session serial, never recycled)
  ▼
Track stabilizer             app/detector/stabilizer.py  TrackStabilizer
  │   confirm debounce · confidence hysteresis · coasting + track lock
  │   confidence-weighted box smoothing with a jump gate
  ▼
Sticky max-targets cap       app/detector/stabilizer.py  cap_targets_sticky()
  ▼
DetectionEvent + overlay
```

## Layer 1 — backend tracker association

The first line of defence is the association tracker itself. For the
Ultralytics backends the repo ships **marine-tuned presets** in
`vision-service/config/trackers/`:

- **`bytetrack_marine.yaml`** — [ByteTrack][bytetrack] with `track_buffer: 90`
  (~7.5 s of shadow life for a lost id, vs the stock ~2.5 s), looser
  `match_thresh: 0.7` (pitch/roll shifts every box a little between frames),
  and `new_track_thresh: 0.4` (a glint can't mint a one-frame id). ByteTrack's
  two-pass BYTE association is itself an anti-flicker device: weak detections
  (0.1–0.25) can't create tracks but can *keep an existing one matched*
  through a confidence dip. Fall back to this preset if the Jetson CPU is
  saturated or GMC misbehaves on a near-featureless sea (no flow anchors).
- **`botsort_marine.yaml`** (default in `jetson` mode) — same tuning on
  [BoT-SORT][botsort], adding **camera-motion compensation**
  (`gmc_method: sparseOptFlow`): the frame-to-frame global transform is
  estimated from the background and subtracted before association, so the
  boat's own pitch/roll/yaw no longer reads as every target jumping at once —
  the biggest cause of association breaks in a seaway. Costs some CPU per
  frame (optical flow). Native appearance re-id (`with_reid`) is left off —
  it is extra GPU work and the waterline re-id below already recovers
  identity.

The DeepStream backend (the production target) gets the equivalent — and
more — from [NvDCF][nvdcf] in `vision-service/deepstream/nvdcf_config.yml`:

- **Correlation-filter visual tracking** (`VisualTracker`): NvDCF localises
  each target visually between detector hits, with a low filter learning
  rate (long visual memory through spray) and a widened search region for
  wave-induced apparent motion. Note the section names are DeepStream
  6.x/7.x format — the parser silently ignores unknown sections, so a stale
  5.x-style `DCF:` block means the visual tracker runs on defaults.
- **Shadow tracking** (`maxShadowTrackingAge: 240`, ~40 s at the measured
  ~6 FPS per camera): a lost target keeps its id alive (unreported) and is
  re-acquired under the same number. COUPLING: the Python side must retain
  its per-track state at least as long (`detector.track_memory_frames`,
  default 260) or a shadow-reacquired raw id finds its display id already
  freed and the vessel re-blips under a new identity anyway.
- **Motion-based re-association** (`enableReAssoc: 1`, DeepStream 6.2+): a
  lost tracklet's trajectory is projected forward and a newborn tracklet
  matching it in position/velocity/size is re-linked to the old id — the
  tracker-level cure for dropout-induced id switches, with no ReID network
  cost. Projection windows are rescaled for slow marine motion at the
  pipeline's frame rate.
- **Cascaded association** (`associationMatcherType: 1`): confirmed targets
  match before tentative ones, so a flickery newborn can't steal a confirmed
  vessel's detection; the size-similarity gate is relaxed (0.6 → 0.4) so a
  hull-only ↔ hull+mast extent flip doesn't break association at the source.
- **REGULAR state estimator** (`stateEstimatorType: 2`, upgraded from
  `SIMPLE`): a Kalman filter on location/size/velocity that actually honours
  the configured noise covariances, so a vessel changing range is predicted
  more smoothly. **Not yet validated on-box** — the noise vars were tuned for
  `SIMPLE`; if a maneuvering target's box visibly lags, revert to
  `stateEstimatorType: 1` or lower `measurementNoiseVar4Detector`.

## Layer 2 — waterline re-identification (`VelocityTracker.resolve`)

Backend trackers associate by box IoU, so when a vessel's detected extent
flips between *hull only* and *hull + mast*, association breaks and a fresh
raw id appears — the same target then flickers between two numbers. The
waterline re-id aliases a NEW raw id back onto a recently seen track when
both boxes stand on the same **waterline footprint** (aligned bottom edge,
high horizontal overlap, similar hull width), with the stored footprint
advanced by the track's pixel velocity so a mover is re-acquired where it
*is now*. `person` is exempt in both directions — two swimmers must never be
fused into one MOB target.

Two association refinements come from the MOT literature:

- **Buffered matching** (from the [C-BIoU tracker][cbiou]): the overlap and
  waterline gates *widen with the dropout gap* —
  `reid_buffer_frac_per_frame` (default 0.03) of the hull width per missed
  frame, capped at `reid_buffer_max_frac` (0.25). A fresh flip is judged
  tightly; a target unseen for 20 frames, whose predicted position is
  correspondingly less certain, gets a proportionally wider matching space.
  This is C-BIoU's cascaded small-buffer/large-buffer idea expressed
  continuously in the gap.
- **Direction consistency** (from [OC-SORT][ocsort]'s observation-centric
  momentum): a track moving at ≥ `reid_dir_min_speed_px` (default 2 px/frame)
  is only re-identified by a candidate displaced broadly *along* its
  direction of travel. A box appearing clearly behind a mover is a different
  vessel, even where the widened buffered gate would geometrically accept
  it — the momentum gate counterbalances the buffer.

Two practical gate details: `reid_max_gap_frames` (default 120) is aligned
with NvDCF's re-association search range so both layers give up on a dropout
at the same age, and the hull-width gate **stands down for frame-edge-clipped
boxes** — a box cut off by the left/right frame edge has whatever width
happened to fit on screen (observed live: a vessel exiting frame-right churned
through four ids because each re-entry width failed the gate).

## Layer 2b — alternation merge, and its undo (`_merge_alternating` / `_check_split`)

The birth-time re-id above gets exactly one chance per raw id. If the gates
momentarily failed then (pitch, a bad first box), the vessel ends up holding
**two live tracks** — typically a hull track and a hull+mast track — that take
turns being detected, and its number, box extent and published range flap
between the two forever (measured live: pairs alternating 60–1250 times per
15 min). The merge pass repairs this: a same-footprint pair accumulates
**alternation evidence** (one side detected while the other is briefly dark),
required from *both* sides so a newcomer can never swallow a departed
neighbour's identity; enough evidence merges the younger track into the older,
which keeps its display id, serial and age.

Same-frame co-detections carry the discriminating signal:

- **Side by side** (horizontal overlap 0.5–0.8 of the narrower box): two
  simultaneous detections are two real vessels — the pair is blocked from
  merging and its evidence reset. Boats genuinely moored alongside keep
  co-occurring, so they stay apart.
- **Nested** (≥ 0.8, the same signature the contained-duplicate drop uses):
  the routine hull-inside-full double box of a *single* vessel — nested boxes
  survive NMS, so the raw tracker sees both together even though the event
  shows only one. This counts *for* the merge, on both sides at once: a hull
  track co-detected under its mast track never goes dark, so nested
  co-detections are the only evidence such a pair can produce.

Merges are **reversible**: two raw ids resolving to one canonical in the same
frame at *disjoint* footprints contradict the alias binding them, and enough
contradiction frames dissolve it (the pair is then co-blocked from an
immediate re-merge). This is the safety valve for a pair merged while
genuinely co-located that later separates. `person` is exempt from all of it,
as everywhere in re-id.

## Layer 3 — display ids and the stable serial

Raw tracker ids grow without bound; emitted detections carry a compact
2-digit display id (10–99, per camera). Lowest-free-first allocation keeps
numbers small and familiar; a freed id is **quarantined** for 150 frames
(~25 s at the measured ~6 FPS) so a number that just left one vessel cannot
reappear on a different one while the operator still associates it with the
first. Idle-track identity (velocity history, display id, serial) is retained
for `track_memory_frames` (default 260) — deliberately past NvDCF's shadow
window, so a shadow-reacquired raw id walks back into its existing ids.

Because the 2-digit pool is recycled, events also carry **`stable_id`**: a
per-camera, per-session serial that is never reused. Downstream identity —
the SignalK blip name/context `VIS-<camera>-<stable_id>` — keys on it, so a
chart contact can never change physical vessel mid-session; `track_id` stays
the number drawn on the video overlay.

## Layer 4 — track stabilizer (`TrackStabilizer`)

Per-track lifecycle over the canonical ids:

- **Confirm debounce** — a new track must be seen `stabilize_confirm_frames`
  times before it is shown (`person`: `stabilize_person_confirm_frames`,
  default first frame, MOB-critical).
- **Confidence hysteresis** — turns on at `confidence`, stays on until the
  EMA-smoothed confidence falls below `confidence × stabilize_hysteresis_ratio`,
  so a value hovering at the threshold cannot blink the box.
- **Coasting with track lock** — a confirmed track missing this frame is
  re-emitted (dashed, `coasting: true`) at its last box advanced by damped
  velocity, for up to `stabilize_max_coast_frames`. A track with at least
  `stabilize_lock_hits` (default 30) fresh detections is **locked** and may
  coast `stabilize_lock_coast_factor` (default 2×) longer — the same idea as
  ByteTrack's `track_buffer` and NvDCF's shadow tracking: an established
  vessel rides out a wave-occlusion dropout with box and id intact, while a
  young (possibly false) track still dies fast.
- **Confidence-weighted box smoothing with a jump gate** — every shown box is
  the rolling average of the track's recent raw boxes, each weighted by its
  detection confidence (the NSA-Kalman idea from [StrongSORT][strongsort],
  applied to the windowed average): a marginal detection nudges the box, a
  solid one moves it. A raw box that leaps implausibly far or resizes
  implausibly fast is rejected as a false measurement and the held average is
  emitted, until the leap persists `stabilize_jump_confirm_frames` frames and
  is accepted as real. Deliberately observation-only: no motion model, no
  prediction.
- **Same-frame duplicate continuity** — re-id/merge can put one id on two
  boxes in a frame (a partial and a full detection of the same vessel
  co-occur). The keeper is the box *closest to the track's currently shown
  box*, not the confidence winner: duplicate extents run neck-and-neck in
  confidence, so the winner would flip between them frame to frame and the
  drawn box (and its range) would flap ~a box-width each flip. The velocity
  anchor applies the same continuity rule to its one-sample-per-frame choice.

## Layer 5 — sticky output cap

When more targets are live than `max_det` allows, an incumbent keeps its slot
unless a challenger beats it by `max_det_sticky_margin` — two near-tied
targets can't swap the last slot (and blink) on every confidence wobble.
`person` always ranks first.

## Symptom → knob

| Symptom on the water | First knob to try |
|---|---|
| Box blinks off for a frame or two | raise `stabilize_max_coast_frames` |
| Established vessel drops id after a long wave occlusion | raise `stabilize_lock_hits`/`stabilize_lock_coast_factor`, or `track_buffer` in the tracker preset (deepstream: `maxShadowTrackingAge`, `maxTrackletMatchingTimeSearchRange`), or `reid_max_gap_frames` |
| Vessel re-blips as a NEW identity after a long dropout | `track_memory_frames` must exceed the backend's shadow window (deepstream: `maxShadowTrackingAge`) |
| (deepstream) same vessel gets a new id after every dropout | check `enableReAssoc: 1` is set and the config uses 6.x/7.x section names (a 5.x `DCF:` block is silently ignored) |
| Same vessel alternates between two numbers | should self-heal via the alternation merge within seconds; if not, its evidence gates (`VelocityTracker._MERGE_*`) or the re-id footprint gates (`reid_min_x_overlap`, `reid_buffer_frac_per_frame`) are rejecting the pair |
| One id flaps between two separated places | a wrong/stale merge mid-dissolve — the split (`_SPLIT_CONFIRM` contradiction frames) undoes it; persistent flapping means the two boxes still overlap ambiguously (0.5–0.8) |
| Vessel exiting the frame edge churns ids | the width gate already stands down for edge-clipped boxes; check the box actually touches the edge (within 2 px) |
| A departed vessel's id lands on a newcomer | tighten `reid_max_width_ratio`, raise `reid_dir_min_speed_px`, lower `reid_max_gap_frames` |
| Everything loses lock together in a seaway | already on `botsort_marine.yaml` (GMC) by default in `jetson` mode; check `gmc_method` didn't get reverted, or a near-featureless sea is starving optical flow of anchors |
| Boxes jitter / breathe | raise `stabilize_smooth_window`; check `stabilize_conf_weight` is on |
| Phantom box lingers after a target leaves | lower `stabilize_max_coast_frames` / `stabilize_lock_coast_factor` |
| Numbers churn upward in a busy scene | raise `new_track_thresh` in the tracker preset; check the confirm debounce |

## References

- ByteTrack: [Zhang et al., *ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box*, ECCV 2022][bytetrack]
- BoT-SORT: [Aharon et al., *BoT-SORT: Robust Associations Multi-Pedestrian
  Tracking*, 2022][botsort]
- C-BIoU: [Yang et al., *Hard to Track Objects with Irregular Motions and
  Similar Appearances? Make It Easier by Buffering the Matching Space*,
  WACV 2023][cbiou]
- OC-SORT: [Cao et al., *Observation-Centric SORT: Rethinking SORT for Robust
  Multi-Object Tracking*, CVPR 2023][ocsort]
- StrongSORT: [Du et al., *StrongSORT: Make DeepSORT Great Again*, 2022][strongsort]
- NvDCF / re-association: [NVIDIA DeepStream Gst-nvtracker documentation][nvdcf]

[bytetrack]: https://arxiv.org/abs/2110.06864
[botsort]: https://arxiv.org/abs/2206.14651
[cbiou]: https://arxiv.org/abs/2211.14317
[ocsort]: https://arxiv.org/abs/2203.14360
[strongsort]: https://arxiv.org/abs/2202.13514
[nvdcf]: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html
