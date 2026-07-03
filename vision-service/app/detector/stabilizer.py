"""Track stabilizer: damps the on/off flicker of per-frame detections.

The detector emits only the tracks matched in the *current* frame, so a box
blinks out whenever YOLO misses for a frame or its confidence dips below the
publish threshold. This stage gives each track a short lifecycle instead:

* **confidence hysteresis** — a track turns on at ``conf_on`` and stays on until
  its smoothed confidence falls below ``conf_off`` (< ``conf_on``), so a value
  hovering around the threshold no longer flickers.
* **coasting** — a confirmed track that isn't detected this frame is re-emitted
  (flagged ``coasting``) using its last box advanced by its pixel velocity, for
  up to ``max_coast_frames``. This is the "keep drawing the box + info even when
  not detected every frame" behaviour.
* **appearance debounce** — a new track must be seen ``confirm_frames`` times
  before it is shown, so a single-frame false positive never draws a box.
  ``person`` tracks use the (lower) ``person_confirm_frames`` instead: a person in
  the water is a man-overboard candidate, and holding it back for two extra frames
  adds latency to the most safety-critical detection. A single-frame false person
  is still debounced downstream by the plugin's MOB persistence counter before any
  alarm is raised, so confirming it here on the first frame is safe.
* **box smoothing** — each shown box is the **rolling average of the track's
  last few raw boxes**. No motion model and no prediction (explicit operator
  decision after predictive variants misbehaved on dense marina scenes): the
  emitted box simply follows the detections, with per-frame jitter and shape
  flips spread across the window instead of snapping. Averaging is linear, so
  the emitted bottom edge is exactly the average of the detected bottom
  edges — a steady waterline stays steady.

One instance per camera (state is keyed by the backend's stable track id).
Untracked detections (no id) can't be coasted and pass through on a plain
``conf_on`` gate. Confidence is smoothed with an EMA over detected frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

from .base import RawTrack


class _BoxSmoother:
    """Rolling average over the track's last ``window`` raw boxes. Just follows
    the detections — no velocity, no prediction, no holding."""

    def __init__(self, window: int):
        self._boxes: deque[tuple[float, float, float, float]] = deque(
            maxlen=max(1, window))

    def apply(self, tr: RawTrack) -> RawTrack:
        self._boxes.append((tr.x, tr.y, tr.w, tr.h))
        n = len(self._boxes)
        x, y, w, h = (sum(b[i] for b in self._boxes) / n for i in range(4))
        return replace(tr, x=x, y=y, w=w, h=h)


@dataclass
class _State:
    conf: float        # EMA of detection confidence
    track: RawTrack    # most recent (smoothed) detected track (box, velocity, ...)
    last_seq: int      # frame seq of the last fresh detection
    hits: int          # number of fresh detections so far
    confirmed: bool    # has cleared the appearance debounce
    smoother: _BoxSmoother | None = field(default=None)


def cap_targets_sticky(targets: list, max_det: int, prev_ids: set,
                       margin: float) -> list:
    """Cap a ranked target list to ``max_det`` with STICKY membership: a target
    emitted last frame keeps its slot unless a challenger's confidence beats it
    by ``margin``. Without this, targets whose smoothed confidences hover within
    noise of each other swap the last slot every few frames, and the loser
    blinks in and out of the event/overlay/target list. ``person`` always ranks
    first regardless of confidence: the cap must never squeeze out a possible
    man-overboard in a crowded frame.

    Duck-typed over anything with ``track_id``/``label``/``confidence`` (works
    for both ``Target`` and ``RawTrack``). The caller keeps ``prev_ids`` per
    camera and refreshes it from the returned list each frame."""
    ranked = sorted(
        targets,
        key=lambda t: (t.label == "person",
                       t.confidence + (margin if t.track_id in prev_ids else 0.0)),
        reverse=True,
    )
    return ranked[:max_det]


class TrackStabilizer:
    def __init__(self, confirm_frames: int = 3, max_coast_frames: int = 8,
                 hysteresis_ratio: float = 0.6, ema_alpha: float = 0.4,
                 coast_velocity_factor: float = 0.4,
                 person_confirm_frames: int = 1,
                 smooth: bool = True, smooth_window: int = 5):
        self.confirm_frames = max(1, confirm_frames)
        # MOB-critical: person tracks confirm faster (default: first frame).
        self.person_confirm_frames = max(1, person_confirm_frames)
        self.max_coast_frames = max(0, max_coast_frames)
        self.hysteresis_ratio = min(max(hysteresis_ratio, 0.0), 1.0)
        self.alpha = min(max(ema_alpha, 0.0), 1.0)
        # How much of the track's pixel velocity to apply while coasting.
        # 0 = freeze the box at its last position; 1 = full extrapolation. A
        # damped value keeps a coasted box near the object instead of letting a
        # noisy velocity estimate fling it away over several missed frames.
        self.coast_velocity_factor = max(0.0, coast_velocity_factor)
        # Rolling-average box smoothing (see module docstring). Applied per
        # track to everything emitted, so overlay, event geometry and coasting
        # all work from the same calm box.
        self.smooth = smooth
        self.smooth_window = max(1, smooth_window)
        self._st: dict[int, _State] = {}

    def _confirm_for(self, label: str) -> int:
        """Appearance-debounce threshold for a track, lowered for MOB-critical
        person tracks so they aren't held back by the generic false-positive gate."""
        return self.person_confirm_frames if label == "person" else self.confirm_frames

    def update(self, tracks: list[RawTrack], seq: int, conf_on: float) -> list[RawTrack]:
        conf_off = conf_on * self.hysteresis_ratio
        out: list[RawTrack] = []
        seen: dict[int, RawTrack] = {}
        for tr in tracks:
            if tr.track_id is None:
                # No stable id => no state, no coasting; plain threshold gate.
                if tr.confidence >= conf_on:
                    out.append(tr)
            else:
                # Waterline re-id can put the same id on two boxes in ONE frame
                # (a partial and a full detection of the same vessel co-occur);
                # keep only the stronger so one target never draws two boxes.
                prev = seen.get(tr.track_id)
                if prev is None or tr.confidence > prev.confidence:
                    seen[tr.track_id] = tr

        # Tracks detected this frame: refresh state, smooth confidence and box,
        # emit if confirmed and above the lower (off) threshold. The box filter
        # runs from the first sighting (not first emission), so by the time the
        # debounce clears the smoothed box is already settled.
        for tid, tr in seen.items():
            s = self._st.get(tid)
            if s is None:
                s = self._st[tid] = _State(conf=tr.confidence, track=tr,
                                           last_seq=seq, hits=1, confirmed=False)
                if self.smooth:
                    s.smoother = _BoxSmoother(self.smooth_window)
            else:
                s.conf = self.alpha * tr.confidence + (1 - self.alpha) * s.conf
                s.last_seq = seq
                s.hits += 1
            if s.smoother is not None:
                tr = s.smoother.apply(tr)
            # Coasting continues from the smoothed box, so a dropout doesn't
            # snap the box back to the last raw (jittery) position.
            s.track = tr
            if not s.confirmed and s.hits >= self._confirm_for(tr.label) and s.conf >= conf_on:
                s.confirmed = True
            if s.confirmed and s.conf >= conf_off:
                out.append(replace(tr, confidence=s.conf, coasting=False))

        # Tracks not detected this frame: coast confirmed ones for a while, then
        # forget. Drop unconfirmed (tentative) tracks promptly.
        for tid, s in list(self._st.items()):
            if tid in seen:
                continue
            missed = seq - s.last_seq
            if not s.confirmed:
                if missed > self._confirm_for(s.track.label):
                    del self._st[tid]
                continue
            if missed > self.max_coast_frames or s.conf < conf_off:
                del self._st[tid]
                continue
            base = s.track
            damp = self.coast_velocity_factor * missed
            out.append(replace(
                base,
                x=base.x + base.vx * damp,
                y=base.y + base.vy * damp,
                confidence=s.conf,
                coasting=True,
            ))
        return out
