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
* **box smoothing with a jump gate** — each shown box is the **rolling average
  of the track's recent raw boxes**, and a raw box that leaps implausibly far
  from that average (or changes size implausibly fast) is treated as a FALSE
  measurement: it is not averaged in and the held average is emitted instead.
  Real objects don't teleport — a boat moves a small fraction of its own
  length per frame — so a big jump is detector noise (a glint, a cluster box
  over a marina row) until it *persists*: after ``jump_confirm`` consecutive
  out-of-gate frames the new place is accepted as real and the window restarts
  there. No motion model and no prediction (explicit operator decision):
  everything is judged against boxes already seen, never against an estimate
  of where the target is going. Each box is weighted by its detection
  confidence (StrongSORT's NSA-Kalman idea, arXiv:2202.13514, applied to the
  windowed average): a shaky low-confidence measurement perturbs the shown
  box less than a solid one.
* **track lock** — a track with at least ``lock_hits`` fresh detections has
  proven itself real and is allowed to coast ``lock_coast_factor`` times
  longer than ``max_coast_frames`` before being dropped (the same idea as
  ByteTrack's ``track_buffer`` and NvDCF's shadow tracking: an established
  target survives a longer dropout with its box and id intact, while a young
  track still dies fast).

One instance per camera (state is keyed by the backend's stable track id).
Untracked detections (no id) can't be coasted and pass through on a plain
``conf_on`` gate. Confidence is smoothed with an EMA over detected frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

from .base import RawTrack


class _BoxSmoother:
    """Rolling average over the track's recent raw boxes, guarded by a jump
    gate (see module docstring). Judges each raw box only against boxes
    already seen — no velocity, no prediction.

    Gate: relative to the current average, the center may shift at most
    ``jump_tol`` of the box's larger dimension per elapsed frame, and width/
    height may each grow or shrink at most ``(1 + jump_tol)`` per elapsed
    frame (a dropout naturally earns a proportionally wider gate — that is
    just looser plausibility after not looking, not a motion estimate). An
    out-of-gate box is rejected: the held average is emitted, nothing enters
    the window. ``jump_confirm`` consecutive rejections mean the change is
    real (the detector re-seated, the target actually is elsewhere): the
    window restarts from the latest raw box. An in-gate box resets the
    rejection count, so a lone spike every few frames never accumulates
    acceptance.

    With ``conf_weight`` each box enters the average weighted by its detection
    confidence (NSA-Kalman analogue): a marginal detection nudges the shown box,
    a solid one moves it."""

    def __init__(self, window: int, jump_tol: float, jump_confirm: int,
                 conf_weight: bool = True):
        self._boxes: deque[tuple[float, float, float, float, float]] = deque(
            maxlen=max(1, window))
        self._jump_tol = max(0.0, jump_tol)
        self._jump_confirm = max(1, jump_confirm)
        self._conf_weight = conf_weight
        self._rejects = 0
        self._last_seq = 0

    def _avg(self) -> tuple[float, float, float, float]:
        total = sum(b[4] for b in self._boxes)
        x, y, w, h = (sum(b[i] * b[4] for b in self._boxes) / total
                      for i in range(4))
        return x, y, w, h

    def _plausible(self, tr: RawTrack, gap: int) -> bool:
        ax, ay, aw, ah = self._avg()
        allowance = self._jump_tol * max(1, gap)
        shift = max(abs((tr.x + tr.w / 2.0) - (ax + aw / 2.0)),
                    abs((tr.y + tr.h / 2.0) - (ay + ah / 2.0)))
        if shift > allowance * max(aw, ah):
            return False
        max_ratio = (1.0 + self._jump_tol) ** max(1, gap)
        for new, old in ((tr.w, aw), (tr.h, ah)):
            if new <= 0 or old <= 0 or new / old > max_ratio or old / new > max_ratio:
                return False
        return True

    def apply(self, tr: RawTrack, seq: int) -> RawTrack:
        gap, self._last_seq = seq - self._last_seq, seq
        if self._boxes and self._jump_tol > 0 and not self._plausible(tr, gap):
            self._rejects += 1
            if self._rejects < self._jump_confirm:
                # False measurement: emit the held average unchanged.
                x, y, w, h = self._avg()
                return replace(tr, x=x, y=y, w=w, h=h)
            # The leap persisted: it is real. Follow it from scratch.
            self._boxes.clear()
        self._rejects = 0
        # A floor on the weight keeps a window of all-marginal detections from
        # degenerating (and a zero-confidence box from being averaged out of
        # existence entirely).
        wt = max(tr.confidence, 0.05) if self._conf_weight else 1.0
        self._boxes.append((tr.x, tr.y, tr.w, tr.h, wt))
        x, y, w, h = self._avg()
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
                 smooth: bool = True, smooth_window: int = 5,
                 jump_tol: float = 0.35, jump_confirm: int = 3,
                 lock_hits: int = 30, lock_coast_factor: float = 2.0,
                 conf_weight: bool = True):
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
        # Rolling-average box smoothing + jump gate (see module docstring).
        # Applied per track to everything emitted, so overlay, event geometry
        # and coasting all work from the same calm box.
        self.smooth = smooth
        self.smooth_window = max(1, smooth_window)
        self.jump_tol = jump_tol
        self.jump_confirm = jump_confirm
        self.conf_weight = conf_weight
        # Track lock: hits needed to earn the extended coast window (0 = off)
        # and how much longer a locked track may coast (see module docstring).
        self.lock_hits = max(0, lock_hits)
        self.lock_coast_factor = max(1.0, lock_coast_factor)
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
                    s.smoother = _BoxSmoother(self.smooth_window,
                                              self.jump_tol, self.jump_confirm,
                                              self.conf_weight)
            else:
                s.conf = self.alpha * tr.confidence + (1 - self.alpha) * s.conf
                s.last_seq = seq
                s.hits += 1
            if s.smoother is not None:
                tr = s.smoother.apply(tr, seq)
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
            # Track lock: a track that has proven itself over lock_hits fresh
            # detections earns a longer coast window, so an established vessel
            # rides out a longer dropout (wave occlusion, a wake burst) without
            # losing its box or id; a young track still dies fast.
            coast_limit = self.max_coast_frames
            if self.lock_hits and s.hits >= self.lock_hits:
                coast_limit = int(round(self.max_coast_frames * self.lock_coast_factor))
            if missed > coast_limit or s.conf < conf_off:
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
