// Low-pass filter for the boat attitude forwarded to the vision container for
// horizon compensation.
//
// What the cameras need corrected is the SLOW part of the attitude: trim at the
// dock vs underway, squat and planing, load, heel on a tack. Wave-induced pitch
// and roll (periods of a few seconds) reach the container late — SignalK, the
// HTTP push and the camera's own RTSP latency add up to a few hundred ms — and a
// correction applied that far out of phase can exceed the motion it cancels.
// A first-order low-pass keeps the slow part and leaves waves essentially
// uncompensated (as they are today). Time-weighted, so an irregular poll or
// sensor rate doesn't change the effective time constant. tauS = 0 passes the
// raw attitude through for a low-latency, high-rate install.

import { Attitude } from './types';

export class AttitudeSmoother {
  private state: Attitude | null = null;
  private lastAt = 0;

  constructor(private readonly tauS: number) {}

  reset(): void {
    this.state = null;
  }

  /**
   * Feed one sample (radians) taken at `now` (epoch ms); returns the smoothed
   * attitude. A null sample (missing/stale) resets the filter and returns null,
   * so a gap never resumes from an old heel — the next valid sample restarts it.
   */
  update(sample: Attitude | null, now: number): Attitude | null {
    if (!sample) {
      this.state = null;
      return null;
    }
    const s = this.state;
    if (!s || this.tauS <= 0) {
      this.state = { pitch: sample.pitch, roll: sample.roll };
    } else {
      const dtS = Math.max(0, (now - this.lastAt) / 1000);
      const a = 1 - Math.exp(-dtS / this.tauS);
      this.state = {
        pitch: s.pitch + a * (sample.pitch - s.pitch),
        roll: s.roll + a * (sample.roll - s.roll),
      };
    }
    this.lastAt = now;
    return { ...this.state };
  }
}
