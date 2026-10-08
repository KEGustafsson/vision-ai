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

// When to call navigation.attitude missing, as opposed to not here YET.
//
// After a SignalK restart the plugin starts before the NMEA gateways have
// delivered a single sample, so an immediate check reports a perfectly healthy
// IMU as missing on every boot. A fixed boot timer doesn't settle it either: how
// long the bus takes to come up depends on the gateway, not on us. So the watch
// takes its cue from the boat's own data instead — once own-ship navigation
// (position/heading) is flowing, the sensors are talking, and an attitude that
// still hasn't shown up ATTITUDE_SETTLE_MS later genuinely isn't coming. If the
// bus never comes up at all, ATTITUDE_BOOT_CAP_MS still reports it. An attitude
// that was present and then disappears is a real loss and is reported at once.
//
// Logging only: compensation itself is unchanged — nothing is pushed while there
// is no attitude, and the container keeps the uncompensated horizon.
export const ATTITUDE_SETTLE_MS = 15_000;
export const ATTITUDE_BOOT_CAP_MS = 120_000;

/** What changed this tick, for the caller to log; null when nothing did. */
export type AttitudeReport = 'arrived' | 'never-arrived' | 'lost' | 'resumed' | null;

export class AttitudeWatch {
  private state: 'warming' | 'ok' | 'missing' = 'warming';
  private navSince: number | null = null;

  constructor(
    private readonly startedAt: number,
    private readonly settleMs = ATTITUDE_SETTLE_MS,
    private readonly bootCapMs = ATTITUDE_BOOT_CAP_MS
  ) {}

  /**
   * `available`: a usable attitude this tick. `navFlowing`: whether own-ship
   * navigation is currently fresh — only consulted while still warming up.
   */
  update(available: boolean, navFlowing: () => boolean, now: number): AttitudeReport {
    if (available) {
      const prev = this.state;
      this.state = 'ok';
      if (prev === 'warming') return 'arrived';
      return prev === 'missing' ? 'resumed' : null;
    }
    if (this.state === 'ok') {
      this.state = 'missing';
      return 'lost';
    }
    if (this.state === 'missing') return null;
    if (this.navSince === null && navFlowing()) this.navSince = now;
    const settled = this.navSince !== null && now - this.navSince >= this.settleMs;
    if (settled || now - this.startedAt >= this.bootCapMs) {
      this.state = 'missing';
      return 'never-arrived';
    }
    return null;
  }
}
