// The setup guide's decisions (public/guide-logic.js): when it opens by
// itself, and what it does with a /setup answer that arrives after the
// container status has moved on.

import { createRequire } from 'module';
import { describe, expect, it } from 'vitest';

const require = createRequire(import.meta.url);
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { shouldAutoOpen, onSetupAnswer } = require('../public/guide-logic.js');

const OUTAGE_1 = '2026-11-12T15:00:00.000Z';
const OUTAGE_2 = '2026-11-12T15:05:00.000Z';
const closed = { open: false, hiddenFor: null, openAfterS: 30 };

function down(downSince: string, downForS: number, everConnected: boolean) {
  return { url: 'http://127.0.0.1:7000/', reachable: false, everConnected, downSince, downForS, lastError: 'ECONNREFUSED' };
}
const up = { url: 'http://127.0.0.1:7000/', reachable: true, everConnected: true, downSince: null, downForS: 0, lastError: null };

describe('shouldAutoOpen', () => {
  it('opens at once for a container that never answered', () => {
    expect(shouldAutoOpen(down(OUTAGE_1, 0, false), closed)).toBe(true);
  });

  it('waits openAfterS for a container that went silent', () => {
    expect(shouldAutoOpen(down(OUTAGE_1, 29, true), closed)).toBe(false);
    expect(shouldAutoOpen(down(OUTAGE_1, 30, true), closed)).toBe(true);
  });

  it('never opens while reachable, not yet checked, already open, or hidden for this outage', () => {
    expect(shouldAutoOpen(up, closed)).toBe(false);
    expect(shouldAutoOpen({ ...up, reachable: null }, closed)).toBe(false);
    expect(shouldAutoOpen(down(OUTAGE_1, 60, true), { ...closed, open: true })).toBe(false);
    expect(shouldAutoOpen(down(OUTAGE_1, 60, true), { ...closed, hiddenFor: OUTAGE_1 })).toBe(false);
    // Hiding is per outage: the next one opens again.
    expect(shouldAutoOpen(down(OUTAGE_2, 60, true), { ...closed, hiddenFor: OUTAGE_1 })).toBe(true);
  });
});

describe('onSetupAnswer: a /setup answer that arrives late', () => {
  it('shows the guide when nothing changed meanwhile', () => {
    expect(onSetupAnswer(OUTAGE_1, down(OUTAGE_1, 0, false), false, closed)).toBe('show');
    expect(onSetupAnswer(OUTAGE_1, down(OUTAGE_1, 45, true), false, closed)).toBe('show');
  });

  it('shows nothing if the container answered while /setup loaded', () => {
    expect(onSetupAnswer(OUTAGE_1, up, false, closed)).toBe('discard');
    expect(onSetupAnswer(OUTAGE_1, up, true, closed)).toBe('discard');
  });

  it('never shows a past outage release: an automatic open waits for the next poll', () => {
    // A newer outage began (say after a plugin upgrade) while /setup loaded.
    expect(onSetupAnswer(OUTAGE_1, down(OUTAGE_2, 45, true), false, closed)).toBe('wait');
  });

  it('keeps the delay for a newer outage of a container that answered before', () => {
    // Only 3 s into the new outage: the automatic open must not happen now.
    const status = down(OUTAGE_2, 3, true);
    expect(onSetupAnswer(OUTAGE_1, status, false, closed)).toBe('wait');
    expect(shouldAutoOpen(status, closed)).toBe(false);
    // ...and once the delay has passed, the next poll opens it for OUTAGE_2.
    expect(shouldAutoOpen(down(OUTAGE_2, 30, true), closed)).toBe(true);
    expect(onSetupAnswer(OUTAGE_2, down(OUTAGE_2, 31, true), false, closed)).toBe('show');
  });

  it('refetches at once for a newer outage when the user asked for the guide', () => {
    expect(onSetupAnswer(OUTAGE_1, down(OUTAGE_2, 3, true), true, closed)).toBe('refetch');
  });

  it('opens on request inside the delay, and despite a hide for this outage', () => {
    expect(onSetupAnswer(OUTAGE_2, down(OUTAGE_2, 3, true), true, closed)).toBe('show');
    expect(onSetupAnswer(OUTAGE_2, down(OUTAGE_2, 3, true), true, { ...closed, hiddenFor: OUTAGE_2 })).toBe('show');
  });

  it('does not open automatically for an outage the user hid while /setup loaded', () => {
    expect(onSetupAnswer(OUTAGE_1, down(OUTAGE_1, 60, true), false, { ...closed, hiddenFor: OUTAGE_1 })).toBe('wait');
  });
});
