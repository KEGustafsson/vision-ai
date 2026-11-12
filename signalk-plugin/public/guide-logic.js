// The setup guide's decisions, kept free of the DOM so they can be tested
// (test/setupGuide.test.ts) as well as run in the browser (setup.js).
//
// `status` is /targets' `container` (src/containerStatus.ts). `state` is the
// guide's own: { open, hiddenFor, openAfterS } — whether it is showing, the
// outage (downSince) the user hid it for, and how long a container that has
// answered before may be silent before the guide opens by itself.

(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.visionGuideLogic = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /**
   * Should the guide open by itself now? At once for a container that has
   * never answered since the plugin started; after `openAfterS` for one that
   * went silent, so a restart or a TensorRT engine build does not flash it up;
   * never while it is open, or for the outage the user hid it for.
   */
  function shouldAutoOpen(status, state) {
    if (!status || status.reachable !== false) return false;
    if (state.open) return false;
    if (state.hiddenFor === status.downSince) return false;
    return !status.everConnected || status.downForS >= state.openAfterS;
  }

  /**
   * What to do with a /setup answer requested for the outage `requestedFor`,
   * now that it has arrived and `status` is the latest container status:
   *  - 'discard'  the container answered meanwhile: show nothing;
   *  - 'refetch'  a newer outage began and the user asked for the guide:
   *               fetch that outage's metadata and show it;
   *  - 'wait'     an automatic open that is no longer (or not yet) due, for
   *               instance a newer outage still inside its delay: leave it to
   *               the next poll, which fetches for the current outage;
   *  - 'show'     show the guide with this answer.
   */
  function onSetupAnswer(requestedFor, status, manual, state) {
    if (!status || status.reachable !== false) return 'discard';
    if (status.downSince !== requestedFor) return manual ? 'refetch' : 'wait';
    if (!manual && !shouldAutoOpen(status, state)) return 'wait';
    return 'show';
  }

  return { shouldAutoOpen, onSetupAnswer };
});
