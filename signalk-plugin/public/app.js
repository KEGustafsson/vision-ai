// Captain view. Polls the plugin REST API for enriched targets + own-ship and
// shows the annotated MJPEG stream. Same-origin; relies on SignalK auth.

const API = '/plugins/signalk-vision-ai';
let cameras = [];
let activeCamera = null;
// Last activeCamera the plugin reported (driven by context control). The view
// auto-follows it when it changes; a manual click still lets you peek at the
// other camera until the system next switches.
let systemActiveCamera = null;
let ptzCameras = [];
let detectionEnabled = null; // unknown until first poll
let maxTargets = null;
const PTZ_SPEED = 0.6; // normalised ONVIF velocity for held buttons

const rad2deg = (r) => (r * 180) / Math.PI;
const fmtBrg = (r) => (r == null ? '—' : `${Math.round(((rad2deg(r) % 360) + 360) % 360)}°`);
const fmtRng = (m) => (m == null ? '—' : m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
const fmtTcpa = (s) => (s == null || s <= 0 ? '—' : s >= 60 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);

function setStream(camera) {
  activeCamera = camera;
  document.querySelectorAll('.cameras button').forEach((b) => {
    b.classList.toggle('active', b.dataset.cam === camera);
  });
  applyStream();
  updatePtzVisibility();
}

let reconnectTimer = null;

// Whether we currently want a live stream (detection on and a camera selected).
function streamIntended() {
  return detectionEnabled !== false && !!activeCamera;
}

// Point the <img> at the live MJPEG, unless detection is off (the container has
// released the cameras, so there's nothing to show) — then clear it.
function applyStream() {
  const img = document.getElementById('stream');
  clearTimeout(reconnectTimer);
  if (!streamIntended()) {
    img.removeAttribute('src');
  } else {
    img.src = `${API}/stream/${activeCamera}?t=${Date.now()}`;
  }
}

// An <img> MJPEG stream never reconnects on its own: when the vision container
// restarts the connection drops, the <img> fires `error`, and the frame just
// freezes until a manual page reload. Re-point it at a fresh stream URL a moment
// later (and keep retrying — each failed attempt errors again) so the video
// resumes by itself once the container is back.
function onStreamError() {
  if (!streamIntended()) return; // src was cleared on purpose, not a failure
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(applyStream, 2000);
}

// --- Detection master on/off ----------------------------------------------
function renderDetection(on) {
  if (on === detectionEnabled) return;
  detectionEnabled = on;
  const btn = document.getElementById('detToggle');
  btn.hidden = false;
  btn.textContent = `Detection: ${on ? 'ON' : 'OFF'}`;
  btn.classList.toggle('off', !on);
  applyStream(); // stop the stream when off, resume it when back on
}

async function toggleDetection() {
  if (detectionEnabled === null) return;
  const next = !detectionEnabled;
  const btn = document.getElementById('detToggle');
  btn.disabled = true;
  try {
    await fetch(`${API}/detection`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ enabled: next }),
    });
    renderDetection(next); // optimistic; the poll loop reconciles either way
  } catch {
    /* leave state alone; the next poll will reflect the real value */
  } finally {
    btn.disabled = false;
  }
}

// --- Container target limit --------------------------------------------------
function renderTargetLimit(value) {
  const select = document.getElementById('targetLimit');
  if (!Number.isFinite(value) || value <= 0) return;
  const asText = String(value);
  if (![...select.options].some((opt) => opt.value === asText)) {
    select.add(new Option(asText, asText));
    [...select.options]
      .sort((a, b) => Number(a.value) - Number(b.value))
      .forEach((opt) => select.add(opt));
  }
  maxTargets = value;
  select.value = asText;
  select.hidden = false;
}

async function setTargetLimit(value) {
  if (!Number.isInteger(value) || value <= 0) return;
  const select = document.getElementById('targetLimit');
  select.disabled = true;
  try {
    await fetch(`${API}/target-limit`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ maxTargets: value }),
    });
    renderTargetLimit(value);
  } catch {
    if (maxTargets != null) renderTargetLimit(maxTargets);
  } finally {
    select.disabled = false;
  }
}

// --- ONVIF PTZ -------------------------------------------------------------
// Show the control pad only for PTZ-capable cameras. The list comes from the
// container (via the plugin proxy) and only resolves once the container is up,
// so we (re)load it whenever the camera set changes.
async function loadPtzCameras() {
  try {
    const r = await fetch(`${API}/ptz`).then((res) => res.json());
    ptzCameras = r.cameras || [];
  } catch {
    ptzCameras = [];
  }
  updatePtzVisibility();
}

function updatePtzVisibility() {
  document.getElementById('ptzPad').hidden = !ptzCameras.includes(activeCamera);
}

async function ptzSend(body) {
  if (!activeCamera) return;
  try {
    await fetch(`${API}/ptz/${activeCamera}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    /* transient; the camera's ONVIF Timeout auto-stops it anyway */
  }
}

let ptzHold = null;
function ptzStartMove(pan, tilt, zoom) {
  const body = { action: 'move', pan: pan * PTZ_SPEED, tilt: tilt * PTZ_SPEED, zoom: zoom * PTZ_SPEED };
  ptzSend(body);
  // Re-send while held so the camera's short auto-stop Timeout doesn't halt
  // motion mid-press (and motion stops within ~2s if the page goes away).
  clearInterval(ptzHold);
  ptzHold = setInterval(() => ptzSend(body), 1000);
}
function ptzStopMove() {
  if (ptzHold === null) return;
  clearInterval(ptzHold);
  ptzHold = null;
  ptzSend({ action: 'stop' });
}

function wirePtzPad() {
  document.querySelectorAll('#ptzPad button').forEach((btn) => {
    if (btn.dataset.home) {
      btn.addEventListener('click', () => ptzSend({ action: 'home' }));
      return;
    }
    const pan = Number(btn.dataset.pan);
    const tilt = Number(btn.dataset.tilt);
    const zoom = Number(btn.dataset.zoom);
    btn.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      ptzStartMove(pan, tilt, zoom);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) =>
      btn.addEventListener(ev, ptzStopMove)
    );
  });
}

function renderCameras(list) {
  if (JSON.stringify(list) === JSON.stringify(cameras)) return;
  cameras = list;
  loadPtzCameras();
  const box = document.getElementById('cameraButtons');
  box.innerHTML = '';
  list.forEach((cam) => {
    const b = document.createElement('button');
    b.textContent = cam;
    b.dataset.cam = cam;
    b.onclick = () => setStream(cam);
    box.appendChild(b);
  });
  if (!activeCamera && list.length) setStream(list[0]);
}

// Open the camera the plugin currently marks active (context control picks it
// from speed/time). Only acts on a *change*, so a manual peek at the other
// camera isn't yanked back on every poll — only when the active camera flips.
function followActiveCamera(cam) {
  if (!cam || cam === systemActiveCamera) return;
  systemActiveCamera = cam;
  if (cameras.includes(cam)) setStream(cam);
}

function renderOwnShip(own) {
  const el = document.getElementById('ownship');
  if (!own || !own.position) {
    el.textContent = 'no position';
    return;
  }
  // Show enough precision that small movement is visible: 5 decimal degrees
  // (~1 m), 0.1° heading, 0.01 kn. At a dock a stationary boat still looks
  // near-constant — that's real, not a frozen feed (see the "updated" clock).
  const hdg = own.headingTrue != null ? `${rad2deg(own.headingTrue).toFixed(1)}°` : '—';
  const sog = own.sog != null ? `${(own.sog * 1.94384).toFixed(2)} kn` : '—';
  el.textContent = `${own.position.latitude.toFixed(5)}, ${own.position.longitude.toFixed(5)} · HDG ${hdg} · SOG ${sog}`;
}

function renderTargets(targets) {
  document.getElementById('targetCount').textContent = targets.length;
  const tbody = document.getElementById('targetRows');
  tbody.innerHTML = '';
  targets
    .slice()
    .sort((a, b) => threatRank(b) - threatRank(a))
    .forEach((t) => {
      const tr = document.createElement('tr');
      tr.className = `threat-${t.threatLevel}${t.is_person_in_water ? ' mob' : ''}`;
      tr.innerHTML = `
        <td>${t.is_person_in_water ? '🆘 ' : ''}${t.label}${t.track_id != null ? ' #' + t.track_id : ''}</td>
        <td>${fmtBrg(t.bearingTrue)}</td>
        <td>${fmtRng(t.geometry.range_m)}</td>
        <td>${fmtRng(t.cpa)}</td>
        <td>${fmtTcpa(t.tcpa)}</td>
        <td>${t.aisCorrelated ? '✔ ' + (t.aisMmsi || '') : '<span class="dark">DARK</span>'}</td>
        <td><span class="dot ${t.threatLevel}"></span>${t.threatLevel}</td>`;
      tbody.appendChild(tr);
    });
}

function threatRank(t) {
  if (t.is_person_in_water) return 4;
  return { high: 3, medium: 2, low: 1, none: 0 }[t.threatLevel] ?? 0;
}

async function poll() {
  try {
    // Camera list comes from /targets' `system` block, not /config: SignalK
    // reserves GET /plugins/<id>/config for the plugin's own settings, which
    // shadows the plugin router's /config — so cfg.cameras would be undefined.
    // no-store: the response carries a weak ETag and no Cache-Control, so
    // without this the browser could serve a stale/304 body and freeze the UI.
    const data = await fetch(`${API}/targets`, { cache: 'no-store' }).then((r) => r.json());
    if (data.system && typeof data.system.detectionEnabled === 'boolean') {
      renderDetection(data.system.detectionEnabled);
    }
    if (data.system && Number.isFinite(data.system.maxTargets)) {
      renderTargetLimit(data.system.maxTargets);
    }
    renderCameras((data.system && data.system.cameras) || []);
    if (data.system) followActiveCamera(data.system.activeCamera);
    renderOwnShip(data.ownShip);
    renderTargets(data.targets || []);
    // The trailing clock ticks every poll, so it's obvious the feed is live
    // even when a stationary boat's nav values don't change.
    document.getElementById('status').textContent = detectionEnabled === false
      ? `detection off · updated ${new Date().toLocaleTimeString()}`
      : `${(data.targets || []).length} targets · ${activeCamera || '—'} · updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    document.getElementById('status').textContent = 'plugin offline';
  }
}

// Self-scheduling so a slow SignalK server can't pile up overlapping polls.
async function loop() {
  await poll();
  setTimeout(loop, 1000);
}
document.getElementById('stream').addEventListener('error', onStreamError);
document.getElementById('detToggle').addEventListener('click', toggleDetection);
document.getElementById('targetLimit').addEventListener('change', (e) => {
  setTargetLimit(Number(e.target.value));
});
wirePtzPad();
loop();
