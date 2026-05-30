// Captain view. Polls the plugin REST API for enriched targets + own-ship and
// shows the annotated MJPEG stream. Same-origin; relies on SignalK auth.

const API = '/plugins/signalk-vision-ai';
let cameras = [];
let activeCamera = null;

const rad2deg = (r) => (r * 180) / Math.PI;
const fmtBrg = (r) => (r == null ? '—' : `${Math.round(((rad2deg(r) % 360) + 360) % 360)}°`);
const fmtRng = (m) => (m == null ? '—' : m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${Math.round(m)} m`);
const fmtTcpa = (s) => (s == null || s <= 0 ? '—' : s >= 60 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);

function setStream(camera) {
  activeCamera = camera;
  const img = document.getElementById('stream');
  img.src = `${API}/stream/${camera}?t=${Date.now()}`;
  document.querySelectorAll('.cameras button').forEach((b) => {
    b.classList.toggle('active', b.dataset.cam === camera);
  });
}

function renderCameras(list) {
  if (JSON.stringify(list) === JSON.stringify(cameras)) return;
  cameras = list;
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

function renderOwnShip(own) {
  const el = document.getElementById('ownship');
  if (!own || !own.position) {
    el.textContent = 'no position';
    return;
  }
  const hdg = own.headingTrue != null ? `${Math.round(rad2deg(own.headingTrue))}°` : '—';
  const sog = own.sog != null ? `${(own.sog * 1.94384).toFixed(1)} kn` : '—';
  el.textContent = `${own.position.latitude.toFixed(4)}, ${own.position.longitude.toFixed(4)} · HDG ${hdg} · SOG ${sog}`;
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
    const data = await fetch(`${API}/targets`).then((r) => r.json());
    renderCameras((data.system && data.system.cameras) || []);
    renderOwnShip(data.ownShip);
    renderTargets(data.targets || []);
    document.getElementById('status').textContent =
      `${(data.targets || []).length} targets · ${activeCamera || '—'}`;
  } catch (e) {
    document.getElementById('status').textContent = 'plugin offline';
  }
}

// Self-scheduling so a slow SignalK server can't pile up overlapping polls.
async function loop() {
  await poll();
  setTimeout(loop, 1000);
}
loop();
