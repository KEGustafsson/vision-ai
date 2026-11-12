// Container setup guide. app.js calls visionSetup.update(container) on every
// /targets poll with the plugin's view of the vision container (see
// src/containerStatus.ts). While the container does not answer, this shows a
// banner, and opens a step-by-step guide for installing and starting the
// container on the user's hardware:
//   - at once if the container has never answered since the plugin started
//     (a new install, or a container that is not set up);
//   - after OPEN_AFTER_S if it was connected and went silent, so a container
//     restart or a TensorRT engine build does not flash the guide up.
// It closes by itself when the container answers. "Hide" hides it for the
// current outage only; the banner stays and reopens it.
//
// The commands download the compose file of the GitHub release whose version
// matches the plugin (GET /setup), so the container speaks the plugin's
// contract. Camera addresses typed into the guide stay in this page: they are
// never sent anywhere or stored.

(function () {
  'use strict';

  const API = '/plugins/signalk-vision-ai';
  const OPEN_AFTER_S = 30;
  const DIR = '~/vision-ai';

  let setup = null; // GET /setup, fetched again for every new outage
  let setupFor; // the outage (downSince) `setup` was fetched for
  let setupPending = null;
  let renderedVersion = null; // plugin version the rendered commands are for
  let opening = false;
  let current = null; // hardware choice id; null until known or chosen
  let lastStatus = null;
  let hiddenFor = null; // downSince of the outage the user hid the guide for
  let wasOpen = false;
  let focusBefore = null; // where keyboard focus was before the guide opened

  const $ = (id) => document.getElementById(id);

  const step = {
    runtime: () => ({
      t: 'Check Docker and the NVIDIA runtime',
      cmd: 'docker info | grep -i runtime',
      e: 'the list includes <code>nvidia</code>. If not, install <code>nvidia-container-toolkit</code> and restart Docker.',
    }),
    docker: () => ({
      t: 'Check Docker',
      cmd: 'docker --version && docker compose version',
      e: 'two version lines. If not, install Docker Engine with the Compose plugin first.',
    }),
    download: (file) => ({
      t: 'Download the compose file',
      cmd: `mkdir -p ${DIR} && cd ${DIR}\ncurl -fsSLO ${setup.downloadUrl}/${file}`,
      d: `From release <a href="${setup.releaseUrl}" target="_blank" rel="noopener">${setup.releaseTag}</a>, the version that matches this plugin. Run the later steps in <code>${DIR}</code> too.`,
    }),
    env: () => ({
      t: 'Save the camera addresses',
      cmd: "(umask 077\ntouch .env && chmod 600 .env && cat > .env <<'EOF'\nVISION_CAMERA_FORWARD_URL={{FWD}}\nVISION_CAMERA_AFT_URL={{AFT}}\nEOF\n)",
      d: 'Docker reads <code>.env</code> every time the container starts, so the addresses survive a reboot. Both cameras are required. The file is made readable by you only first, because the addresses hold camera passwords.',
    }),
    up: (file, slow) => ({
      t: 'Start the container',
      cmd: `docker compose -f ${file} up -d`,
      d: 'It starts again by itself after every reboot.',
      slow,
    }),
    verify: () => ({
      t: 'Check that it answers',
      cmd: 'curl -s http://127.0.0.1:7000/health',
      e: 'a reply with <code>"status": "ok"</code>. This window then closes by itself within a few seconds.',
    }),
  };

  const HW = [
    {
      id: 'orin-ds', name: 'Jetson Orin · DeepStream', tag: 'Recommended',
      meta: 'JetPack 6 (L4T R36). The whole pipeline runs on the GPU. Best frame rate.',
      doc: 'docs/jetson-deepstream.md', cams: true,
      steps: () => [
        { t: 'Confirm the JetPack version', cmd: 'head -1 /etc/nv_tegra_release', e: 'the line starts with <code># R36</code>.' },
        step.runtime(),
        { t: 'Set full power mode', cmd: 'sudo nvpmodel -m 0 && sudo jetson_clocks', d: 'Optional, but detection runs noticeably slower without it.' },
        step.download('docker-compose.deepstream.yml'),
        step.env(),
        step.up('docker-compose.deepstream.yml',
          'The first start downloads several GB, then builds the TensorRT engine, which takes several minutes. Later starts take seconds.'),
        step.verify(),
      ],
    },
    {
      id: 'xavier-ds', name: 'Jetson Xavier · DeepStream',
      meta: 'JetPack 5 (L4T R35). Xavier NX or AGX Xavier.',
      doc: 'docs/jetson-deepstream.md#xavier-nx-notes', cams: true,
      steps: () => [
        { t: 'Confirm the JetPack version', cmd: 'head -1 /etc/nv_tegra_release', e: 'the line starts with <code># R35</code>.' },
        step.runtime(),
        { t: 'Set full power mode', cmd: 'sudo nvpmodel -m 8 && sudo jetson_clocks\nnvpmodel -q', d: 'Mode 8 is 20 W, 6 cores on the Xavier NX dev kit. Other boards use other mode numbers.' },
        step.download('docker-compose.deepstream-xavier.yml'),
        step.env(),
        step.up('docker-compose.deepstream-xavier.yml',
          'The first start downloads several GB, then builds the TensorRT engine. The Xavier is short on RAM: stop other heavy containers until it is done.'),
        step.verify(),
      ],
    },
    {
      id: 'orin-trt', name: 'Jetson Orin · TensorRT',
      meta: 'JetPack 6. Simpler pipeline: GPU detection, CPU tracking.',
      doc: 'docs/jetson-setup.md', cams: true,
      steps: () => [
        { t: 'Confirm the JetPack version', cmd: 'head -1 /etc/nv_tegra_release', e: 'the line starts with <code># R36</code>.' },
        step.runtime(),
        step.download('docker-compose.jetson.yml'),
        step.env(),
        {
          t: 'Build the TensorRT engine on this board',
          cmd: 'docker compose -f docker-compose.jetson.yml run --rm vision-service \\\n  python3 scripts/export_engine.py --weights models/yolo11n.pt --imgsz 640',
          e: 'it ends by naming <code>models/yolo11n.engine</code>.',
          slow: 'Takes several minutes. An engine only runs on the board and TensorRT that built it, so it is built here and again after each update.',
        },
        step.up('docker-compose.jetson.yml'),
        step.verify(),
      ],
    },
    {
      id: 'cpu', name: 'CPU only',
      meta: 'Any x86_64 or arm64 computer with Docker. Real detection, about 5 frames per second.',
      doc: 'docs/release.md', cams: true,
      steps: () => [
        step.docker(),
        step.download('docker-compose.cpu.yml'),
        step.env(),
        step.up('docker-compose.cpu.yml', 'The first start downloads about 1 GB.'),
        step.verify(),
      ],
    },
    {
      id: 'mock', name: 'Mock', tag: 'Demo only',
      meta: 'No cameras or GPU. Made-up boats for trying the plugin out.',
      doc: 'docs/dev-quickstart.md', cams: false,
      steps: () => [
        step.docker(),
        step.download('docker-compose.mock.yml'),
        step.up('docker-compose.mock.yml'),
        step.verify(),
      ],
    },
  ];

  const HOST_TEXT = {
    'orin-ds': 'Jetson Orin, JetPack 6',
    'xavier-ds': 'Jetson Xavier, JetPack 5',
    cpu: 'a computer without a Jetson GPU',
  };

  // The release the commands point at must match the plugin, and the plugin
  // can be updated while this page stays open, so /setup is fetched again for
  // every new outage rather than once per page load.
  function fetchSetup(outage) {
    if (setup && setupFor === outage) return Promise.resolve(setup);
    if (!setupPending) {
      setupPending = fetch(`${API}/setup`, { cache: 'no-store' })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then((s) => { setup = s; setupFor = outage; return s; })
        .finally(() => { setupPending = null; });
    }
    return setupPending;
  }

  // ---- rendering -----------------------------------------------------------

  function renderHost() {
    const chip = $('csDetected');
    const tip = $('csInContainer');
    const host = setup && setup.host;
    tip.hidden = !(host && host.inContainer);
    if (host && host.recommended) {
      const l4t = host.l4t ? ` · L4T R${host.l4t.major}${host.l4t.revision ? '.' + host.l4t.revision : ''}` : '';
      chip.textContent = '';
      chip.append('This Signal K computer: ');
      const s = document.createElement('span');
      s.textContent = `${host.arch}${l4t} → ${HOST_TEXT[host.recommended]}`;
      chip.append(s);
      chip.hidden = false;
    } else {
      chip.hidden = true;
    }
  }

  function renderChoices() {
    const box = $('csHw');
    box.textContent = '';
    for (const h of HW) {
      const lab = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'csHw';
      input.id = `cs-hw-${h.id}`;
      input.value = h.id;
      input.checked = h.id === current;
      const name = document.createElement('span');
      name.className = 'cs-hw-name';
      name.textContent = h.name;
      if (h.tag) {
        const tag = document.createElement('span');
        tag.className = h.id === 'mock' ? 'cs-tag cs-tag-demo' : 'cs-tag cs-tag-rec';
        tag.textContent = h.tag;
        name.append(tag);
      }
      const meta = document.createElement('span');
      meta.className = 'cs-hw-meta';
      meta.textContent = h.meta;
      lab.append(input, name, meta);
      box.append(lab);
    }
  }

  function renderSteps() {
    const h = HW.find((x) => x.id === current);
    if (!h) {
      // No hint about the computer (Signal K in a container, or an unknown
      // board): show no commands until the user picks the hardware, rather
      // than guess and hand them the wrong image.
      $('csCamSec').hidden = true;
      $('csStepsNum').textContent = '2';
      $('csMockWarn').hidden = true;
      $('csDoc').href = 'https://github.com/KEGustafsson/vision-ai/blob/main/docs/release.md';
      const li = document.createElement('li');
      li.className = 'cs-choose';
      li.textContent = 'Choose the hardware above to see its commands.';
      $('csSteps').replaceChildren(li);
      return;
    }
    $('csCamSec').hidden = !h.cams;
    $('csStepsNum').textContent = h.cams ? '3' : '2';
    $('csMockWarn').hidden = h.id !== 'mock';
    $('csDoc').href = `https://github.com/KEGustafsson/vision-ai/blob/main/${h.doc}`;
    const list = $('csSteps');
    list.textContent = '';
    for (const s of h.steps()) {
      const li = document.createElement('li');
      const body = document.createElement('div');
      body.className = 'cs-st';
      const t = document.createElement('div');
      t.className = 'cs-st-t';
      t.textContent = s.t;
      body.append(t);
      // d/e/slow are this file's own fixed strings (links and <code> only).
      if (s.d) body.insertAdjacentHTML('beforeend', `<div class="cs-st-d">${s.d}</div>`);
      if (s.cmd) {
        const code = document.createElement('div');
        code.className = 'cs-code';
        const pre = document.createElement('pre');
        pre.dataset.tpl = s.cmd;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'cs-copy';
        btn.textContent = 'Copy';
        code.append(pre, btn);
        body.append(code);
      }
      if (s.e) body.insertAdjacentHTML('beforeend', `<div class="cs-expect"><span>${s.e}</span></div>`);
      if (s.slow) {
        const slow = document.createElement('div');
        slow.className = 'cs-slow';
        slow.textContent = s.slow;
        body.append(slow);
      }
      li.append(body);
      list.append(li);
    }
    fillCommands();
  }

  // A value for Docker Compose's .env file. Compose substitutes $NAME in
  // unquoted and double-quoted values, which would change a password such as
  // "pa$word", so single-quote it (taken literally). A value that itself
  // contains ' goes in double quotes with \, " and $ escaped.
  function envQuote(value) {
    if (!value.includes("'")) return `'${value}'`;
    return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\$/g, '$$$$')}"`;
  }

  function fillCommands() {
    const fwd = envQuote($('csFwd').value.trim() || 'rtsp://user:password@192.168.1.10:554/stream');
    const aft = envQuote($('csAft').value.trim() || 'rtsp://user:password@192.168.1.11:554/stream');
    document.querySelectorAll('#csSteps pre[data-tpl]').forEach((pre) => {
      // textContent: whatever is typed into the address fields never becomes
      // markup. A replacement callback, not a string: in a replacement string
      // "$$", "$&" and friends are patterns, which would undo envQuote's
      // escaping and paste the placeholder into a password containing "$&".
      pre.textContent = pre.dataset.tpl.replace(/\{\{(FWD|AFT)\}\}/g, (_, cam) => (cam === 'FWD' ? fwd : aft));
    });
  }

  function renderStatus(c) {
    let text;
    if (c.everConnected) {
      text = `Not answering for ${fmtDuration(c.downForS)}`;
    } else {
      text = 'Has not answered since Signal K started';
    }
    if (c.lastError) text += ` · ${c.lastError}`;
    text += ' · checking every 5 s';
    $('csConnText').textContent = text;
    $('csUrl').textContent = c.url;
  }

  function fmtDuration(s) {
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
  }

  // ---- open / close ----------------------------------------------------------

  function open() {
    if (opening) return;
    opening = true;
    const outage = lastStatus ? lastStatus.downSince : null;
    fetchSetup(outage).then(
      () => {
        opening = false;
        show();
      },
      () => {
        // Without /setup the guide can't name the release; leave the banner up.
        opening = false;
      }
    );
  }

  function show() {
    if (renderedVersion !== setup.version) {
      // First open, or the plugin was updated since: rebuild for its release.
      // Pre-select only a real hint; the user's own choice is kept.
      if (renderedVersion === null) current = (setup.host && setup.host.recommended) || null;
      renderedVersion = setup.version;
      renderHost();
      renderChoices();
      renderSteps();
    }
    $('csSetupView').hidden = false;
    $('csOkView').hidden = true;
    if ($('csScrim').hidden) {
      focusBefore = document.activeElement;
      $('csScrim').hidden = false;
      setBackgroundInert(true);
      // Into the dialog: the selected hardware choice, else the dialog itself.
      const chosen = document.querySelector('#csHw input:checked');
      (chosen || $('csDialog')).focus();
    }
    wasOpen = true;
  }

  // While the guide is open, the page behind it can be neither clicked nor
  // reached with Tab, so keyboard focus stays in the dialog.
  function setBackgroundInert(on) {
    document.querySelectorAll('body > header, body > main, #csBanner').forEach((el) => {
      el.inert = on;
    });
  }

  function closeScrim() {
    if ($('csScrim').hidden) return;
    $('csScrim').hidden = true;
    setBackgroundInert(false);
    // Back where it was. Otherwise (an automatic open with nothing focused, or
    // the opener has since been hidden, as the banner is once the container
    // answers) to the first visible control: the banner's button while the
    // container is still down, else the page's own controls.
    const candidates = [focusBefore, $('csBannerOpen'), $('detToggle'), document.querySelector('.cameras button')];
    focusBefore = null;
    const back = candidates.find((el) =>
      el && el !== document.body && document.contains(el) && typeof el.focus === 'function' &&
      !el.closest('[hidden]') && el.getClientRects().length > 0);
    if (back) back.focus();
  }

  function hide() {
    closeScrim();
    wasOpen = false;
    if (lastStatus && lastStatus.reachable === false) hiddenFor = lastStatus.downSince;
  }

  function connected() {
    $('csBanner').hidden = true;
    if (!wasOpen) return;
    $('csSetupView').hidden = true;
    $('csOkView').hidden = false;
    $('csDialog').focus(); // the focused control was in the view just hidden
    wasOpen = false;
    setTimeout(closeScrim, 2500);
  }

  /** Called by app.js with /targets' `container`; undefined from an older plugin. */
  function update(c) {
    if (!c) return;
    lastStatus = c;
    if (c.reachable === true) {
      hiddenFor = null;
      connected();
      return;
    }
    if (c.reachable !== false) return; // not checked yet
    $('csBanner').hidden = false;
    $('csBannerText').textContent = c.everConnected
      ? `Vision container not answering for ${fmtDuration(c.downForS)}.`
      : 'Vision container is not running.';
    renderStatus(c);
    const due = !c.everConnected || c.downForS >= OPEN_AFTER_S;
    if (due && !wasOpen && hiddenFor !== c.downSince) open();
  }

  // ---- events ----------------------------------------------------------------

  function copy(btn) {
    const pre = btn.parentElement.querySelector('pre');
    const text = pre.textContent;
    const done = () => {
      btn.textContent = 'Copied';
      btn.classList.add('cs-done');
      setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('cs-done'); }, 1400);
    };
    const select = () => {
      const r = document.createRange();
      r.selectNodeContents(pre);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      return sel;
    };
    // The Clipboard API exists only on https or localhost; a Signal K server on
    // the boat's LAN is usually plain http, so fall back to execCommand.
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, () => { select(); btn.textContent = 'Press Ctrl+C'; });
      return;
    }
    const sel = select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    if (ok) { sel.removeAllRanges(); done(); } else { btn.textContent = 'Press Ctrl+C'; }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('csHw').addEventListener('change', (e) => { current = e.target.value; renderSteps(); });
    $('csFwd').addEventListener('input', fillCommands);
    $('csAft').addEventListener('input', fillCommands);
    $('csClose').addEventListener('click', hide);
    $('csHide').addEventListener('click', hide);
    $('csBannerOpen').addEventListener('click', () => { hiddenFor = null; open(); });
    $('csSteps').addEventListener('click', (e) => {
      const b = e.target.closest('.cs-copy');
      if (b) copy(b);
    });
    ['csWhich', 'csRemote'].forEach((id) => $(id).addEventListener('click', (e) => {
      const b = e.target.closest('.cs-copy');
      if (b) copy(b);
    }));
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !$('csScrim').hidden) hide();
    });
  });

  window.visionSetup = { update };
})();
