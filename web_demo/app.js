/* ---- Per-browser session ID (shared across tabs, survives reload) ---- */
function makeSessionId() {
  if (typeof crypto !== 'undefined') {
    if (typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
    if (typeof crypto.getRandomValues === 'function') {
      const bytes = new Uint8Array(16);
      crypto.getRandomValues(bytes);
      bytes[6] = (bytes[6] & 0x0f) | 0x40;
      bytes[8] = (bytes[8] & 0x3f) | 0x80;
      const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
      return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
    }
  }
  return `sid-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

const SESSION_ID = (() => {
  try {
    const existing = localStorage.getItem('session_id');
    if (existing) return existing;
    const sid = makeSessionId();
    localStorage.setItem('session_id', sid);
    return sid;
  } catch {
    return makeSessionId();
  }
})();

let latestBundle = null;
let currentGenPath = null;
let currentOptimizedPath = null;
let currentSummaryPath = null;
let generatedXyzText = null;
let optimizedXyzText = null;
let generatedViewer = null;
let optimizedViewer = null;
let stepJobPollTimer = null;
let currentStepJobId = null;

const statusEl = document.getElementById('status');
const jobMetaEl = document.getElementById('jobMeta');
const pathMetaEl = document.getElementById('pathMeta');

function $(id) { return document.getElementById(id); }

function getBool(id) {
  return $(id).value === 'true';
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store',
    ...opts,
  });
  const txt = await res.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch { data = { raw: txt }; }
  if (!res.ok) throw new Error(data?.detail || txt || `HTTP ${res.status}`);
  return data;
}

function claimText() {
  return $('claims').value;
}

function renderPathMeta() {
  if (!pathMetaEl) return;
  pathMetaEl.textContent = `gen_path=${currentGenPath || 'N/A'} | optimized_path=${currentOptimizedPath || 'N/A'} | summary_path=${currentSummaryPath || 'N/A'}`;
}

function setBusy(msg) {
  statusEl.textContent = msg;
}

function activateTab(tabName) {
  const tabBtn = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (!tabBtn) return;
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  tabBtn.classList.add('active');
  const panel = document.getElementById(`tab-${tabName}`);
  if (panel) panel.classList.add('active');
}

async function stepGenerate() {
  setBusy('Step 1 running: generating structure...');
  try {
    const payload = {
      claims: claimText(),
      api_key: $('api_key').value || null,
      model: $('model').value,
      mode: $('mode').value,
      no_generator: getBool('no_generator'),
      session_id: SESSION_ID,
    };
    const out = await api('/demo/step/generate', { method: 'POST', body: JSON.stringify(payload) });
    currentGenPath = out.gen_path || null;
    generatedXyzText = out.generated?.xyz_text || null;
    renderPathMeta();

    renderStructure('generatedViewer', generatedXyzText);
    $('generatedInfo').textContent = JSON.stringify({ gen_path: out.gen_path, generated: out.generated }, null, 2);
    $('rawJson').textContent = JSON.stringify(out, null, 2);

    statusEl.textContent = 'Step 1 completed. You can inspect generated structure and rerun Step 1 or continue to Step 2.';
    jobMetaEl.textContent = 'Step mode: generate';
    activateTab('generated');
    return out;
  } catch (err) {
    statusEl.textContent = `Step 1 failed: ${err.message}`;
    throw err;
  }
}

async function waitForStepJob(jobId) {
  return new Promise((resolve, reject) => {
    let done = false;
    const poll = async () => {
      try {
        const job = await api(`/demo/step/jobs/${jobId}`);
        $('optLog').textContent = job.log || '';
        $('optLog').scrollTop = $('optLog').scrollHeight;
        if (job.status === 'failed') {
          clearInterval(stepJobPollTimer);
          currentStepJobId = null;
          done = true;
          reject(new Error(job.error?.message || 'Step job failed'));
          return;
        }
        if (job.status === 'cancelled') {
          clearInterval(stepJobPollTimer);
          currentStepJobId = null;
          done = true;
          reject(new Error('Step job stopped by user'));
          return;
        }
        if (job.status === 'completed') {
          clearInterval(stepJobPollTimer);
          currentStepJobId = null;
          done = true;
          resolve(job.result || {});
        }
      } catch (err) {
        clearInterval(stepJobPollTimer);
        currentStepJobId = null;
        done = true;
        reject(err);
      }
    };
    stepJobPollTimer = setInterval(() => {
      if (!done) poll();
    }, 1000);
    poll();
  });
}

async function stepUma() {
  setBusy('Step 2 running: UMA optimization...');
  try {
    const payload = {
      gen_path: currentGenPath,
      preset: $('preset').value,
      device: $('device').value,
      loops: parseInt($('loops').value || '2', 10),
      orca_nprocs: parseInt($('orca_nprocs').value || '8', 10),
      orca_maxcore: parseInt($('orca_maxcore').value || '4000', 10),
      claim: claimText(),
      summary_path: currentSummaryPath,
      designer_client: $('model').value,
      api_key: $('api_key').value || null,
      session_id: SESSION_ID,
    };
    const start = await api('/demo/step/optimize/start', { method: 'POST', body: JSON.stringify(payload) });
    const jobId = start.job_id;
    currentStepJobId = jobId;
    jobMetaEl.textContent = `Step mode: UMA | job=${jobId}`;
    $('optLog').textContent = '[demo] waiting for UMA logs...\n';

    const out = await waitForStepJob(jobId);

    currentGenPath = out.gen_path || currentGenPath;
    currentSummaryPath = out.optimize?.summary_path || currentSummaryPath;
    currentOptimizedPath = out.optimize?.uma?.traj || out.optimized?.source || currentOptimizedPath;
    optimizedXyzText = out.optimized?.xyz_text || null;
    renderPathMeta();

    renderStructure('optimizedViewer', optimizedXyzText);
    $('optimizedInfo').textContent = JSON.stringify(out.optimize || {}, null, 2);

    renderMd(out.md_metrics?.metrics || {});
    $('mdInfo').textContent = JSON.stringify(out.md_metrics || {}, null, 2);

    latestBundle = out;
    $('rawJson').textContent = JSON.stringify(out, null, 2);
    statusEl.textContent = 'Step 2 completed. You can inspect UMA result and run Step 3 for DFT.';
    $('optLog').scrollTop = $('optLog').scrollHeight;
    activateTab('optimized');
    return out;
  } catch (err) {
    statusEl.textContent = `Step 2 failed: ${err.message}`;
    throw err;
  }
}

async function stepDft() {
  setBusy('Step 3 running: DFT computation...');
  try {
    const payload = {
      outdir: latestBundle?.outdir || './artifacts/mdopt',
      orca_nprocs: parseInt($('orca_nprocs').value || '8', 10),
      orca_maxcore: parseInt($('orca_maxcore').value || '4000', 10),
      claim: claimText(),
      gen_path: currentGenPath,
      designer_client: $('model').value,
      api_key: $('api_key').value || null,
      session_id: SESSION_ID,
    };
    const start = await api('/demo/step/dft/start', { method: 'POST', body: JSON.stringify(payload) });
    const jobId = start.job_id;
    currentStepJobId = jobId;
    jobMetaEl.textContent = `Step mode: DFT | job=${jobId}`;
    $('optLog').textContent = '[demo] waiting for DFT logs...\n';

    const out = await waitForStepJob(jobId);
    currentSummaryPath = out.summary_path || currentSummaryPath;
    renderPathMeta();

    renderDft(out.dft || {});
    if (out.verification) {
      renderVerification(out.verification);
    }

    latestBundle = { ...(latestBundle || {}), ...out };
    $('rawJson').textContent = JSON.stringify(out, null, 2);
    statusEl.textContent = 'Step 3 completed. DFT JSON returned.';
    activateTab('dft');
    return out;
  } catch (err) {
    statusEl.textContent = `Step 3 failed: ${err.message}`;
    throw err;
  }
}

async function stepVerify() {
  setBusy('Step 4 running: claim verification...');
  try {
    const payload = {
      claim: claimText(),
      summary_path: currentSummaryPath,
      gen_path: currentGenPath,
      designer_client: $('model').value,
      api_key: $('api_key').value || null,
      session_id: SESSION_ID,
    };
    const out = await api('/demo/step/verify', { method: 'POST', body: JSON.stringify(payload) });
    renderVerification(out);
    $('rawJson').textContent = JSON.stringify(out, null, 2);
    statusEl.textContent = 'Step 4 completed. Verification updated.';
    jobMetaEl.textContent = 'Step mode: verify';
    activateTab('verify');
    return out;
  } catch (err) {
    statusEl.textContent = `Step 4 failed: ${err.message}`;
    throw err;
  }
}

async function stopCurrentRun() {
  try {
    const out = await api('/demo/step/jobs/stop-all', { method: 'POST', body: JSON.stringify({ session_id: SESSION_ID }) });
    if (stepJobPollTimer) {
      clearInterval(stepJobPollTimer);
      stepJobPollTimer = null;
    }
    statusEl.textContent = `Stopped ${out.count || 0} running job(s).`;
    jobMetaEl.textContent = `Step jobs cancelled: ${(out.stopped_jobs || []).join(', ') || 'none'}`;
    currentStepJobId = null;
  } catch (err) {
    statusEl.textContent = `Stop failed: ${err.message}`;
  }
}

async function runAllSteps() {
  try {
    await stepGenerate();
    await stepUma();
    if (getBool('run_dft')) {
      await stepDft();
    }
    await stepVerify();
    statusEl.textContent = 'All steps completed.';
  } catch {
  }
}

async function refreshCurrent() {
  if (!currentSummaryPath && !currentGenPath) {
    statusEl.textContent = 'No current results to refresh. Please run Step 1 first.';
    return;
  }
  if (currentGenPath) {
    try {
      const gen = await api(`/demo/structure?path=${encodeURIComponent(currentGenPath)}`);
      generatedXyzText = gen.xyz_text || null;
      renderStructure('generatedViewer', generatedXyzText);
      $('generatedInfo').textContent = JSON.stringify(gen, null, 2);
    } catch {
    }
  }
  if (currentSummaryPath) {
    try {
      const dftOut = await api(`/demo/dft?summary_path=${encodeURIComponent(currentSummaryPath)}`);
      renderDft(dftOut.dft || {});
      $('dftRaw').textContent = JSON.stringify(dftOut, null, 2);
    } catch {
    }
  }
  try {
    const outdir = latestBundle?.outdir || './artifacts/mdopt';
    const md = await api(`/demo/md-metrics?outdir=${encodeURIComponent(outdir)}`);
    renderMd(md.metrics || {});
    $('mdInfo').textContent = JSON.stringify(md, null, 2);
  } catch {
  }
  statusEl.textContent = 'Refresh finished.';
}

function renderStructure(containerId, xyzText) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  if (!xyzText) {
    el.textContent = 'No structure available.';
    if (containerId === 'generatedViewer') generatedViewer = null;
    if (containerId === 'optimizedViewer') optimizedViewer = null;
    return;
  }
  const viewer = $3Dmol.createViewer(el, { backgroundColor: '#ffffff' });
  viewer.addModel(xyzText, 'xyz');
  viewer.setStyle({}, { stick: { radius: 0.20 }, sphere: { scale: 0.28 } });
  viewer.zoomTo();
  viewer.render();
  if (containerId === 'generatedViewer') generatedViewer = viewer;
  if (containerId === 'optimizedViewer') optimizedViewer = viewer;
}

function resetViewer(viewer) {
  if (!viewer) return;
  try {
    viewer.zoomTo();
    viewer.render();
  } catch {
  }
}

async function resetGeneratedView() {
  if (!currentGenPath) {
    statusEl.textContent = 'No generated structure path available. Please run Step 1 first.';
    return;
  }
  try {
    statusEl.textContent = 'Reset Generated View: reloading structure file...';
    const gen = await api(`/demo/structure?path=${encodeURIComponent(currentGenPath)}&_ts=${Date.now()}`);
    generatedXyzText = gen.xyz_text || null;
    renderStructure('generatedViewer', generatedXyzText);
    $('generatedInfo').textContent = JSON.stringify(gen, null, 2);
    statusEl.textContent = 'Generated view reloaded from file.';
  } catch (err) {
    statusEl.textContent = `Reset Generated View failed: ${err.message}`;
  }
}

async function resetOptimizedView() {
  if (!currentOptimizedPath) {
    statusEl.textContent = 'No optimized structure path available. Please run Step 2 first.';
    return;
  }
  try {
    statusEl.textContent = 'Reset Optimized View: reloading structure file...';
    const opt = await api(`/demo/structure?path=${encodeURIComponent(currentOptimizedPath)}&_ts=${Date.now()}`);
    optimizedXyzText = opt.xyz_text || null;
    renderStructure('optimizedViewer', optimizedXyzText);
    const currentOptJson = $('optimizedInfo').textContent ? JSON.parse($('optimizedInfo').textContent) : {};
    $('optimizedInfo').textContent = JSON.stringify({ ...currentOptJson, reloaded_structure: opt }, null, 2);
    statusEl.textContent = 'Optimized view reloaded from file.';
  } catch (err) {
    statusEl.textContent = `Reset Optimized View failed: ${err.message}`;
  }
}

function renderMd(metrics) {
  const loops = metrics.loop || [];
  const energies = metrics.energy_eV || [];
  const volumes = metrics.volume_A3 || [];
  const rmsd = metrics.rmsd_A || [];

  Plotly.newPlot('energyPlot', [{ x: loops, y: energies, mode: 'lines+markers', name: 'Energy (eV)' }],
    { title: 'Energy vs Loop', margin: { t: 40, l: 50, r: 20, b: 40 } });

  Plotly.newPlot('volumePlot', [{ x: loops, y: volumes, mode: 'lines+markers', name: 'Volume (Å^3)' }],
    { title: 'Volume vs Loop', margin: { t: 40, l: 50, r: 20, b: 40 } });

  const rmsdX = loops.slice(1);
  Plotly.newPlot('rmsdPlot', [{ x: rmsdX, y: rmsd, mode: 'lines+markers', name: 'RMSD (Å)' }],
    { title: 'RMSD Between Loops', margin: { t: 40, l: 50, r: 20, b: 40 } });
}

function resizeMdPlots() {
  ['energyPlot', 'volumePlot', 'rmsdPlot'].forEach((id) => {
    const el = $(id);
    if (!el || !el.data || !el.data.length) return;
    try {
      Plotly.Plots.resize(el);
    } catch {
    }
  });
}

function rerenderActiveTab() {
  const activeTab = document.querySelector('.tab-content.active');
  if (!activeTab) return;
  if (activeTab.id === 'tab-generated' && generatedXyzText) {
    renderStructure('generatedViewer', generatedXyzText);
  }
  if (activeTab.id === 'tab-optimized') {
    if (optimizedXyzText) {
      renderStructure('optimizedViewer', optimizedXyzText);
    }
    resizeMdPlots();
  }
}

function card(k, v) {
  return `<div class="card"><div class="k">${k}</div><div class="v">${v ?? 'N/A'}</div></div>`;
}

function renderDft(dft) {
  const cards = [];
  cards.push(card('Energy (eV)', fmt(dft.energy_eV)));
  cards.push(card('Gap (eV)', fmt(dft.gap_eV)));
  cards.push(card('HOMO (eV)', fmt(dft.homo_eV)));
  cards.push(card('LUMO (eV)', fmt(dft.lumo_eV)));
  cards.push(card('Dipole (D)', fmt(dft.dipole_D)));
  cards.push(card('Forces #', Array.isArray(dft.forces) ? dft.forces.length : 'N/A'));
  $('dftCards').innerHTML = cards.join('');
  $('dftRaw').textContent = JSON.stringify(dft, null, 2);
}

function renderVerification(v) {
  if (!v || !v.verdict) {
    $('verifySummary').innerHTML = '<p>No verification output.</p>';
    $('verifyChecks').innerHTML = '';
    return;
  }
  const cls = v.verdict === 'supported' ? 'ok' : (v.verdict === 'not-supported' ? 'bad' : 'warn');
  const scoreInt = Number.isFinite(Number(v.score)) ? Math.max(-2, Math.min(2, Math.round(Number(v.score)))) : 0;
  const scoreLabel = likertLabel(scoreInt);
  const reason = v.reason ? `<p class="verify-reason"><b>Reason:</b> ${escapeHtml(v.reason)}</p>` : '';
  const modelUsed = v.model_used ? `<p class="verify-model"><b>Model:</b> ${escapeHtml(v.model_used)}</p>` : '';

  const constraints = (v.extracted_constraints || []).map((c) => {
    const p = c?.property ? escapeHtml(String(c.property)) : 'unknown';
    const t = c?.target !== undefined ? escapeHtml(JSON.stringify(c.target)) : 'N/A';
    const s = c?.source ? escapeHtml(String(c.source)) : 'claim';
    return `<li><b>${p}</b>: target=${t} <span class="verify-meta">(${s})</span></li>`;
  });

  $('verifySummary').innerHTML = `
    <div class="verify-summary-card">
      <p><b>Verdict:</b> <span class="${cls}">${escapeHtml(v.verdict)}</span> | <b>Score:</b> ${scoreInt} <span class="verify-meta">(${escapeHtml(scoreLabel)})</span></p>
      ${reason}
      ${modelUsed}
      ${constraints.length ? `<div class="verify-constraints"><b>Claim constraints:</b><ul>${constraints.join('')}</ul></div>` : ''}
    </div>
  `;

  const rawRows = Array.isArray(v.parameter_comparisons) && v.parameter_comparisons.length
    ? v.parameter_comparisons
    : (v.checks || []);

  if (!rawRows.length) {
    $('verifyChecks').innerHTML = '<p>No parameter-level comparisons returned.</p>';
    return;
  }

  const normalizedRows = rawRows.map((c) => {
    const status = (c?.status || 'uncertain').toString();
    const ccls = status === 'pass' ? 'ok' : (status === 'fail' ? 'bad' : 'warn');

    const property = c?.property ?? 'unknown';
    const target = c?.claim_target ?? c?.target ?? 'N/A';
    const actual = c?.actual ?? 'N/A';
    const why = c?.comparison ?? c?.reason ?? c?.op ?? 'N/A';
    const importance = c?.importance ?? 'N/A';

    return {
      status,
      html: `
      <tr>
        <td>${escapeHtml(String(property))}</td>
        <td>${escapeHtml(typeof target === 'string' ? target : JSON.stringify(target))}</td>
        <td>${escapeHtml(typeof actual === 'number' ? fmt(actual) : (typeof actual === 'string' ? actual : JSON.stringify(actual)))}</td>
        <td><span class="${ccls}">${escapeHtml(status)}</span></td>
        <td>${escapeHtml(String(importance))}</td>
        <td>${escapeHtml(String(why))}</td>
      </tr>
      `,
    };
  });

  const passCount = normalizedRows.filter((r) => r.status === 'pass').length;
  const failCount = normalizedRows.filter((r) => r.status === 'fail').length;
  const uncertainCount = normalizedRows.filter((r) => r.status !== 'pass' && r.status !== 'fail').length;

  const renderRows = (filter) => {
    const rows = normalizedRows
      .filter((r) => filter === 'all' || r.status === filter)
      .map((r) => r.html);

    const empty = '<tr><td colspan="6" class="verify-empty">No rows under current filter.</td></tr>';
    $('verifyTableBody').innerHTML = rows.length ? rows.join('') : empty;
  };

  $('verifyChecks').innerHTML = `
    <div class="verify-toolbar">
      <div class="verify-badges">
        <span class="badge pass">pass ${passCount}</span>
        <span class="badge fail">fail ${failCount}</span>
        <span class="badge uncertain">uncertain ${uncertainCount}</span>
      </div>
      <div class="verify-filters">
        <button id="vf-all" class="vf-btn active" type="button">All</button>
        <button id="vf-fail" class="vf-btn" type="button">Fail</button>
        <button id="vf-uncertain" class="vf-btn" type="button">Uncertain</button>
      </div>
    </div>
    <div class="verify-table-wrap">
      <table class="verify-table">
        <thead>
          <tr>
            <th>Property</th>
            <th>Claim Target</th>
            <th>DFT / Actual</th>
            <th>Status</th>
            <th>Importance</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody id="verifyTableBody"></tbody>
      </table>
    </div>
  `;

  const setFilter = (name) => {
    ['vf-all', 'vf-fail', 'vf-uncertain'].forEach((id) => {
      const btn = $(id);
      if (btn) btn.classList.toggle('active', id === `vf-${name}`);
    });
    renderRows(name === 'uncertain' ? 'uncertain' : name);
  };

  $('vf-all')?.addEventListener('click', () => setFilter('all'));
  $('vf-fail')?.addEventListener('click', () => setFilter('fail'));
  $('vf-uncertain')?.addEventListener('click', () => setFilter('uncertain'));

  renderRows('all');
}

function likertLabel(score) {
  if (score === 2) return 'strongly correct / clearly feasible';
  if (score === 1) return 'mostly correct / likely feasible';
  if (score === 0) return 'uncertain or mixed evidence';
  if (score === -1) return 'somewhat not correct / likely infeasible';
  return 'strongly not correct / clearly infeasible';
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function fmt(v, digits = 6) {
  if (v === null || v === undefined) return 'N/A';
  if (typeof v === 'number') return Number.isFinite(v) ? v.toFixed(digits) : String(v);
  return String(v);
}

function setupTabs() {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(t => t.addEventListener('click', () => {
    tabs.forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    t.classList.add('active');
    const id = `tab-${t.dataset.tab}`;
    document.getElementById(id).classList.add('active');

    setTimeout(() => rerenderActiveTab(), 50);
  }));
}

window.addEventListener('resize', () => {
  rerenderActiveTab();
});

if (typeof ResizeObserver !== 'undefined') {
  const panel = document.querySelector('main > .panel:nth-child(2)');
  if (panel) {
    const ro = new ResizeObserver(() => rerenderActiveTab());
    ro.observe(panel);
  }
}

document.getElementById('stepGenerateBtn').addEventListener('click', stepGenerate);
document.getElementById('stepUmaBtn').addEventListener('click', stepUma);
document.getElementById('stepDftBtn').addEventListener('click', stepDft);
document.getElementById('stepVerifyBtn').addEventListener('click', stepVerify);
document.getElementById('runAllBtn').addEventListener('click', runAllSteps);
document.getElementById('refreshBtn').addEventListener('click', refreshCurrent);
document.getElementById('stopRunBtn').addEventListener('click', stopCurrentRun);
document.getElementById('resetGeneratedViewBtn').addEventListener('click', resetGeneratedView);
document.getElementById('resetOptimizedViewBtn').addEventListener('click', resetOptimizedView);

setupTabs();
renderPathMeta();
