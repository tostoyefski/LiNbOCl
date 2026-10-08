const apiBase = "/api";
let selectedJob = null;
let logPoll = null;
let jobsCache = [];
let viewer3d = null;

const statusClass = {
  queued: "queued",
  running: "running",
  succeeded: "succeeded",
  failed: "failed",
};

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

function splitList(str) {
  return str
    .split(/[,;\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function showBadge(text) {
  const node = document.createElement("span");
  node.className = "chip";
  node.textContent = text;
  return node;
}

function setHealth(status) {
  const holder = document.getElementById("health-badge");
  holder.innerHTML = "";
  if (status && status.mattergen_root) {
    holder.appendChild(showBadge("MatterGen 根: " + status.mattergen_root));
  }
  holder.appendChild(showBadge("API 正常"));
}

async function init() {
  try {
    const health = await fetchJSON(`${apiBase}/health`);
    setHealth(health);
  } catch (err) {
    const holder = document.getElementById("health-badge");
    holder.innerHTML = "";
    const badge = showBadge("API 不可用: " + err.message);
    badge.classList.add("muted");
    holder.appendChild(badge);
  }
  await applyPathDefaults();
  bindForms();
  document.getElementById("refresh-jobs").addEventListener("click", () => refreshJobs(true));
  document.getElementById("run-full").addEventListener("click", runFullPipeline);
  document.getElementById("load-voltage").addEventListener("click", loadVoltage);
  const loadBtn = document.getElementById("viewer-load");
  if (loadBtn) loadBtn.addEventListener("click", loadStructures);
  refreshJobs(true);
  logPoll = setInterval(refreshJobs, 4000);
  initViewer();
}

async function applyPathDefaults() {
  try {
    const defaults = await fetchJSON(`${apiBase}/defaults`);
    const fd = document.getElementById("form-dd");
    const fe = document.getElementById("form-eval");
    const fs = document.getElementById("form-screen");
    const ft = document.getElementById("form-top300");
    if (fd && defaults.base_results_dir) fd.base_results_dir.value = defaults.base_results_dir;
    if (fe && defaults.eval_root) fe.root.value = defaults.eval_root;
    if (fs) {
      if (defaults.screen_base) fs.base.value = defaults.screen_base;
      if (defaults.screen_out) fs.out.value = defaults.screen_out;
    }
    if (ft) {
      if (defaults.top300_stage2_csv) ft.stage2_csv.value = defaults.top300_stage2_csv;
      if (defaults.top300_output_dir) ft.output_dir.value = defaults.top300_output_dir;
      if (defaults.top300_export_dir) ft.export_dir.value = defaults.top300_export_dir;
    }
    const voltagePath = document.getElementById("voltage-path");
    if (voltagePath && defaults.voltage_path) voltagePath.value = defaults.voltage_path;
    const viewerDir = document.getElementById("viewer-dir");
    if (viewerDir && defaults.viewer_dir) viewerDir.value = defaults.viewer_dir;
  } catch (err) {
    console.warn("Failed to load path defaults", err);
  }
}

function bindForms() {
  document.getElementById("form-dd").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = e.submitter;
    const f = e.target;
    const chemSystems = splitList(f.chemical_systems.value);
    const elements = splitList(f.elements.value);
    const comboSizes = splitList(f.combo_sizes.value).map(Number).filter((x) => !Number.isNaN(x));
    const payload = {
      model_name: f.model_name.value,
      base_results_dir: f.base_results_dir.value,
      batch_size: Number(f.batch_size.value),
      num_batches: Math.max(1, Number(f.num_batches.value) || 1),
      e_ah: Number(f.e_ah.value),
      guidance: Number(f.guidance.value),
      chemical_systems: chemSystems.length ? chemSystems : null,
      chemical_systems_file: f.chemical_systems_file.value || null,
      elements,
      combo_sizes: comboSizes.length ? comboSizes : [4],
    };
    await submitJob("dd.sh", `${apiBase}/run/dd`, payload, btn);
  });

  document.getElementById("form-eval").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = e.submitter;
    const payload = { root: e.target.root.value };
    await submitJob("eval_all.sh", `${apiBase}/run/eval`, payload, btn);
  });

  document.getElementById("form-screen").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = e.submitter;
    const f = e.target;
    const payload = {
      base: f.base.value,
      out: f.out.value,
      r_cut: Number(f.r_cut.value),
      supercell: f.supercell.value.trim().split(/\s+/).map(Number).filter((x) => !Number.isNaN(x)),
      light_oxy: f.light_oxy.value.trim().split(/\s+/).map(Number).filter((x) => !Number.isNaN(x)),
      topk: Number(f.topk.value),
      refs_out: f.refs_out.value,
      require_charge_balance: f.require_charge_balance.checked,
      use_smact: f.use_smact.checked,
      filter_light_oxy: f.filter_light_oxy.checked,
      required_elements: splitList(f.required_elements.value),
      allowed_elements: splitList(f.allowed_elements.value).length ? splitList(f.allowed_elements.value) : null,
    };
    await submitJob("screen_all_extxyz.py", `${apiBase}/run/screen`, payload, btn);
  });

  document.getElementById("form-top300").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = e.submitter;
    const f = e.target;
    const payload = {
      stage2_csv: f.stage2_csv.value,
      output_dir: f.output_dir.value,
      topk: Number(f.topk.value),
      selection_mode: f.selection_mode.value,
      refs_out: f.refs_out.value,
      export_dir: f.export_dir.value,
      export_prefix: f.export_prefix.value,
      export_index_name: f.export_index_name.value,
      ehull_threshold: Number(f.ehull_threshold.value),
      voltage_step: f.voltage_step.value ? Number(f.voltage_step.value) : null,
      voltage_threshold: Number(f.voltage_threshold.value),
      target_voltage: f.target_voltage.value.trim() ? Number(f.target_voltage.value) : null,
      min_voltage_window: Number(f.min_voltage_window.value),
      dry_run: f.dry_run.checked,
    };
    await submitJob("run_top300_pipeline.py", `${apiBase}/run/top300`, payload, btn);
  });
}

async function submitJob(label, url, payload, btn) {
  try {
    if (btn) btn.disabled = true;
    const job = await fetchJSON(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    toast(`${label} 已启动 (#${job.id})`);
    selectedJob = job.id;
    refreshJobs(true);
  } catch (err) {
    toast(`启动失败: ${err.message}`, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function toast(msg, danger = false) {
  const node = document.createElement("div");
  node.textContent = msg;
  node.style.position = "fixed";
  node.style.bottom = "18px";
  node.style.right = "18px";
  node.style.padding = "12px 14px";
  node.style.background = danger ? "#ff6b6b" : "rgba(118, 224, 194, 0.25)";
  node.style.color = danger ? "#0b1024" : "#0b1024";
  node.style.borderRadius = "12px";
  node.style.boxShadow = "0 10px 25px rgba(0,0,0,0.3)";
  node.style.zIndex = "20";
  document.body.appendChild(node);
  setTimeout(() => node.remove(), 3200);
}

async function refreshJobs(forceLog = false) {
  try {
    const data = await fetchJSON(`${apiBase}/jobs`);
    jobsCache = data;
    renderJobs(data);
    if (selectedJob && (forceLog || document.hasFocus())) {
      await loadLog(selectedJob);
    }
  } catch (err) {
    console.error(err);
  }
}

function renderJobs(data) {
  const tbody = document.getElementById("jobs-body");
  tbody.innerHTML = "";
  if (!data || !data.length) {
    const row = document.createElement("tr");
    row.innerHTML = `<td colspan="7" class="muted">暂无任务</td>`;
    tbody.appendChild(row);
    return;
  }
  data.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  for (const job of data) {
    const tr = document.createElement("tr");
    tr.addEventListener("click", () => {
      selectedJob = job.id;
      loadLog(job.id);
    });
    const status = job.status || "queued";
    const pill = `<span class="status-pill ${statusClass[status] || ""}">${status}</span>`;
    const elapsed = job.elapsed_seconds != null
      ? `${job.elapsed_seconds.toFixed(1)}s`
      : job.started_at
        ? `${((Date.now() - new Date(job.started_at)) / 1000).toFixed(1)}s`
        : "—";
    const finish = job.finished_at ? formatTimeBeijing(job.finished_at) : "—";
    const cancelBtn = `<button data-id="${job.id}" class="ghost small cancel-btn" onclick="event.stopPropagation();cancelJob('${job.id}')">终止</button>`;
    tr.innerHTML = `<td>${job.kind}</td><td>${job.id}</td><td>${pill}</td><td>${formatTimeBeijing(job.created_at)}</td><td>${elapsed}</td><td>${finish}</td><td>${cancelBtn}</td>`;
    tbody.appendChild(tr);
  }
}

async function loadStructures() {
  const dir = document.getElementById("viewer-dir").value;
  try {
    const res = await fetchJSON(`${apiBase}/structures?directory=${encodeURIComponent(dir)}`);
    const list = document.getElementById("viewer-files");
    list.innerHTML = "";
    if (!res.files.length) {
      list.innerHTML = `<li class="muted">目录下未找到 CIF/EXTXYZ</li>`;
      return;
    }
    res.files.forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f;
      li.addEventListener("click", () => loadStructureContent(f));
      list.appendChild(li);
    });
  } catch (err) {
    toast(`加载失败: ${err.message}`, true);
  }
}

async function loadStructureContent(path) {
  try {
    const text = await fetch(`${apiBase}/structures/content?path=${encodeURIComponent(path)}`).then((r) => r.text());
    document.getElementById("viewer-text").textContent = text || "（空文件）";
    render3D(text, path);
  } catch (err) {
    document.getElementById("viewer-text").textContent = err.message;
  }
}

function initViewer() {
  const container = document.getElementById("viewer3d");
  if (!container) return;
  if (!window.$3Dmol) {
    container.textContent = "3Dmol.js 未加载（确认 frontend/libs/3Dmol.js 存在）";
    return;
  }
  container.textContent = "";
  viewer3d = new $3Dmol.GLViewer(container, { backgroundColor: "#050814" });
}

function render3D(text, path) {
  const container = document.getElementById("viewer3d");
  if (!container) return;
  if (!window.$3Dmol) {
    container.textContent = "3Dmol.js 未加载";
    return;
  }
  if (!viewer3d) initViewer();
  if (!viewer3d) return;
  viewer3d.clear();
  const fmt = path.toLowerCase().endsWith(".cif") ? "cif" : "xyz";
  viewer3d.addModel(text, fmt);
  viewer3d.setStyle({}, { stick: { radius: 0.2 }, sphere: { radius: 0.35 } });
  viewer3d.zoomTo();
  viewer3d.render();
}

function formatTimeBeijing(val) {
  if (!val) return "—";
  const d = new Date(val);
  const utc = d.getTime() + d.getTimezoneOffset() * 60000;
  const bj = new Date(utc + 8 * 3600000);
  return bj.toLocaleString();
}

async function loadLog(jobId) {
  const logTitle = document.getElementById("log-title");
  const logStatus = document.getElementById("log-status");
  logTitle.textContent = `作业 ${jobId} 的日志`;
  logStatus.textContent = "";
  try {
    const job = await fetchJSON(`${apiBase}/jobs/${jobId}`);
    logStatus.textContent = job.status;
    logStatus.className = "chip";
    const body = await fetch(`${apiBase}/jobs/${jobId}/log?lines=800`);
    if (!body.ok) throw new Error("日志读取失败");
    const text = await body.text();
    const viewer = document.getElementById("log-viewer");
    viewer.textContent = text || "（暂无输出）";
    viewer.scrollTop = viewer.scrollHeight;
  } catch (err) {
    document.getElementById("log-viewer").textContent = err.message;
  }
}

async function cancelJob(jobId) {
  try {
    await fetchJSON(`${apiBase}/jobs/${jobId}/cancel`, { method: "POST" });
    toast(`已请求终止 ${jobId}`);
    refreshJobs(true);
  } catch (err) {
    toast(`终止失败: ${err.message}`, true);
  }
}

async function runFullPipeline() {
  try {
    const payload = buildFullPayload();
    const job = await fetchJSON(`${apiBase}/run/full`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    toast(`全流程已启动 (#${job.id})`);
    selectedJob = job.id;
    refreshJobs(true);
  } catch (err) {
    toast(`启动失败: ${err.message}`, true);
  }
}

function buildFullPayload() {
  const fd = document.getElementById("form-dd");
  const fe = document.getElementById("form-eval");
  const fs = document.getElementById("form-screen");
  const ft = document.getElementById("form-top300");
  const batchInput = document.getElementById("full-batches");
  const numBatches = Math.max(1, Number(batchInput ? batchInput.value : 1) || 1);
  const chemSystems = splitList(fd.chemical_systems.value);
  const elements = splitList(fd.elements.value);
  const comboSizes = splitList(fd.combo_sizes.value).map(Number).filter((x) => !Number.isNaN(x));
  return {
    dd: {
      model_name: fd.model_name.value,
      base_results_dir: fd.base_results_dir.value,
      batch_size: Number(fd.batch_size.value),
      num_batches: Math.max(1, Number(fd.num_batches.value) || 1),
      e_ah: Number(fd.e_ah.value),
      guidance: Number(fd.guidance.value),
      chemical_systems: chemSystems.length ? chemSystems : null,
      chemical_systems_file: fd.chemical_systems_file.value || null,
      elements,
      combo_sizes: comboSizes.length ? comboSizes : [4],
    },
    eval: { root: fe.root.value },
    screen: {
      base: fs.base.value,
      out: fs.out.value,
      r_cut: Number(fs.r_cut.value),
      supercell: fs.supercell.value.trim().split(/\s+/).map(Number).filter((x) => !Number.isNaN(x)),
      light_oxy: fs.light_oxy.value.trim().split(/\s+/).map(Number).filter((x) => !Number.isNaN(x)),
      topk: Number(fs.topk.value),
      refs_out: fs.refs_out.value,
      require_charge_balance: fs.require_charge_balance.checked,
      use_smact: fs.use_smact.checked,
      filter_light_oxy: fs.filter_light_oxy.checked,
      required_elements: splitList(fs.required_elements.value),
      allowed_elements: splitList(fs.allowed_elements.value).length ? splitList(fs.allowed_elements.value) : null,
    },
    top300: {
      stage2_csv: ft.stage2_csv.value,
      output_dir: ft.output_dir.value,
      topk: Number(ft.topk.value),
      selection_mode: ft.selection_mode.value,
      refs_out: ft.refs_out.value,
      export_dir: ft.export_dir.value,
      export_prefix: ft.export_prefix.value,
      export_index_name: ft.export_index_name.value,
      ehull_threshold: Number(ft.ehull_threshold.value),
      voltage_step: ft.voltage_step.value ? Number(ft.voltage_step.value) : null,
      voltage_threshold: Number(ft.voltage_threshold.value),
      target_voltage: ft.target_voltage.value.trim() ? Number(ft.target_voltage.value) : null,
      min_voltage_window: Number(ft.min_voltage_window.value),
      dry_run: ft.dry_run.checked,
    },
    num_batches: numBatches,
  };
}

async function loadVoltage() {
  const p = document.getElementById("voltage-path").value;
  try {
    const res = await fetchJSON(`${apiBase}/voltage?path=${encodeURIComponent(p)}`);
    renderVoltage(res.rows);
  } catch (err) {
    toast(`加载失败: ${err.message}`, true);
  }
}

function renderVoltage(rows) {
  const head = document.getElementById("voltage-head");
  const body = document.getElementById("voltage-body");
  head.innerHTML = "";
  body.innerHTML = "";
  if (!rows || !rows.length) {
    body.innerHTML = `<tr><td class="muted">暂无数据</td></tr>`;
    return;
  }
  const columns = Object.keys(rows[0]);
  head.innerHTML = `<tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = columns.map((c) => `<td>${r[c]}</td>`).join("");
    body.appendChild(tr);
  });
}

window.addEventListener("DOMContentLoaded", init);
