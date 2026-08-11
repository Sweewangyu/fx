"use strict";

const state = {
  defaults: null,
  catalog: null,
  preview: null,
  previewFingerprint: null,
  exporting: false,
  stages: {
    stage1: { sources: new Set(), qualities: new Set(), difficulties: new Set(), abilities: new Set() },
    stage2: { sources: new Set(), qualities: new Set(), difficulties: new Set(), abilities: new Set() },
  },
};

const $ = (selector) => document.querySelector(selector);
const number = new Intl.NumberFormat("zh-CN");
const paths = {
  registry_path: $("#registry-path"),
  annotations_root: $("#annotations-root"),
  data_root: $("#data-root"),
  output_root: $("#output-root"),
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function formatCount(value) {
  return number.format(Number(value || 0));
}

function showToast(message, isError = false) {
  const toast = node("div", `toast${isError ? " is-error" : ""}`, message);
  $("#toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`服务返回了无法解析的响应（HTTP ${response.status}）`);
  }
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

function pathPayload() {
  return {
    registry_path: paths.registry_path.value.trim(),
    annotations_root: paths.annotations_root.value.trim(),
    data_root: paths.data_root.value.trim() || null,
    output_root: paths.output_root.value.trim(),
  };
}

function stagePayload(stageName) {
  const stage = state.stages[stageName];
  return {
    sources: [...stage.sources].sort(),
    qualities: [...stage.qualities],
    difficulties: [...stage.difficulties],
    abilities: [...stage.abilities].sort(),
  };
}

function selectionPayload() {
  return {
    ...pathPayload(),
    stage1: stagePayload("stage1"),
    stage2: stagePayload("stage2"),
  };
}

function selectionFingerprint() {
  return JSON.stringify(selectionPayload());
}

function markSelectionDirty() {
  if (state.previewFingerprint) {
    const dirty = state.previewFingerprint !== selectionFingerprint();
    if (dirty) {
      $("#preview-message").hidden = false;
      $("#preview-message").textContent = "筛选条件已变化，请重新计算样本量。";
      $("#preview-message").classList.remove("is-error");
      $("#preview-detail").hidden = true;
      $("#export-section").classList.add("is-locked");
    } else if (state.preview) {
      renderPreview(state.preview);
    }
  }
  updateStageCounts();
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.classList.toggle("loading", busy);
  if (label) button.querySelector("span") ? button.querySelector("span").replaceChildren(label) : button.replaceChildren(label);
}

function setHealth(kind, text) {
  const pill = $("#health-pill");
  pill.classList.toggle("is-online", kind === "online");
  pill.classList.toggle("is-error", kind === "error");
  $("#health-text").textContent = text;
}

function setUnlocked(unlocked) {
  for (const selector of ["#datasets-section", "#rules-section", "#preview-section"]) {
    $(selector).classList.toggle("is-locked", !unlocked);
  }
  if (!unlocked) $("#export-section").classList.add("is-locked");
}

function initializeRules() {
  for (const stageName of ["stage1", "stage2"]) {
    const preset = state.defaults.presets[stageName];
    state.stages[stageName].qualities = new Set(preset.qualities);
    state.stages[stageName].difficulties = new Set(preset.difficulties);
    state.stages[stageName].abilities = new Set();
  }
}

function renderRuleGroup(stageName, type, title, description, values, labels, selected) {
  const group = node("fieldset", "rule-group");
  const legend = node("legend", "sr-only", title);
  group.append(legend);
  const heading = node("div", "rule-heading");
  heading.append(node("strong", "", title), node("small", "", description));
  group.append(heading);
  const chips = node("div", "chips");
  values.forEach((value) => {
    const label = node("label", "chip");
    const input = node("input");
    input.type = "checkbox";
    input.checked = selected.has(value);
    input.dataset.stage = stageName;
    input.dataset.type = type;
    input.value = value;
    const translated = labels && labels[value] ? labels[value] : value;
    label.append(input, node("span", "", translated));
    chips.append(label);
  });
  group.append(chips);
  return group;
}

function renderAbilityGroup(stageName) {
  const selected = state.stages[stageName].abilities;
  const group = node("fieldset", "rule-group");
  group.append(node("legend", "sr-only", "能力维度"));
  const heading = node("div", "rule-heading");
  heading.append(node("strong", "", "能力维度"), node("small", "", "不限定时包含全部能力"));
  group.append(heading);
  const chips = node("div", "chips");
  const allLabel = node("label", "chip chip-all");
  const allInput = node("input");
  allInput.type = "checkbox";
  allInput.checked = selected.size === 0;
  allInput.dataset.stage = stageName;
  allInput.dataset.type = "abilities-all";
  allInput.value = "__all__";
  allLabel.append(allInput, node("span", "", "全部能力"));
  chips.append(allLabel);
  for (const ability of state.catalog.abilities) {
    const label = node("label", "chip");
    const input = node("input");
    input.type = "checkbox";
    input.checked = selected.has(ability);
    input.dataset.stage = stageName;
    input.dataset.type = "abilities";
    input.value = ability;
    label.append(input, node("span", "", ability));
    chips.append(label);
  }
  group.append(chips);
  return group;
}

function renderRules() {
  if (!state.catalog) return;
  for (const stageName of ["stage1", "stage2"]) {
    const stage = state.stages[stageName];
    const root = $(`#${stageName}-rules`);
    root.replaceChildren(
      renderRuleGroup(
        stageName, "qualities", "质量等级", "可多选，weak 以上为默认",
        state.defaults.quality_levels, state.defaults.quality_labels_zh, stage.qualities,
      ),
      renderRuleGroup(
        stageName, "difficulties", "难度等级", stageName === "stage1" ? "默认中等及以下" : "默认中等及以上",
        state.defaults.difficulty_levels, state.defaults.difficulty_labels_zh, stage.difficulties,
      ),
      renderAbilityGroup(stageName),
    );
  }
}

function sourceSearchText(source) {
  return `${source.name} ${source.family} ${source.training_role} ${source.split}`.toLocaleLowerCase();
}

function createSourceToggle(source, stageName) {
  const label = node("label", "source-toggle");
  label.title = source.available ? `${stageName === "stage1" ? "Stage 1" : "Stage 2"} 使用 ${source.name}` : "标注不可用";
  const input = node("input");
  input.type = "checkbox";
  input.checked = state.stages[stageName].sources.has(source.name);
  input.disabled = !source.available;
  input.dataset.source = source.name;
  input.dataset.stage = stageName;
  input.setAttribute("aria-label", label.title);
  label.append(input, node("span"));
  return label;
}

function renderSources() {
  const root = $("#source-list");
  root.replaceChildren();
  for (const source of state.catalog.sources) {
    const row = node("article", `source-row${source.available ? "" : " is-unavailable"}`);
    row.dataset.search = sourceSearchText(source);
    const identity = node("div", "source-name");
    identity.append(node("strong", "", source.name));
    const tags = node("div", "source-tags");
    tags.append(node("span", "", source.family), node("span", "", source.training_role), node("span", "", source.split));
    identity.append(tags);
    const meta = node("div", "source-meta");
    meta.append(node("strong", "", formatCount(source.rows)));
    const availability = node(
      "span",
      source.available ? "available" : "missing",
      source.available ? `${source.annotation_mode} 标注可用` : "标注不可用",
    );
    if (!source.available && source.errors && source.errors.length) availability.title = source.errors.join("\n");
    meta.append(availability);
    row.append(identity, meta, createSourceToggle(source, "stage1"), createSourceToggle(source, "stage2"));
    root.append(row);
  }
  applySourceSearch();
}

function applySourceSearch() {
  const query = $("#dataset-search").value.trim().toLocaleLowerCase();
  let visible = 0;
  document.querySelectorAll(".source-row").forEach((row) => {
    row.hidden = query && !row.dataset.search.includes(query);
    if (!row.hidden) visible += 1;
  });
  $("#source-list").classList.toggle("has-no-match", Boolean(state.catalog && visible === 0));
}

function updateStageCounts() {
  for (const stageName of ["stage1", "stage2"]) {
    const size = state.stages[stageName].sources.size;
    $(`#${stageName}-source-count`).textContent = `${size} 个数据源`;
  }
}

function chooseSources(mode) {
  if (!state.catalog) return;
  const available = state.catalog.sources.filter((item) => item.available).map((item) => item.name);
  let names = [];
  if (mode === "all") names = available;
  if (mode === "defaults") names = state.defaults.target_sources.filter((name) => available.includes(name));
  for (const stageName of ["stage1", "stage2"]) state.stages[stageName].sources = new Set(names);
  renderSources();
  markSelectionDirty();
  showToast(mode === "clear" ? "已清空两个阶段的数据源" : `两个阶段已选择 ${names.length} 个数据源`);
}

function catalogStats() {
  const root = $("#catalog-stats");
  root.replaceChildren();
  root.append(
    node("span", "", `${state.catalog.available_sources} / ${state.catalog.total_sources} 源可用`),
    node("span", "", `${formatCount(state.catalog.total_rows)} 条已标注`),
    node("span", "", `${state.catalog.abilities.length} 个能力维度`),
  );
  const missing = state.catalog.total_sources - state.catalog.available_sources;
  if (missing) root.append(node("span", "is-warn", `${missing} 个源不可用`));
}

async function scanCatalog() {
  const button = $("#scan-button");
  const status = $("#scan-status");
  if (!paths.registry_path.value.trim() || !paths.annotations_root.value.trim()) {
    showToast("请先填写数据源注册表和标注目录", true);
    return;
  }
  setBusy(button, true, "正在扫描");
  status.textContent = "正在逐行读取标注，数据较大时可能需要几分钟…";
  status.classList.remove("is-error");
  try {
    state.catalog = await api("/api/catalog", { method: "POST", body: JSON.stringify(pathPayload()) });
    initializeRules();
    const available = new Set(state.catalog.sources.filter((item) => item.available).map((item) => item.name));
    const defaults = state.defaults.target_sources.filter((name) => available.has(name));
    state.stages.stage1.sources = new Set(defaults);
    state.stages.stage2.sources = new Set(defaults);
    state.preview = null;
    state.previewFingerprint = null;
    renderSources();
    renderRules();
    catalogStats();
    updateStageCounts();
    setUnlocked(true);
    resetPreview();
    status.textContent = `扫描完成：${state.catalog.available_sources} 个可用数据源，默认选中 ${defaults.length} 个示例源。`;
    showToast("数据目录扫描完成");
    $("#datasets-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setUnlocked(false);
    status.textContent = error.message;
    status.classList.add("is-error");
    showToast(error.message, true);
  } finally {
    setBusy(button, false, "扫描数据与标注");
  }
}

function validateSelection() {
  for (const stageName of ["stage1", "stage2"]) {
    const label = stageName === "stage1" ? "Stage 1" : "Stage 2";
    const stage = state.stages[stageName];
    if (!stage.sources.size) throw new Error(`${label} 至少需要选择一个数据源`);
    if (!stage.qualities.size) throw new Error(`${label} 至少需要选择一个质量等级`);
    if (!stage.difficulties.size) throw new Error(`${label} 至少需要选择一个难度等级`);
  }
}

function resetPreview() {
  state.preview = null;
  state.previewFingerprint = null;
  for (const id of ["#metric-stage1", "#metric-stage2", "#metric-overlap"]) $(id).textContent = "—";
  $("#metric-stage1-detail").textContent = "等待预览";
  $("#metric-stage2-detail").textContent = "等待预览";
  $("#metric-overlap-detail").textContent = "中等难度可同时出现";
  $("#preview-detail").hidden = true;
  $("#preview-message").hidden = false;
  $("#preview-message").textContent = "完成规则选择后，点击“计算样本量”。";
  $("#preview-message").classList.remove("is-error");
  $("#export-section").classList.add("is-locked");
}

function displayLabel(type, key) {
  if (type === "quality") return state.defaults.quality_labels_zh[key] || key;
  if (type === "difficulty") return state.defaults.difficulty_labels_zh[key] || key;
  return key;
}

function renderDistribution(stageName, preview) {
  const root = $(`#${stageName}-distribution`);
  root.classList.toggle("stage-two-bars", stageName === "stage2");
  root.replaceChildren();
  const total = preview.counts[stageName];
  const groups = [
    ["quality", "质量"],
    ["difficulty", "难度"],
  ];
  for (const [type, prefix] of groups) {
    const distribution = preview.distributions[stageName][type];
    for (const [key, value] of Object.entries(distribution).sort((a, b) => b[1] - a[1])) {
      const item = node("div", "bar-item");
      item.append(node("span", "", `${prefix} · ${displayLabel(type, key)}`));
      const track = node("progress", "bar-track");
      track.max = total || 1;
      track.value = value;
      track.textContent = total ? `${((value / total) * 100).toFixed(1)}%` : "0%";
      item.append(track, node("span", "bar-value", formatCount(value)));
      root.append(item);
    }
  }
  $(`#${stageName}-dist-total`).textContent = `${formatCount(total)} 条`;
}

function renderPreview(preview) {
  state.preview = preview;
  const { counts } = preview;
  $("#metric-stage1").textContent = formatCount(counts.stage1);
  $("#metric-stage2").textContent = formatCount(counts.stage2);
  $("#metric-overlap").textContent = formatCount(counts.overlap);
  $("#metric-stage1-detail").textContent = `${state.stages.stage1.sources.size} 个数据源`;
  $("#metric-stage2-detail").textContent = `${state.stages.stage2.sources.size} 个数据源`;
  const denominator = Math.min(counts.stage1, counts.stage2);
  $("#metric-overlap-detail").textContent = denominator ? `占较小阶段 ${((counts.overlap / denominator) * 100).toFixed(1)}%` : "无重叠";
  renderDistribution("stage1", preview);
  renderDistribution("stage2", preview);
  const tbody = $("#source-results");
  tbody.replaceChildren();
  for (const row of preview.by_source) {
    const tr = node("tr");
    for (const value of [row.source, row.source_rows, row.stage1, row.stage2, row.overlap]) tr.append(node("td", "", typeof value === "number" ? formatCount(value) : value));
    tbody.append(tr);
  }
  $("#preview-detail").hidden = false;
  $("#preview-message").hidden = true;
  $("#export-section").classList.remove("is-locked");
}

async function calculatePreview({ quiet = false } = {}) {
  const button = $("#preview-button");
  const message = $("#preview-message");
  validateSelection();
  setBusy(button, true);
  message.hidden = false;
  message.textContent = "正在计算精确筛选结果…";
  message.classList.remove("is-error");
  try {
    const preview = await api("/api/preview", { method: "POST", body: JSON.stringify(selectionPayload()) });
    state.previewFingerprint = selectionFingerprint();
    renderPreview(preview);
    if (!quiet) showToast("预览已更新");
    return preview;
  } catch (error) {
    message.hidden = false;
    message.textContent = error.message;
    message.classList.add("is-error");
    $("#export-section").classList.add("is-locked");
    if (!quiet) showToast(error.message, true);
    throw error;
  } finally {
    setBusy(button, false);
  }
}

function updateProgress(job) {
  const total = Number(job.total_rows || 0);
  const processed = Number(job.processed_rows || 0);
  const completed = job.status === "completed";
  const failed = job.status === "failed";
  const percent = completed ? 100 : total ? Math.min(99, Math.floor((processed / total) * 100)) : 0;
  const phaseLabels = {
    queued: "任务排队中",
    preparing: "正在准备导出",
    exporting: "正在流式导出",
    completed: "导出完成",
    failed: "导出失败",
  };
  $("#progress-phase").textContent = phaseLabels[job.phase] || phaseLabels[job.status] || "正在处理";
  $("#progress-percent").textContent = `${percent}%`;
  $("#progress-bar").value = percent;
  $("#progress-bar").textContent = `${percent}%`;
  $("#export-progress").classList.toggle("is-failed", failed);
  if (failed) {
    $("#progress-detail").textContent = job.error || "后台任务失败";
  } else if (completed) {
    $("#progress-detail").textContent = `${formatCount(job.result.counts.stage1)} 条 Stage 1 · ${formatCount(job.result.counts.stage2)} 条 Stage 2`;
  } else if (job.phase === "exporting") {
    $("#progress-detail").textContent = `${job.source || "数据源"} · ${formatCount(processed)} / ${formatCount(total)} 行`;
  } else {
    $("#progress-detail").textContent = "正在校验选择和创建输出目录…";
  }
}

function addResultPath(root, label, value) {
  const row = node("div", "result-path-row");
  row.append(node("dt", "", label), node("dd", "", value));
  const copy = node("button", "copy-button", "复制");
  copy.type = "button";
  copy.dataset.copy = value;
  copy.setAttribute("aria-label", `复制${label}路径`);
  row.append(copy);
  root.append(row);
}

function renderExportResult(result) {
  $("#result-summary").textContent = `Stage 1 ${formatCount(result.counts.stage1)} 条，Stage 2 ${formatCount(result.counts.stage2)} 条，重叠 ${formatCount(result.counts.overlap)} 条。`;
  const pathsRoot = $("#result-paths");
  pathsRoot.replaceChildren();
  addResultPath(pathsRoot, "输出目录", result.output_dir);
  addResultPath(pathsRoot, "训练环境", result.training_env);
  addResultPath(pathsRoot, "数据注册表", result.dataset_info);
  addResultPath(pathsRoot, "Manifest", result.manifest);
  $("#export-result").hidden = false;
}

function wait(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function pollExport(jobId) {
  while (true) {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    updateProgress(job);
    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error || "导出失败");
    await wait(700);
  }
}

async function startExport() {
  const button = $("#export-button");
  const runName = $("#run-name").value.trim();
  if (!/^[A-Za-z0-9_.-]+$/.test(runName)) {
    showToast("运行名称只能包含字母、数字、点、下划线和短横线", true);
    $("#run-name").focus();
    return;
  }
  if (!paths.output_root.value.trim()) {
    showToast("请填写导出根目录", true);
    paths.output_root.focus();
    return;
  }
  state.exporting = true;
  setBusy(button, true, "正在导出");
  $("#export-result").hidden = true;
  try {
    if (state.previewFingerprint !== selectionFingerprint()) await calculatePreview({ quiet: true });
    const request = { ...selectionPayload(), run_name: runName };
    const started = await api("/api/export", { method: "POST", body: JSON.stringify(request) });
    updateProgress({ status: "queued", phase: "queued" });
    const result = await pollExport(started.job_id);
    renderExportResult(result);
    showToast("训练数据导出完成");
  } catch (error) {
    updateProgress({ status: "failed", phase: "failed", error: error.message });
    showToast(error.message, true);
  } finally {
    state.exporting = false;
    setBusy(button, false, "开始导出");
  }
}

function handleRuleChange(event) {
  const input = event.target.closest("input[data-stage][data-type]");
  if (!input) return;
  const stage = state.stages[input.dataset.stage];
  const type = input.dataset.type;
  if (type === "abilities-all") {
    stage.abilities.clear();
    renderRules();
    markSelectionDirty();
    return;
  }
  const collection = stage[type];
  if (input.checked) collection.add(input.value);
  else collection.delete(input.value);
  if (type === "abilities") renderRules();
  markSelectionDirty();
}

function handleSourceChange(event) {
  const input = event.target.closest("input[data-source][data-stage]");
  if (!input) return;
  const collection = state.stages[input.dataset.stage].sources;
  if (input.checked) collection.add(input.dataset.source);
  else collection.delete(input.dataset.source);
  markSelectionDirty();
}

function defaultRunName() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `six-source-split-${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
}

async function copyPath(value, button) {
  try {
    await navigator.clipboard.writeText(value);
    button.textContent = "已复制";
    window.setTimeout(() => { button.textContent = "复制"; }, 1400);
  } catch {
    showToast("浏览器无法访问剪贴板，请手动选择路径", true);
  }
}

function bindEvents() {
  $("#scan-button").addEventListener("click", scanCatalog);
  $("#preview-button").addEventListener("click", () => calculatePreview().catch(() => {}));
  $("#export-button").addEventListener("click", startExport);
  $("#dataset-search").addEventListener("input", applySourceSearch);
  $("#select-defaults").addEventListener("click", () => chooseSources("defaults"));
  $("#select-all").addEventListener("click", () => chooseSources("all"));
  $("#clear-sources").addEventListener("click", () => chooseSources("clear"));
  $("#source-list").addEventListener("change", handleSourceChange);
  $("#rules-section").addEventListener("change", handleRuleChange);
  for (const input of Object.values(paths)) input.addEventListener("change", markSelectionDirty);
  $("#result-paths").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-copy]");
    if (button) copyPath(button.dataset.copy, button);
  });
}

async function boot() {
  bindEvents();
  $("#run-name").value = defaultRunName();
  try {
    await api("/api/health");
    state.defaults = await api("/api/defaults");
    for (const [key, input] of Object.entries(paths)) {
      const value = state.defaults[key];
      if (typeof value === "string") input.value = value;
    }
    initializeRules();
    $("#scan-button").disabled = false;
    setHealth("online", "本地服务已连接");
  } catch (error) {
    setHealth("error", "本地服务不可用");
    showToast(error.message, true);
  }
}

boot();
