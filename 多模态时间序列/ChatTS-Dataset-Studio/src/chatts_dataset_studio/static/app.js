"use strict";

const state = {
  defaults: null,
  catalog: null,
  preview: null,
  previewFingerprint: null,
  versions: [],
  nextVersion: null,
  activeVersion: null,
  selectedTrainingVersion: null,
  pipelineEnabled: false,
  standaloneEvaluationEnabled: false,
  modelOutputBaseTemplate: "",
  evaluationOutputBaseTemplate: "",
  jobs: [],
  activeJobId: null,
  busy: new Set(),
  stages: {
    stage1: { sources: new Set(), sourceRules: new Map() },
    stage2: { sources: new Set(), sourceRules: new Map() },
  },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const countFormatter = new Intl.NumberFormat("zh-CN");
const pathFields = {
  registry_path: $("#registry-path"),
  annotations_root: $("#annotations-root"),
  data_root: $("#data-root"),
  output_root: $("#output-root"),
};

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function element(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function formatCount(value) {
  return countFormatter.format(Number(value || 0));
}

function showToast(message, kind = "info") {
  const toast = element("div", `toast ${kind}`, message);
  $("#toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json", ...(options.headers || {}) } : options.headers,
  });
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    try { payload = await response.json(); } catch { payload = null; }
  }
  if (!response.ok || (payload && payload.error)) {
    const fallback = response.status === 404 ? "当前服务未启用此功能" : `请求失败（HTTP ${response.status}）`;
    throw new ApiError(payload?.error || fallback, response.status);
  }
  return payload ?? {};
}

async function optionalGet(url, fallback) {
  try { return await api(url); } catch (error) {
    if (error.status === 404) return fallback;
    throw error;
  }
}

function pathPayload() {
  return Object.fromEntries(Object.entries(pathFields).map(([key, input]) => [key, input.value.trim() || null]));
}

function stagePayload(stageName) {
  const stage = state.stages[stageName];
  const preset = stagePreset(stageName);
  const sourceRules = {};
  for (const sourceName of [...stage.sources].sort()) {
    const rule = ensureSourceRule(stageName, sourceName);
    sourceRules[sourceName] = {
      qualities: [...rule.qualities].sort(),
      difficulties: [...rule.difficulties].sort(),
      abilities: [...rule.abilities].sort(),
      sample_percent: rule.samplePercent,
    };
  }
  return {
    sources: [...stage.sources].sort(),
    qualities: [...preset.qualities].sort(),
    difficulties: [...preset.difficulties].sort(),
    abilities: [],
    source_rules: sourceRules,
  };
}

function recipePayload() {
  return { stage1: stagePayload("stage1"), stage2: stagePayload("stage2") };
}

function selectionPayload() {
  return { ...pathPayload(), ...recipePayload() };
}

function selectionFingerprint() {
  return JSON.stringify(selectionPayload());
}

function normalizeCollection(payload, key) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.[key])) return payload[key];
  if (payload?.[key] && typeof payload[key] === "object") {
    const idKey = key === "versions" ? "version" : "job_id";
    return Object.entries(payload[key]).filter(([, value]) => value && typeof value === "object").map(([id, value]) => ({ [idKey]: id, ...value }));
  }
  if (payload && typeof payload === "object" && key === "versions") {
    return Object.entries(payload).filter(([, value]) => value && typeof value === "object").map(([version, value]) => ({ version, ...value }));
  }
  return [];
}

function versionName(version) {
  const value = version?.version || version?.name || version?.data_version || "";
  return value && typeof value === "object" ? versionName(value) : String(value);
}

function jobId(job) {
  return String(job?.job_id || job?.id || "");
}

function selectedTrainingVersionName() {
  return String(state.selectedTrainingVersion || "");
}

function selectedTrainingVersion() {
  const selected = selectedTrainingVersionName();
  return state.versions.find((version) => versionName(version) === selected) || null;
}

function setHealth(kind, text) {
  $("#health-pill").className = `connection-pill ${kind}`;
  $("#health-text").textContent = text;
}

function setBusy(button, busy, label) {
  if (!button) return;
  const key = button.id || button.dataset.runMode || button.textContent.trim();
  if (!button.dataset.defaultLabel) button.dataset.defaultLabel = button.textContent.trim();
  if (busy) state.busy.add(key); else state.busy.delete(key);
  button.classList.toggle("is-busy", busy);
  button.setAttribute("aria-busy", String(busy));
  button.textContent = busy ? label : button.dataset.defaultLabel;
  renderActionState();
}

async function withBusy(button, label, task) {
  setBusy(button, true, label);
  try { return await task(); } finally { setBusy(button, false, label); }
}

function renderActionState() {
  const scanned = Boolean(state.catalog);
  const previewReady = Boolean(state.preview && state.previewFingerprint === selectionFingerprint());
  const hasVersion = Boolean(selectedTrainingVersionName());
  const slurmReady = $("#execution-backend").value !== "slurm" || Boolean($("#slurm-sbatch-path").value);
  const canRun = hasVersion && state.pipelineEnabled && slurmReady && $("#base-model-path").value.trim().startsWith("/");
  $("#scan-button").disabled = !state.defaults || state.busy.has("scan-button");
  $("#rebuild-sources").disabled = !state.defaults || state.busy.has("rebuild-sources");
  $("#preview-button").disabled = !scanned || state.busy.has("preview-button");
  $("#publish-button").disabled = !previewReady || state.busy.has("publish-button");
  $$(".run-button").forEach((button) => { button.disabled = !canRun || state.busy.has(button.dataset.runMode); });
  $("#overview-preflight").disabled = !canRun || state.busy.has("overview-preflight");
  $("#overview-run").disabled = !canRun || state.busy.has("overview-run");
  renderStandaloneEvaluationState();
}

function activateTab(name, { focus = false, updateHash = true } = {}) {
  const tab = $(`[data-tab="${name}"]`);
  const panel = $(`[data-panel="${name}"]`);
  if (!tab || !panel) return;
  $$("[role=tab]").forEach((item) => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  $$("[role=tabpanel]").forEach((item) => { item.hidden = item !== panel; });
  if (focus) tab.focus();
  if (updateHash) history.replaceState(null, "", `#${name}`);
  if (name === "jobs") refreshJobs({ quiet: true });
}

function handleTabKeys(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = $$("[role=tab]");
  const index = tabs.indexOf(event.currentTarget);
  let next = index;
  if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
  if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = tabs.length - 1;
  event.preventDefault();
  activateTab(tabs[next].dataset.tab, { focus: true });
}

function openSettings(open) {
  const panel = $("#settings-panel");
  const backdrop = $("#settings-backdrop");
  panel.hidden = !open;
  backdrop.hidden = !open;
  $("#open-settings").setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("settings-open", open);
  if (open) $("#registry-path").focus(); else $("#open-settings").focus();
}

function initializeRules() {
  for (const stageName of ["stage1", "stage2"]) {
    state.stages[stageName].sources = new Set();
    state.stages[stageName].sourceRules = new Map();
  }
}

function stagePreset(stageName) {
  const fallback = {
    stage1: { qualities: ["weak", "acceptable", "good", "excellent"], difficulties: ["very_easy", "easy", "moderate"] },
    stage2: { qualities: ["weak", "acceptable", "good", "excellent"], difficulties: ["moderate", "hard", "very_hard"] },
  };
  const value = state.defaults?.presets?.[stageName] || fallback[stageName];
  return {
    qualities: new Set(value.qualities || []),
    difficulties: new Set(value.difficulties || []),
  };
}

function catalogSource(sourceName) {
  return (state.catalog?.sources || []).find((source) => source.name === sourceName) || null;
}

function orderedDimension(source, type) {
  const values = Object.keys(source?.[type] || {});
  if (type === "quality") {
    return (state.defaults?.quality_levels || []).filter((value) => values.includes(value));
  }
  if (type === "difficulty") {
    return (state.defaults?.difficulty_levels || []).filter((value) => values.includes(value));
  }
  return values.sort((left, right) => Number(source.ability[right] || 0) - Number(source.ability[left] || 0) || left.localeCompare(right));
}

function defaultSourceRule(stageName, sourceName) {
  const source = catalogSource(sourceName);
  const preset = stagePreset(stageName);
  const availableQualities = orderedDimension(source, "quality");
  const availableDifficulties = orderedDimension(source, "difficulty");
  const presetQualities = availableQualities.filter((value) => preset.qualities.has(value));
  const presetDifficulties = availableDifficulties.filter((value) => preset.difficulties.has(value));
  return {
    qualities: new Set(presetQualities.length ? presetQualities : availableQualities),
    difficulties: new Set(presetDifficulties.length ? presetDifficulties : availableDifficulties),
    abilities: new Set(),
    samplePercent: 100,
  };
}

function ensureSourceRule(stageName, sourceName) {
  const stage = state.stages[stageName];
  if (!stage.sourceRules.has(sourceName)) {
    stage.sourceRules.set(sourceName, defaultSourceRule(stageName, sourceName));
  }
  return stage.sourceRules.get(sourceName);
}

function markRecipeDirty() {
  updateStageCounts();
  if (state.previewFingerprint !== selectionFingerprint()) {
    $("#recipe-state").textContent = state.preview ? "已修改" : "未预览";
    $("#recipe-state").className = "dirty-chip dirty";
    $("#preview-message").textContent = state.preview ? "配方已变化，请重新预览。" : "选择数据源后计算。";
    $("#preview-detail").hidden = true;
  }
  renderActionState();
}

function ruleGroup(stageName, sourceName, type, title, values, labels, selected, counts) {
  const fieldset = element("fieldset", "rule-group");
  fieldset.append(element("legend", "", title));
  const chips = element("div", "chips");
  for (const value of values || []) {
    const label = element("label", "chip");
    const input = element("input");
    input.type = "checkbox";
    input.checked = selected.has(value);
    input.dataset.stage = stageName;
    input.dataset.sourceRule = sourceName;
    input.dataset.type = type;
    input.value = value;
    const text = `${labels?.[value] || value} · ${formatCount(counts?.[value])}`;
    label.append(input, element("span", "", text));
    chips.append(label);
  }
  fieldset.append(chips);
  return fieldset;
}

function abilityGroup(stageName, sourceName, source, selected) {
  const fieldset = element("fieldset", "rule-group");
  fieldset.append(element("legend", "", "能力"));
  const chips = element("div", "chips");
  const all = element("label", "chip all-chip");
  const allInput = element("input");
  allInput.type = "checkbox";
  allInput.checked = selected.size === 0;
  allInput.dataset.stage = stageName;
  allInput.dataset.sourceRule = sourceName;
  allInput.dataset.type = "abilities-all";
  all.append(allInput, element("span", "", "全部"));
  chips.append(all);
  for (const ability of orderedDimension(source, "ability")) {
    const label = element("label", "chip");
    const input = element("input");
    input.type = "checkbox";
    input.checked = selected.has(ability);
    input.dataset.stage = stageName;
    input.dataset.sourceRule = sourceName;
    input.dataset.type = "abilities";
    input.value = ability;
    label.append(input, element("span", "", `${ability} · ${formatCount(source.ability?.[ability])}`));
    chips.append(label);
  }
  fieldset.append(chips);
  return fieldset;
}

function samplingGroup(stageName, sourceName, samplePercent) {
  const fieldset = element("fieldset", "rule-group source-sampling");
  fieldset.append(element("legend", "", "随机抽样"));
  const field = element("label", "sample-percent-field");
  const description = element("span");
  description.append(
    element("strong", "", "筛选后保留比例"),
    element("small", "", "按样本稳定随机抽样；同一配方重复预览结果一致。"),
  );
  const control = element("span", "sample-percent-control");
  const input = element("input");
  input.type = "number";
  input.min = "1";
  input.max = "100";
  input.step = "0.1";
  input.value = String(samplePercent);
  input.inputMode = "decimal";
  input.dataset.stage = stageName;
  input.dataset.sourceRule = sourceName;
  input.dataset.type = "sample-percent";
  input.setAttribute("aria-label", `${sourceName} 随机抽样百分比`);
  control.append(input, element("b", "", "%"));
  field.append(description, control);
  fieldset.append(field);
  return fieldset;
}

function sourceRuleSummary(rule) {
  const abilitySummary = rule.abilities.size ? `${rule.abilities.size} 能力` : "全部能力";
  return `${rule.qualities.size} 质量 · ${rule.difficulties.size} 难度 · ${abilitySummary} · 抽样 ${rule.samplePercent}%`;
}

function validSamplePercent(value, raw = String(value)) {
  if (!Number.isFinite(value) || value < 1 || value > 100) return false;
  const decimal = String(raw).trim().split(".")[1] || "";
  return decimal.length <= 6;
}

function renderRules() {
  if (!state.catalog) return;
  for (const stageName of ["stage1", "stage2"]) {
    const stage = state.stages[stageName];
    const root = $(`#${stageName}-rules`);
    root.replaceChildren();
    for (const sourceName of [...stage.sources].sort()) {
      const source = catalogSource(sourceName);
      if (!source) continue;
      const rule = ensureSourceRule(stageName, sourceName);
      const details = element("details", "source-rule-card");
      if (stage.sources.size <= 6) details.open = true;
      const summary = element("summary", "source-rule-summary");
      const identity = element("span");
      identity.append(element("strong", "", sourceName), element("small", "", `${formatCount(source.rows)} 条 · 逐数据集规则`));
      summary.append(identity, element("b", "", sourceRuleSummary(rule)));
      const groups = element("div", "source-rule-groups");
      groups.append(
        samplingGroup(stageName, sourceName, rule.samplePercent),
        ruleGroup(stageName, sourceName, "qualities", "质量", orderedDimension(source, "quality"), state.defaults.quality_labels_zh, rule.qualities, source.quality),
        ruleGroup(stageName, sourceName, "difficulties", "难度", orderedDimension(source, "difficulty"), state.defaults.difficulty_labels_zh, rule.difficulties, source.difficulty),
        abilityGroup(stageName, sourceName, source, rule.abilities),
      );
      details.append(summary, groups);
      root.append(details);
    }
    if (!root.childElementCount) root.append(element("div", "list-empty", "请先为该阶段选择数据集"));
  }
}

function sourceSearchText(source) {
  return `${source.name || ""} ${source.family || ""} ${source.training_role || ""} ${source.split || ""}`.toLocaleLowerCase();
}

function sourceToggle(source, stageName) {
  const label = element("label", "source-toggle");
  const input = element("input");
  input.type = "checkbox";
  input.checked = state.stages[stageName].sources.has(source.name);
  input.disabled = !source.available;
  input.dataset.source = source.name;
  input.dataset.stage = stageName;
  input.setAttribute("aria-label", `${stageName === "stage1" ? "Stage 1" : "Stage 2"} ${input.checked ? "移除" : "使用"} ${source.name}`);
  label.append(input, element("span"));
  return label;
}

function renderSources() {
  const root = $("#source-list");
  root.replaceChildren();
  for (const source of state.catalog?.sources || []) {
    const row = element("div", `source-row${source.available ? "" : " unavailable"}`);
    row.setAttribute("role", "row");
    row.dataset.search = sourceSearchText(source);
    const identity = element("div", "source-identity");
    identity.setAttribute("role", "cell");
    identity.append(element("strong", "", source.name));
    identity.append(element("small", "", [source.family, source.training_role, source.split].filter(Boolean).join(" · ")));
    const meta = element("div", "source-meta");
    meta.setAttribute("role", "cell");
    meta.append(element("strong", "", formatCount(source.rows)), element("small", source.available ? "ok" : "error", source.available ? "可用" : "不可用"));
    if (!source.available && source.errors?.length) meta.title = source.errors.join("\n");
    row.append(identity, meta, sourceToggle(source, "stage1"), sourceToggle(source, "stage2"));
    root.append(row);
  }
  if (!root.childElementCount) root.append(element("div", "list-empty", "没有数据源"));
  applySourceSearch();
}

function applySourceSearch() {
  const query = $("#dataset-search").value.trim().toLocaleLowerCase();
  let visible = 0;
  $$(".source-row", $("#source-list")).forEach((row) => {
    row.hidden = Boolean(query && !row.dataset.search.includes(query));
    if (!row.hidden) visible += 1;
  });
  $("#source-list").classList.toggle("no-match", Boolean(state.catalog && visible === 0));
}

function updateStageCounts() {
  for (const stageName of ["stage1", "stage2"]) {
    const size = state.stages[stageName].sources.size;
    $(`#${stageName}-source-count`).textContent = `${size} 源`;
    $(`#metric-${stageName}-detail`).textContent = `${size} 源`;
  }
}

function chooseSources(mode) {
  if (!state.catalog) return;
  const available = state.catalog.sources.filter((source) => source.available).map((source) => source.name);
  const names = mode === "all" ? available : [];
  for (const stageName of ["stage1", "stage2"]) {
    state.stages[stageName].sources = new Set(names);
    state.stages[stageName].sourceRules = new Map(
      names.map((name) => [name, defaultSourceRule(stageName, name)]),
    );
  }
  renderSources();
  renderRules();
  markRecipeDirty();
  showToast(mode === "clear" ? "已清空" : `已选择 ${names.length} 个数据源`, "success");
}

function renderCatalogSummary() {
  const catalog = state.catalog;
  const missing = Number(catalog.total_sources || 0) - Number(catalog.available_sources || 0);
  $("#catalog-stats").textContent = `${catalog.available_sources} / ${catalog.total_sources} 可用 · ${formatCount(catalog.total_rows)} 条${missing ? ` · ${missing} 不可用` : ""}`;
  $("#overview-sources").textContent = String(catalog.available_sources || 0);
  $("#overview-source-meta").textContent = `${formatCount(catalog.total_rows)} 条已标注`;
}

async function scanCatalog() {
  const button = $("#scan-button");
  if (!pathFields.registry_path.value.trim() || !pathFields.annotations_root.value.trim()) {
    showToast("请填写注册表和标注目录", "error");
    return;
  }
  await withBusy(button, "扫描中…", async () => {
    $("#scan-status").textContent = "扫描中…";
    try {
      state.catalog = await api("/api/catalog", { method: "POST", body: JSON.stringify(pathPayload()) });
      initializeRules();
      const available = state.catalog.sources.filter((source) => source.available).map((source) => source.name);
      for (const stageName of ["stage1", "stage2"]) {
        state.stages[stageName].sources = new Set(available);
        state.stages[stageName].sourceRules = new Map(
          available.map((name) => [name, defaultSourceRule(stageName, name)]),
        );
      }
      state.preview = null;
      state.previewFingerprint = null;
      renderSources();
      renderRules();
      updateStageCounts();
      renderCatalogSummary();
      markRecipeDirty();
      $("#scan-status").textContent = `${state.catalog.available_sources} 个源可用`;
      $("#overview-status").textContent = "数据已就绪，可编辑配方。";
      openSettings(false);
      activateTab("recipe");
      showToast("扫描完成", "success");
    } catch (error) {
      $("#scan-status").textContent = error.message;
      showToast(error.message, "error");
    }
  });
}

async function rebuildSources() {
  const button = $("#rebuild-sources");
  await withBusy(button, "刷新中…", async () => {
    try {
      const result = await api("/api/registry/rebuild", {
        method: "POST",
        body: JSON.stringify({}),
      });
      showToast(`已创建 ${formatCount(result.source_count)} 个 source`, "success");
      await scanCatalog();
    } catch (error) {
      showToast(error.message, "error");
    }
  });
}

function validateRecipe() {
  for (const stageName of ["stage1", "stage2"]) {
    const label = stageName === "stage1" ? "Stage 1" : "Stage 2";
    const stage = state.stages[stageName];
    if (!stage.sources.size) throw new Error(`${label} 至少选择一个数据源`);
    for (const sourceName of stage.sources) {
      const rule = ensureSourceRule(stageName, sourceName);
      if (!rule.qualities.size) throw new Error(`${label} 的 ${sourceName} 至少选择一个质量等级`);
      if (!rule.difficulties.size) throw new Error(`${label} 的 ${sourceName} 至少选择一个难度等级`);
      if (!validSamplePercent(rule.samplePercent)) {
        throw new Error(`${label} 的 ${sourceName} 抽样比例必须是 1–100 的有限数字（最多 6 位小数）`);
      }
    }
  }
}

function distributionLabel(type, key) {
  if (type === "quality") return state.defaults.quality_labels_zh?.[key] || key;
  if (type === "difficulty") return state.defaults.difficulty_labels_zh?.[key] || key;
  return key;
}

function renderDistribution(stageName, preview) {
  const root = $(`#${stageName}-distribution`);
  root.replaceChildren();
  const total = Number(preview.counts[stageName] || 0);
  for (const [type, prefix] of [["quality", "质量"], ["difficulty", "难度"]]) {
    const entries = Object.entries(preview.distributions?.[stageName]?.[type] || {}).sort((left, right) => right[1] - left[1]);
    for (const [key, value] of entries) {
      const item = element("div", "bar-item");
      const progress = element("progress");
      progress.max = total || 1;
      progress.value = value;
      progress.setAttribute("aria-label", `${prefix}${distributionLabel(type, key)} ${formatCount(value)} 条`);
      item.append(element("span", "", `${prefix} · ${distributionLabel(type, key)}`), progress, element("strong", "", formatCount(value)));
      root.append(item);
    }
  }
  $(`#${stageName}-dist-total`).textContent = `${formatCount(total)} 条`;
}

function allAbilityDimensions(preview, stageName) {
  const configuredLevels = Array.isArray(state.catalog?.ability_levels)
    ? state.catalog.ability_levels.filter((value) => typeof value === "string" && value)
    : [];
  const standard = configuredLevels.length
    ? [...new Set(configuredLevels)]
    : [...new Set(state.catalog?.abilities || [])].sort((left, right) => left.localeCompare(right));
  const standardSet = new Set(standard);
  const observed = new Set(state.catalog?.ability_extras || []);
  for (const ability of state.catalog?.abilities || []) observed.add(ability);
  for (const source of state.catalog?.sources || []) {
    for (const ability of Object.keys(source.ability || {})) observed.add(ability);
  }
  for (const ability of Object.keys(preview.distributions?.[stageName]?.ability || {})) observed.add(ability);
  const percentages = preview.ability_percentages?.[stageName]
    || preview.distributions?.[stageName]?.ability_percentages
    || {};
  for (const ability of Object.keys(percentages)) observed.add(ability);
  const extras = [...observed]
    .filter((ability) => !standardSet.has(ability))
    .sort((left, right) => left.localeCompare(right));
  return {
    standard: standard.map((name) => ({ name, extra: false })),
    extras: extras.map((name) => ({ name, extra: true })),
  };
}

function renderAbilityDistribution(stageName, preview) {
  const root = $(`#${stageName}-ability-distribution`);
  root.replaceChildren();
  const total = Number(preview.counts?.[stageName] || 0);
  const counts = preview.distributions?.[stageName]?.ability || {};
  const rawPercentages = preview.ability_percentages?.[stageName]
    || preview.distributions?.[stageName]?.ability_percentages
    || {};
  const percentageSum = Object.values(rawPercentages).reduce((sum, value) => sum + Number(value || 0), 0);
  const percentageScale = percentageSum > 0 && percentageSum <= 1.0001 ? 100 : 1;
  const dimensions = allAbilityDimensions(preview, stageName);
  const abilities = [...dimensions.standard, ...dimensions.extras];
  const standardText = dimensions.standard.length === 15
    ? "15 个标准维度"
    : `${dimensions.standard.length} 个标准维度`;
  $(`#${stageName}-ability-count`).textContent = dimensions.extras.length
    ? `${standardText} · ${dimensions.extras.length} 个非标准标签`
    : standardText;
  for (const dimension of abilities) {
    const ability = dimension.name;
    const count = Number(counts[ability] || 0);
    const supplied = rawPercentages[ability];
    const percent = supplied === undefined
      ? (total ? (count / total) * 100 : 0)
      : Number(supplied || 0) * percentageScale;
    const normalizedPercent = Math.max(0, Math.min(100, percent));
    const item = element("div", `ability-bar-item${dimension.extra ? " nonstandard" : ""}`);
    const progress = element("progress");
    progress.max = 100;
    progress.value = normalizedPercent;
    progress.setAttribute("aria-label", `${ability} ${normalizedPercent.toFixed(1)}%，${formatCount(count)} 条`);
    item.append(
      element("span", "", dimension.extra ? `${ability} · 非标准` : ability),
      progress,
      element("strong", "", `${normalizedPercent.toFixed(1)}%`),
      element("small", "", `${formatCount(count)} 条`),
    );
    root.append(item);
  }
  if (!abilities.length) root.append(element("div", "list-empty", "catalog 中暂无能力维度"));
}

function renderPreview(preview) {
  state.preview = preview;
  const counts = preview.counts || {};
  $("#metric-stage1").textContent = formatCount(counts.stage1);
  $("#metric-stage2").textContent = formatCount(counts.stage2);
  const filteredCounts = preview.filtered_counts || {};
  $("#metric-stage1-detail").textContent = `${state.stages.stage1.sources.size} 源${filteredCounts.stage1 === undefined ? "" : ` · 候选 ${formatCount(filteredCounts.stage1)}`}`;
  $("#metric-stage2-detail").textContent = `${state.stages.stage2.sources.size} 源${filteredCounts.stage2 === undefined ? "" : ` · 候选 ${formatCount(filteredCounts.stage2)}`}`;
  $("#metric-overlap").textContent = formatCount(counts.overlap);
  const denominator = Math.min(Number(counts.stage1 || 0), Number(counts.stage2 || 0));
  $("#metric-overlap-detail").textContent = denominator ? `${((Number(counts.overlap || 0) / denominator) * 100).toFixed(1)}%` : "无重叠";
  $("#overview-stage-counts").textContent = `${formatCount(counts.stage1)} / ${formatCount(counts.stage2)}`;
  $("#overview-overlap").textContent = `重叠 ${formatCount(counts.overlap)} 条`;
  renderDistribution("stage1", preview);
  renderDistribution("stage2", preview);
  renderAbilityDistribution("stage1", preview);
  renderAbilityDistribution("stage2", preview);
  const body = $("#source-results");
  body.replaceChildren();
  for (const source of preview.by_source || []) {
    const row = element("tr");
    const stage1 = `${formatCount(source.stage1)} / ${formatCount(source.stage1_filtered ?? source.stage1)}`;
    const stage2 = `${formatCount(source.stage2)} / ${formatCount(source.stage2_filtered ?? source.stage2)}`;
    for (const value of [source.source, formatCount(source.source_rows), stage1, stage2, formatCount(source.overlap)]) row.append(element("td", "", value));
    body.append(row);
  }
  $("#preview-detail").hidden = false;
  $("#preview-detail").open = true;
  $("#preview-message").textContent = "配方已计算，可发布版本。";
  $("#recipe-state").textContent = "已预览";
  $("#recipe-state").className = "dirty-chip ready";
  renderVersionSummary();
  renderActionState();
}

async function calculatePreview({ quiet = false } = {}) {
  const button = $("#preview-button");
  validateRecipe();
  return withBusy(button, "计算中…", async () => {
    try {
      $("#preview-message").textContent = "正在计算…";
      const payload = selectionPayload();
      const fingerprint = JSON.stringify(payload);
      const preview = await api("/api/preview", { method: "POST", body: JSON.stringify(payload) });
      if (selectionFingerprint() !== fingerprint) {
        $("#preview-message").textContent = "配方已在计算期间改变，请重新预览。";
        markRecipeDirty();
        return null;
      }
      state.previewFingerprint = fingerprint;
      renderPreview(preview);
      if (!quiet) showToast("预览已更新", "success");
      return preview;
    } catch (error) {
      $("#preview-message").textContent = error.message;
      if (!quiet) showToast(error.message, "error");
      throw error;
    }
  });
}

function nextVersionName() {
  if (/^datav[1-9]\d*$/.test(state.nextVersion || "")) return state.nextVersion;
  const numbers = state.versions.map((version) => /^datav(\d+)$/.exec(versionName(version))?.[1]).filter(Boolean).map(Number);
  return `datav${numbers.length ? Math.max(...numbers) + 1 : 3}`;
}

function renderVersionSummary() {
  if (!state.preview) {
    $("#version-summary").textContent = "先在“数据配方”中完成预览。";
    return;
  }
  const counts = state.preview.counts;
  $("#version-summary").textContent = `${state.stages.stage1.sources.size}/${state.stages.stage2.sources.size} 个源 · S1 ${formatCount(counts.stage1)} · S2 ${formatCount(counts.stage2)} · 重叠 ${formatCount(counts.overlap)}`;
}

function renderVersions() {
  const root = $("#version-list");
  root.replaceChildren();
  const sorted = [...state.versions].sort((left, right) => versionName(right).localeCompare(versionName(left), undefined, { numeric: true }));
  for (const version of sorted) {
    const name = versionName(version);
    if (!name) continue;
    const item = element("article", "version-item");
    const header = element("div", "version-item-heading");
    const identity = element("div");
    identity.append(element("strong", "", name));
    identity.append(element("small", "", version.notes || version.description || version.created_at || "无说明"));
    const isActive = name === versionName(state.activeVersion);
    const status = element("span", `status-chip ${isActive ? "active" : ""}`, isActive ? "当前" : (version.registered ? "已注册" : version.status || "已发布"));
    header.append(identity, status);
    const meta = element("p", "", [version.parent ? `基于 ${version.parent}` : "", version.dataset_snapshot_hash ? `#${String(version.dataset_snapshot_hash).slice(0, 10)}` : ""].filter(Boolean).join(" · "));
    const actions = element("div", "compact-actions");
    const register = element("button", "", "注册");
    register.type = "button";
    register.dataset.versionAction = "register";
    register.dataset.version = name;
    register.disabled = Boolean(version.registered);
    const activate = element("button", "", "设为当前");
    activate.type = "button";
    activate.dataset.versionAction = "activate";
    activate.dataset.version = name;
    activate.disabled = isActive;
    actions.append(register, activate);
    item.append(header, meta, actions);
    root.append(item);
  }
  if (!root.childElementCount) root.append(element("div", "list-empty", "暂无版本"));

  const parent = $("#version-parent");
  const selected = parent.value;
  parent.replaceChildren(new Option("无", ""));
  for (const version of sorted) {
    const name = versionName(version);
    if (name) parent.append(new Option(name, name));
  }
  if ([...parent.options].some((option) => option.value === selected)) parent.value = selected;
  $("#version-name").value = nextVersionName();
  renderTrainingVersionOptions();
  renderCurrentVersion();
}

function isTrainableVersion(version) {
  if (!versionName(version) || version?.available === false) return false;
  const status = String(version?.status || "").toLocaleLowerCase();
  return !["deleted", "deleting", "failed", "invalid", "unavailable"].includes(status);
}

function renderTrainingVersionOptions() {
  const select = $("#training-data-version");
  const previous = selectedTrainingVersionName();
  const trainable = state.versions
    .filter(isTrainableVersion)
    .sort((left, right) => versionName(right).localeCompare(versionName(left), undefined, { numeric: true }));
  const activeName = versionName(state.activeVersion);
  const availableNames = new Set(trainable.map(versionName));
  const selected = availableNames.has(previous)
    ? previous
    : (availableNames.has(activeName) ? activeName : (versionName(trainable[0]) || ""));

  select.replaceChildren();
  for (const version of trainable) {
    const name = versionName(version);
    const status = version.registered ? "已注册" : "运行时自动注册";
    const active = name === activeName ? "当前 · " : "";
    select.append(new Option(`${name} · ${active}${status}`, name));
  }
  if (!trainable.length) select.append(new Option("暂无可训练的已发布版本", ""));
  select.value = selected;
  state.selectedTrainingVersion = selected || null;
}

function renderCurrentVersion() {
  const activeName = versionName(state.activeVersion);
  for (const id of ["#current-version-chip", "#overview-version"]) $(id).textContent = activeName || "未发布";
  $("#overview-version-meta").textContent = activeName ? (state.activeVersion.dataset_snapshot_hash ? `#${String(state.activeVersion.dataset_snapshot_hash).slice(0, 12)}` : "已设为当前版本") : "先发布数据版本";

  const trainingVersion = selectedTrainingVersion();
  const trainingName = selectedTrainingVersionName();
  for (const id of ["#training-version", "#evaluation-version"]) $(id).textContent = trainingName ? `训练：${trainingName}` : "未选择版本";
  $("#training-version-help").textContent = trainingVersion
    ? (trainingVersion.registered
      ? `${trainingName} 已注册；本次选择不会修改全局当前版本。`
      : `${trainingName} 尚未注册，启动时会自动注册；不会修改全局当前版本。`)
    : "请先在“版本”页发布数据版本。";
  const stage1 = trainingVersion?.dataset_names?.stage1 || trainingVersion?.composition?.stage1?.dataset_names || trainingVersion?.datasets?.stage1 || [];
  const stage2 = trainingVersion?.dataset_names?.stage2 || trainingVersion?.composition?.stage2?.dataset_names || trainingVersion?.datasets?.stage2 || [];
  $("#stage1-datasets").textContent = Array.isArray(stage1) && stage1.length ? stage1.join(", ") : "由数据版本生成";
  $("#stage2-datasets").textContent = Array.isArray(stage2) && stage2.length ? stage2.join(", ") : "由数据版本生成";
  updateDerivedPaths();
  renderLaunchSteps();
  renderActionState();
}

async function refreshVersions({ quiet = false } = {}) {
  try {
    const payload = await optionalGet("/api/versions", { versions: [] });
    state.versions = normalizeCollection(payload, "versions");
    state.nextVersion = payload?.next_version || null;
    const activeName = payload?.active_version || payload?.current_version;
    state.activeVersion = state.versions.find((version) => version.active || versionName(version) === activeName) || (activeName ? { version: activeName } : null);
    renderVersions();
  } catch (error) {
    if (!quiet) showToast(error.message, "error");
  }
}

async function waitForJob(id, onUpdate) {
  while (true) {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    if (onUpdate) onUpdate(job);
    if (["completed", "failed", "canceled", "cancelled"].includes(job.status)) return job;
    await new Promise((resolve) => window.setTimeout(resolve, 900));
  }
}

async function publishVersion() {
  const button = $("#publish-button");
  const version = $("#version-name").value.trim();
  const notes = $("#version-notes").value.trim();
  if (!/^datav[1-9]\d*$/.test(version)) { showToast("版本号格式应为 datavN", "error"); $("#version-name").focus(); return; }
  if (!notes) { showToast("请填写变更说明", "error"); $("#version-notes").focus(); return; }
  if (state.previewFingerprint !== selectionFingerprint()) { showToast("请先重新预览配方", "error"); activateTab("recipe"); return; }
  await withBusy(button, "发布中…", async () => {
    $("#publish-state").textContent = "发布中";
    const body = {
      version,
      parent: $("#version-parent").value || null,
      notes,
      register: $("#publish-register").checked,
      activate: $("#publish-activate").checked,
      paths: pathPayload(),
      recipe: recipePayload(),
      preview_fingerprint: state.previewFingerprint,
    };
    try {
      let result;
      try {
        const started = await api("/api/versions/publish", { method: "POST", body: JSON.stringify(body) });
        if (started.job_id) {
          state.jobs.unshift({ ...started, kind: "publish", version });
          renderJobs();
          const completed = await waitForJob(started.job_id, (job) => {
            const index = state.jobs.findIndex((item) => jobId(item) === started.job_id);
            if (index >= 0) state.jobs[index] = job; else state.jobs.unshift(job);
            renderJobs();
          });
          if (completed.status === "failed") throw new Error(completed.error || "版本发布失败");
          result = completed.result || completed;
        } else {
          result = started;
        }
      } catch (error) {
        if (error.status !== 404) throw error;
        const legacy = await api("/api/export", { method: "POST", body: JSON.stringify({ ...selectionPayload(), run_name: version }) });
        const completed = legacy.job_id ? await waitForJob(legacy.job_id) : legacy;
        result = { ...(completed.result || completed), version, registered: false, legacy_export: true };
      }
      const publishedVersion = versionName(result.version) || version;
      $("#publish-result").hidden = false;
      $("#publish-result").textContent = result.legacy_export ? `${publishedVersion} 已导出；当前服务未启用训练注册。` : `${publishedVersion} 已发布${body.register ? "并注册" : ""}${result.idempotent ? "（内容未变，复用原版本）" : ""}。`;
      $("#publish-state").textContent = result.legacy_export ? "已导出" : "已发布";
      $("#version-notes").value = "";
      if (result.legacy_export) {
        state.versions.push({ ...result, version, notes, active: false, registered: false });
        renderVersions();
      } else {
        await refreshVersions({ quiet: true });
        const published = state.versions.find((item) => versionName(item) === publishedVersion) || { ...result, version: publishedVersion, notes, active: body.activate, registered: body.register };
        if (body.activate) state.activeVersion = published;
        renderVersions();
      }
      showToast(result.legacy_export ? "版本已导出，注册功能不可用" : "版本发布完成", result.legacy_export ? "info" : "success");
    } catch (error) {
      $("#publish-state").textContent = "失败";
      showToast(error.message, "error");
    }
  });
}

async function versionAction(button) {
  const action = button.dataset.versionAction;
  const version = button.dataset.version;
  const endpoint = action === "register" ? "/api/versions/register" : "/api/versions/activate";
  await withBusy(button, "处理中…", async () => {
    try {
      await api(endpoint, { method: "POST", body: JSON.stringify({ version }) });
      await refreshVersions({ quiet: true });
      showToast(action === "register" ? `${version} 已注册` : `${version} 已激活`, "success");
    } catch (error) { showToast(error.message, "error"); }
  });
}

function valueOf(input) {
  if (input.type === "number") return Number(input.value);
  const value = input.value.trim();
  return value === "" ? null : value;
}

function trainingStagePayload(stageName) {
  const root = $(`[data-train-stage="${stageName}"]`);
  return Object.fromEntries($$("[data-train-param]", root).map((input) => [input.dataset.trainParam, valueOf(input)]));
}

function evaluationPayload() {
  return {
    benchmarks: $$("#benchmark-suites input:checked").map((input) => input.value),
    max_samples: Number($("#eval-max-samples").value),
    offline: $("#eval-offline").checked,
    force_eval: $("#force-eval").checked,
    haystack_split: $("#haystack-split").value,
    tiny_data_partition: $("#tiny-partition").value,
    tiny_partition_seed: Number($("#tiny-seed").value),
    tsr_prompt_mode: $("#tsr-prompt-mode").value,
    protocol_hash: $("#eval-protocol-hash").value.trim() || null,
    tsr_max_model_len: Number($("#tsr-max-model-len").value),
    tsr_max_new_tokens: Number($("#tsr-max-new-tokens").value),
    tsr_batch_size: Number($("#tsr-batch-size").value),
    tsr_request_chunk_size: Number($("#tsr-request-chunk-size").value),
    tiny_max_model_len: Number($("#tiny-max-model-len").value),
    tiny_request_chunk_size: Number($("#tiny-request-chunk-size").value),
    tiny_gpu_memory_utilization: Number($("#tiny-gpu-memory-utilization").value),
    haystack_max_model_len: Number($("#haystack-max-model-len").value),
    haystack_max_new_tokens: Number($("#haystack-max-new-tokens").value),
    haystack_batch_size: Number($("#haystack-batch-size").value),
    haystack_request_chunk_size: Number($("#haystack-request-chunk-size").value),
    exam_max_model_len: Number($("#exam-max-model-len").value),
    exam_max_new_tokens: Number($("#exam-max-new-tokens").value),
    exam_batch_size: Number($("#exam-batch-size").value),
    exam_request_chunk_size: Number($("#exam-request-chunk-size").value),
  };
}

function parseStandaloneModelPaths() {
  const lines = $("#standalone-model-paths").value
    .split(/\r?\n/)
    .map((value) => value.trim())
    .filter(Boolean);
  const modelPaths = [];
  const invalid = [];
  const seen = new Set();
  let duplicateCount = 0;
  for (const path of lines) {
    if (!path.startsWith("/")) {
      invalid.push({ path, reason: "不是绝对路径" });
      continue;
    }
    if (path === "/" || path.endsWith("/")) {
      invalid.push({ path, reason: "必须指向模型目录且末尾不能带 /" });
      continue;
    }
    if (path.split("/").some((part) => part === "." || part === "..")) {
      invalid.push({ path, reason: "不能包含 . 或 .. 路径段" });
      continue;
    }
    const leading = path.startsWith("//") && !path.startsWith("///") ? "//" : "/";
    const canonical = `${leading}${path.split("/").filter(Boolean).join("/")}`;
    if (seen.has(canonical)) {
      duplicateCount += 1;
      continue;
    }
    seen.add(canonical);
    modelPaths.push(canonical);
  }
  return { modelPaths, invalid, duplicateCount, inputCount: lines.length };
}

function standaloneEvaluationIntegration() {
  const pipeline = state.defaults?.pipeline || {};
  const integration = pipeline.integration || state.defaults?.integration || {};
  const evaluationOnly = integration.evaluation_only || {};
  const backend = $("#standalone-eval-backend").value || evaluationOnly.execution_mode || "docker_host";
  const backendStatus = evaluationOnly.backends?.[backend] || {};
  const enabled = Boolean(backendStatus.enabled ?? evaluationOnly.enabled);
  const disabledReasons = Array.isArray(backendStatus.disabled_reasons)
    ? backendStatus.disabled_reasons
    : (Array.isArray(evaluationOnly.disabled_reasons) ? evaluationOnly.disabled_reasons : []);
  return { pipeline, integration, evaluationOnly, backend, backendStatus, enabled, disabledReasons };
}

function configureStandaloneSlurmLaunchers(integration) {
  const evaluationOnly = integration.evaluation_only || {};
  const slurm = evaluationOnly.backends?.slurm || {};
  const select = $("#standalone-eval-sbatch-path");
  const previous = select.value;
  select.replaceChildren();
  for (const launcher of slurm.launchers || []) {
    const value = typeof launcher === "string" ? launcher : launcher.relative_path;
    if (value) select.append(new Option(value, value));
  }
  if (!select.options.length) select.append(new Option("没有可用的单独评测 sbatch", ""));
  const preferred = previous || slurm.default_sbatch;
  if (preferred && [...select.options].some((option) => option.value === preferred)) select.value = preferred;
}

function renderStandaloneEvaluationState() {
  const input = $("#standalone-model-paths");
  const parsed = parseStandaloneModelPaths();
  const contract = standaloneEvaluationIntegration();
  const slurm = contract.backend === "slurm";
  const maxBatchModels = Number(contract.evaluationOnly.max_batch_models || 64);
  const tooMany = parsed.modelPaths.length > maxBatchModels;
  const hasInvalid = parsed.invalid.length > 0;
  const hasBenchmarks = evaluationPayload().benchmarks.length > 0;
  const slurmReady = !slurm || Boolean($("#standalone-eval-sbatch-path").value);
  const canSubmit = contract.enabled
    && parsed.modelPaths.length > 0
    && !hasInvalid
    && !tooMany
    && hasBenchmarks
    && slurmReady;

  state.standaloneEvaluationEnabled = contract.enabled;
  $("#standalone-eval-slurm-field").hidden = !slurm;
  $("#standalone-model-count").textContent = `${parsed.modelPaths.length} 个模型`;
  input.setAttribute("aria-invalid", String(hasInvalid || tooMany));

  const notes = [];
  if (!parsed.inputCount) notes.push("等待输入模型路径");
  if (hasInvalid) notes.push(`${parsed.invalid.length} 条路径无效：${parsed.invalid.slice(0, 2).map((item) => `${item.path}（${item.reason}）`).join("、")}`);
  if (tooMany) notes.push(`单批最多支持 ${maxBatchModels} 个模型`);
  if (parsed.duplicateCount) notes.push(`已忽略 ${parsed.duplicateCount} 条重复路径`);
  if (parsed.modelPaths.length && !hasInvalid && !tooMany) notes.push(`将创建 ${parsed.modelPaths.length} 个独立任务`);
  if (!hasBenchmarks) notes.push("请至少选择一个评测套件");
  if (slurm && !slurmReady) notes.push("请选择单独评测 Slurm 脚本");
  const summary = $("#standalone-model-summary");
  summary.textContent = notes.join("；");
  summary.classList.toggle("error", hasInvalid || tooMany);

  const stateChip = $("#standalone-evaluation-state");
  stateChip.textContent = contract.enabled ? "服务已就绪" : "需要配置";
  stateChip.className = `status-chip ${contract.enabled ? "active" : "error"}`;
  const diagnostic = $("#standalone-evaluation-diagnostic");
  diagnostic.hidden = contract.enabled;
  $("#standalone-evaluation-diagnostic-summary").textContent = contract.enabled
    ? ""
    : `当前${slurm ? " Slurm" : " Docker"} 单独评测后端尚未就绪：`;
  const missing = $("#standalone-evaluation-missing");
  missing.replaceChildren();
  const reasons = contract.disabledReasons.length
    ? contract.disabledReasons
    : ["integration.evaluation_only 未启用或当前后端未配置"];
  for (const reason of reasons) missing.append(element("li", "", reason));

  const container = contract.evaluationOnly.evaluation_container
    || contract.integration.evaluation_container
    || "ragas";
  $("#standalone-eval-topology").textContent = slurm
    ? "Dataset Studio → Slurm 队列 → Singularity（ragas.sif）"
    : `Dataset Studio → Docker FIFO → ${container}（评测）`;
  const outputBase = state.evaluationOutputBaseTemplate.replace(/\/$/, "");
  $("#standalone-eval-output-root").value = outputBase
    ? `${outputBase}/<模型名-路径哈希>/protocol-<服务端计算>`
    : "由服务端配置";

  $("#standalone-eval-preflight").disabled = !canSubmit || state.busy.has("standalone-eval-preflight");
  $("#standalone-eval-submit").disabled = !canSubmit || state.busy.has("standalone-eval-submit");
}

function validateStandaloneEvaluation() {
  const parsed = parseStandaloneModelPaths();
  if (!parsed.inputCount) throw new Error("请至少填写一个模型绝对路径");
  if (parsed.invalid.length) throw new Error(`模型路径无效：${parsed.invalid[0].path}（${parsed.invalid[0].reason}）`);
  const maxBatchModels = Number(standaloneEvaluationIntegration().evaluationOnly.max_batch_models || 64);
  if (parsed.modelPaths.length > maxBatchModels) throw new Error(`单批最多支持 ${maxBatchModels} 个模型`);
  if (!evaluationPayload().benchmarks.length) throw new Error("至少选择一个评测套件");
  const contract = standaloneEvaluationIntegration();
  if (!contract.enabled) throw new Error("当前单独评测后端尚未就绪，请查看配置诊断");
  if (contract.backend === "slurm" && !$("#standalone-eval-sbatch-path").value) {
    throw new Error("请选择可信的单独评测 Slurm sbatch 脚本");
  }
  return parsed.modelPaths;
}

function standaloneEvaluationPayload() {
  const contract = standaloneEvaluationIntegration();
  return {
    model_paths: validateStandaloneEvaluation(),
    execution: {
      backend: contract.backend,
      sbatch_path: contract.backend === "slurm" ? $("#standalone-eval-sbatch-path").value : null,
    },
    evaluation: evaluationPayload(),
  };
}

async function startStandaloneEvaluation(preflight, button) {
  let payload;
  try { payload = standaloneEvaluationPayload(); } catch (error) { showToast(error.message, "error"); return; }
  const endpoint = preflight ? "/api/evaluations/preflight" : "/api/evaluations";
  await withBusy(button, preflight ? "提交预检中…" : "批量提交中…", async () => {
    try {
      const result = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      const submitted = normalizeCollection(result, "jobs");
      const submittedIds = new Set(submitted.map(jobId));
      state.jobs = [...submitted, ...state.jobs.filter((job) => !submittedIds.has(jobId(job)))];
      if (submitted[0]) state.activeJobId = jobId(submitted[0]);
      renderJobs();
      const count = Number(result.job_count || submitted.length || payload.model_paths.length);
      const queueLabel = payload.execution.backend === "slurm"
        ? "已提交到 Slurm，后续由调度器排队"
        : "已加入本地 FIFO，将按提交顺序运行";
      const actionLabel = preflight ? "预检" : "评测";
      $("#standalone-evaluation-status").textContent = `批次 ${result.batch_id || "—"}：${count} 个${actionLabel}任务${queueLabel}。`;
      showToast(`${count} 个${actionLabel}任务${queueLabel}`, "success");
      activateTab("jobs");
      await refreshJobs({ quiet: true });
    } catch (error) {
      $("#standalone-evaluation-status").textContent = error.message;
      showToast(error.message, "error");
    }
  });
}

function runPayload(mode) {
  const profile = $("#train-profile").value;
  const backend = $("#execution-backend").value;
  const payload = {
    mode,
    version: selectedTrainingVersionName(),
    execution: {
      backend,
      sbatch_path: backend === "slurm" ? $("#slurm-sbatch-path").value : null,
    },
    training: {
      profile,
      base_model_path: $("#base-model-path").value.trim(),
      seed: Number($("#train-seed").value),
      deepspeed_include: $("#deepspeed-include").value.trim(),
      master_port: Number($("#master-port").value),
      keep_stage1: $("#keep-stage1").checked,
      force_train: $("#force-train").checked,
      stage1: trainingStagePayload("stage1"),
      stage2: trainingStagePayload("stage2"),
    },
  };
  payload.evaluation = evaluationPayload();
  return payload;
}

function effectiveRunMode() {
  return "train_eval";
}

function validateRun(mode) {
  if (!selectedTrainingVersionName()) throw new Error("请先在训练页选择一个已发布的数据版本");
  const baseModelPath = $("#base-model-path").value.trim();
  if (!baseModelPath.startsWith("/")) throw new Error("请填写训练容器内可见的基础模型绝对路径");
  if (mode === "train_eval" && !evaluationPayload().benchmarks.length) throw new Error("至少选择一个评测套件");
  if ($("#execution-backend").value === "slurm" && !$("#slurm-sbatch-path").value) throw new Error("请选择可信的 Slurm sbatch 脚本");
}

async function startRun(mode, button) {
  const effectiveMode = effectiveRunMode();
  try { validateRun(effectiveMode); } catch (error) { showToast(error.message, "error"); return; }
  const endpoint = mode === "preflight" ? "/api/runs/preflight" : "/api/runs";
  await withBusy(button, mode === "preflight" ? "预检中…" : "启动中…", async () => {
    try {
      const result = await api(endpoint, { method: "POST", body: JSON.stringify(runPayload(effectiveMode)) });
      const id = jobId(result);
      const submittedStatus = result.status === "queued"
        ? `已加入队列，当前第 ${result.queue_position || 1} 位。`
        : "已经开始运行。";
      $("#run-status").textContent = mode === "preflight" ? `预检任务${submittedStatus}` : `训练评测任务${submittedStatus}`;
      if (id) {
        state.activeJobId = id;
        state.jobs.unshift(result);
        renderJobs();
      }
      const queueSuffix = result.status === "queued" ? `，当前第 ${result.queue_position || 1} 位` : "";
      showToast(mode === "preflight" ? `预检已提交${queueSuffix}` : (result.status === "queued" ? `流水线已加入队列${queueSuffix}` : "流水线已启动"), "success");
      activateTab("jobs");
      refreshJobs({ quiet: true });
    } catch (error) {
      $("#run-status").textContent = error.message;
      showToast(error.message, "error");
    }
  });
}

function jobStatusLabel(status) {
  return ({ queued: "本地排队", scheduled: "Slurm 排队", preparing: "准备", running: "运行中", exporting: "导出中", training: "训练中", evaluating: "评测中", completed: "完成", failed: "失败", canceled: "已取消", cancelled: "已取消" })[status] || status || "未知";
}

function jobType(job) {
  const value = job.type || job.kind || job.mode || job.phase || "任务";
  const stage1 = job.profile === "chronos2-stage1"
    || job.training_profile === "chronos2-stage1"
    || job.training?.profile === "chronos2-stage1";
  return ({
    train_eval: stage1 ? "Stage 1 训练 + 评测" : "两阶段训练 + 评测",
    train_eval_stage1: "Stage 1 训练 + 评测",
    train_eval_full: "两阶段训练 + 评测（Slurm）",
    train_stage1: "Stage 1 训练 + 评测",
    train_full: "两阶段训练 + 评测",
    preflight: "Preflight",
    evaluation: "单独评测",
    evaluation_preflight: "单独评测 Preflight",
    publish: "发布版本",
  })[value] || value;
}

function jobTargetLabel(job) {
  const explicit = job.version || job.data_version || job.model_name || job.derived?.model_name;
  if (explicit) return String(explicit);
  const path = job.model_path || job.derived?.model_path;
  if (!path) return "—";
  return String(path).split("/").filter(Boolean).at(-1) || String(path);
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
}

function formatDuration(job) {
  let seconds = Number(job.duration_seconds || 0);
  if (!seconds && job.started_at && !["completed", "failed", "canceled", "cancelled"].includes(job.status)) seconds = Math.max(0, (Date.now() - new Date(job.started_at).getTime()) / 1000);
  if (!seconds) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m ${Math.floor(seconds % 60)}s`;
}

function jobRow(job) {
  const button = element("button", "job-row");
  button.type = "button";
  button.dataset.jobId = jobId(job);
  button.setAttribute("role", "row");
  button.setAttribute("aria-label", `查看${jobType(job)}任务日志`);
  const queueSuffix = job.status === "queued" && job.queue_position ? ` #${job.queue_position}` : "";
  const statusLabel = job.status === "queued" && job.execution_backend === "docker_host"
    ? "Docker 排队"
    : jobStatusLabel(job.status);
  const status = element("span", `job-status ${job.status || "unknown"}`, `${statusLabel}${queueSuffix}`);
  for (const value of [jobType(job), jobTargetLabel(job)]) button.append(element("span", "", String(value)));
  button.append(status, element("span", "", formatDate(job.started_at || job.created_at)), element("span", "", formatDuration(job)));
  return button;
}

function renderJobs() {
  const root = $("#job-list");
  root.replaceChildren(...state.jobs.map(jobRow));
  if (!state.jobs.length) root.append(element("div", "list-empty", "暂无任务"));
  const recent = $("#recent-jobs");
  recent.replaceChildren(...state.jobs.slice(0, 5).map(jobRow));
  if (!state.jobs.length) recent.append(element("div", "list-empty", "暂无任务"));
  const running = state.jobs.filter((job) => !["completed", "failed", "canceled", "cancelled", "queued"].includes(job.status)).length;
  const queued = state.jobs.filter((job) => job.status === "queued").length;
  $("#running-job-count").hidden = running + queued === 0;
  $("#running-job-count").textContent = String(running + queued);
  $("#overview-pipeline").textContent = running || queued ? `${running} 个运行中 · ${queued} 个排队` : (state.jobs[0] ? jobStatusLabel(state.jobs[0].status) : "空闲");
  $("#overview-job-meta").textContent = state.jobs[0] ? `${jobType(state.jobs[0])} · ${jobTargetLabel(state.jobs[0])}` : "没有运行中的任务";
  renderLaunchSteps();
}

async function refreshJobs({ quiet = false } = {}) {
  try {
    const payload = await optionalGet("/api/jobs", { jobs: [] });
    state.jobs = normalizeCollection(payload, "jobs");
    renderJobs();
  } catch (error) { if (!quiet) showToast(error.message, "error"); }
}

function renderJobDialog(job) {
  $("#job-dialog-title").textContent = `${jobType(job)} · ${jobStatusLabel(job.status)}`;
  const batchMeta = job.batch_id ? ` · 批次 ${job.batch_id}` : "";
  $("#job-dialog-meta").textContent = `${jobTargetLabel(job)} · ${jobId(job)}${batchMeta}`;
  $("#job-phase").textContent = job.phase || jobStatusLabel(job.status);
  const total = Number(job.total_rows || job.total || 0);
  const processed = Number(job.processed_rows || job.processed || 0);
  const percent = Number(job.progress_percent ?? (total ? Math.floor((processed / total) * 100) : (job.status === "completed" ? 100 : 0)));
  $("#job-percent").textContent = `${Math.max(0, Math.min(100, percent))}%`;
  $("#job-progress").value = Math.max(0, Math.min(100, percent));
  const waiting = job.status === "queued" ? `任务已冻结并进入队列，当前第 ${job.queue_position || "?"} 位。` : "等待日志";
  const log = Array.isArray(job.log_tail) ? job.log_tail.join("\n") : (job.log_tail || job.log || job.error || waiting);
  $("#job-log").textContent = log;
  const diff = job.diff_from_previous;
  const diffPanel = $("#job-diff");
  const diffList = $("#job-diff-list");
  diffList.replaceChildren();
  const trainingKinds = new Set([
    "train_eval",
    "train_eval_stage1",
    "train_eval_full",
    "train_stage1",
    "train_full",
  ]);
  diffPanel.hidden = !diff || !trainingKinds.has(job.kind);
  if (diff && trainingKinds.has(job.kind)) {
    if (!diff.has_previous_run) {
      $("#job-diff-summary").textContent = "这是第一条训练记录，没有可比较的上一次训练。";
    } else if (!diff.change_count) {
      $("#job-diff-summary").textContent = `与上一次训练 ${diff.previous_job_id} 的数据和参数完全一致。`;
    } else {
      const displaySuffix = diff.truncated ? `（页面显示前 ${diff.displayed_change_count} 项，完整差异见产物）` : "";
      $("#job-diff-summary").textContent = `相对训练 ${diff.previous_job_id}，共有 ${diff.change_count} 项变化。${displaySuffix}`;
      for (const change of diff.changes || []) {
        const row = element("div", "job-diff-row");
        row.append(
          element("strong", "", change.path),
          element("code", "", JSON.stringify(change.before)),
          element("span", "", "→"),
          element("code", "", JSON.stringify(change.after)),
        );
        diffList.append(row);
      }
    }
  }
  const artifacts = $("#job-artifacts");
  artifacts.replaceChildren();
  const values = job.artifacts || job.result?.artifacts || job.result || {};
  if (values && typeof values === "object") {
    for (const [label, value] of Object.entries(values)) {
      if (typeof value !== "string" || !value.startsWith("/")) continue;
      const row = element("div", "artifact-row");
      row.append(element("span", "", label), element("code", "", value));
      const copy = element("button", "text-button", "复制");
      copy.type = "button";
      copy.dataset.copy = value;
      row.append(copy);
      artifacts.append(row);
    }
  }
}

async function openJob(id) {
  if (!id) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(id)}`);
    state.activeJobId = id;
    renderJobDialog(job);
    const dialog = $("#job-dialog");
    if (!dialog.open) dialog.showModal();
  } catch (error) { showToast(error.message, "error"); }
}

async function copyText(value, button) {
  try {
    await navigator.clipboard.writeText(value);
    const original = button.textContent;
    button.textContent = "已复制";
    window.setTimeout(() => { button.textContent = original; }, 1200);
  } catch { showToast("无法访问剪贴板", "error"); }
}

function updateDerivedPaths() {
  const version = selectedTrainingVersionName();
  const root = $("#train-output-root").value.trim();
  const seed = $("#train-seed").value;
  const cleanRoot = root.replace(/\/$/, "").replace(/(?:[-_]?data-?v\d+)$/i, "");
  const finalName = $("#train-profile").value === "chronos2-stage1" ? `best_stage1_seed${seed}` : `best_seed${seed}`;
  const finalModelPreview = cleanRoot && version
    ? `${cleanRoot}-${version}/experiments/recipe-<训练参数哈希>/${finalName}`
    : "由服务端按版本与训练参数生成";
  $("#final-model-preview").textContent = finalModelPreview;
  $("#eval-model-path").value = cleanRoot && version ? finalModelPreview : "";
  const evaluationBase = state.evaluationOutputBaseTemplate.replace(/\/$/, "");
  $("#eval-output-root").value = evaluationBase && version
    ? `${evaluationBase}/<模型名>/protocol-<服务端计算>`
    : evaluationBase;
  $("#suite-count").textContent = `${$$("#benchmark-suites input:checked").length} 项`;
}

function modelScaleFromPath(path) {
  const name = path.split("/").filter(Boolean).at(-1) || "";
  const matches = [...name.matchAll(/(^|[^A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BM])(?=$|[^A-Za-z0-9])/gi)];
  if (!matches.length) return null;
  const match = matches.at(-1);
  return `${match[2]}${match[3].toUpperCase()}`;
}

function withModelScale(template, scale) {
  if (!template || !scale) return template;
  const matches = [...template.matchAll(/(^|[^A-Za-z0-9])(\d+(?:\.\d+)?)\s*([BM])(?=$|[^A-Za-z0-9])/gi)];
  if (!matches.length) return template;
  const match = matches.at(-1);
  const tokenStart = match.index + match[1].length;
  const tokenEnd = match.index + match[0].length;
  return `${template.slice(0, tokenStart)}${scale}${template.slice(tokenEnd)}`;
}

function syncModelOutputScale() {
  const scale = modelScaleFromPath($("#base-model-path").value.trim());
  $("#train-output-root").value = withModelScale(state.modelOutputBaseTemplate, scale);
  $("#base-model-sync-hint").textContent = scale
    ? `已识别 ${scale}，模型输出目录已同步。`
    : "未识别到 8B / 4B / 1.7B 形式的参数量，将保留服务端输出目录模板。";
  updateDerivedPaths();
}

function integrationMissingItems(pipeline, integration) {
  const training = pipeline.training || {};
  const evaluation = pipeline.evaluation || {};
  const required = [
    ["integration.pipeline_script", integration.pipeline_script],
    ["integration.training_root", integration.training_root],
    ["integration.evaluation_root", integration.evaluation_root],
    ["training.base_model_path（也可在训练页填写）", $("#base-model-path").value.trim() || training.base_model_path],
    ["integration.model_output_base", training.output_root],
    ["integration.evaluation_output_base", evaluation.output_root],
    ["integration.tsrbench_root", evaluation.tsrbench_root],
    ["integration.tinybench_dataset_root", evaluation.tinybench_dataset_root],
    ["integration.ts_haystack_root", evaluation.ts_haystack_root],
    ["integration.timeseriesexam_root", evaluation.timeseriesexam_root],
    ["integration.timeseriesexam_data_file", evaluation.timeseriesexam_data_file],
  ];
  return required.filter(([, value]) => value === null || value === undefined || value === "").map(([name]) => name);
}

function selectedBackendStatus(integration) {
  const backend = $("#execution-backend").value || integration.execution_mode || "docker_host";
  const profile = $("#train-profile").value || "chronos2-full";
  const backendStatus = integration.backends?.[backend];
  return backendStatus?.profiles?.[profile] || backendStatus || {
    enabled: integration.enabled,
    disabled_reasons: integration.disabled_reasons || integration.missing || [],
  };
}

function refreshTrainingModeUI() {
  const profile = $("#train-profile").value;
  const backend = $("#execution-backend").value;
  const stage1Only = profile === "chronos2-stage1";
  const slurm = backend === "slurm";
  $("#slurm-sbatch-field").hidden = !slurm;
  $("#stage2-training-card").hidden = stage1Only;
  $("#evaluation-config-card").hidden = false;
  $("#keep-stage1").disabled = stage1Only;
  if (stage1Only) $("#keep-stage1").checked = true;

  const primaryLabel = stage1Only
    ? (slurm ? "提交 Stage 1 Slurm 训练 + 评测" : "Stage 1 训练 + 评测")
    : (slurm ? "提交两阶段 Slurm 训练 + 评测" : "两阶段训练 + 评测");
  $("#primary-run-button").textContent = primaryLabel;
  $("#primary-run-button").dataset.defaultLabel = primaryLabel;
  const overview = $("#overview-run");
  $("strong", overview).textContent = primaryLabel;
  $("small", overview).textContent = slurm ? "提交到 Slurm，任务配置会被冻结" : "启动或加入 FIFO 队列";
  const launchTitle = $("#launch-step-run strong");
  const launchHelp = $("#launch-step-run small");
  $("#launch-step-training small").textContent = stage1Only
    ? "在“训练”页确认基础模型、GPU 和 Stage 1 参数。"
    : "在“训练”页确认基础模型、GPU 和两阶段参数。";
  $("#launch-guide-title + p").textContent = "训练完成后，将使用该阶段的最终模型继续评测。";
  launchTitle.textContent = primaryLabel;
  launchHelp.textContent = slurm
    ? "提交可信 sbatch；训练完成后自动衔接所选 benchmark。"
    : stage1Only
      ? "保存 Stage 1 最终权重后，直接使用它运行所选 benchmark。"
      : "有任务运行时自动排队，训练完成后继续评测。";

  const pipeline = state.defaults?.pipeline || {};
  renderIntegrationStatus(pipeline, pipeline.integration || state.defaults?.integration || {});
  updateDerivedPaths();
  renderActionState();
}

function renderLaunchSteps() {
  const version = selectedTrainingVersionName();
  const sameVersion = (job) => !job.version || !version || job.version === version;
  const preflightDone = state.jobs.some((job) => job.kind === "preflight" && job.status === "completed" && sameVersion(job));
  const pipelineDone = state.jobs.some((job) => [
    "train_eval",
    "train_eval_stage1",
    "train_eval_full",
    "train_stage1",
    "train_full",
    "pipeline",
  ].includes(job.kind) && job.status === "completed" && sameVersion(job));
  $("#launch-step-version").classList.toggle("complete", Boolean(version));
  $("#launch-step-training").classList.toggle("complete", Boolean($("#base-model-path").value.trim()));
  $("#launch-step-preflight").classList.toggle("complete", preflightDone);
  $("#launch-step-run").classList.toggle("complete", pipelineDone);
}

function renderIntegrationStatus(pipeline, integration) {
  const status = selectedBackendStatus(integration);
  const enabled = Boolean(status.enabled);
  const missing = Array.isArray(status.disabled_reasons)
    ? status.disabled_reasons
    : (Array.isArray(integration.missing) ? integration.missing : integrationMissingItems(pipeline, integration));
  state.pipelineEnabled = enabled;
  $("#integration-state").textContent = enabled ? "服务已就绪" : "需要配置";
  $("#integration-state").className = `status-chip ${enabled ? "active" : "error"}`;
  $("#integration-diagnostic").hidden = enabled;
  $("#integration-summary").textContent = enabled ? "" : "当前训练方案与执行后端尚未就绪：";
  const list = $("#integration-missing");
  list.replaceChildren();
  const items = missing.length ? missing : ["integration.pipeline_script（路径未配置或文件不存在）"];
  for (const item of items) list.append(element("li", "", item));
  const trainingContainer = integration.training_container || "chatts";
  const evaluationContainer = integration.evaluation_container || "ragas";
  const backend = $("#execution-backend").value || integration.execution_mode;
  const trainingLabel = $("#train-profile").value === "chronos2-stage1" ? "Stage 1 训练" : "两阶段训练";
  $("#execution-topology").textContent = backend === "docker_host"
    ? `宿主机 Dataset Studio → ${trainingContainer}（${trainingLabel}）→ ${evaluationContainer}（评测）`
    : `宿主机 Dataset Studio → Slurm sbatch → Singularity（${trainingLabel}）→ 评测流水线`;
  renderLaunchSteps();
}

function applyDefaults(defaults) {
  const paths = defaults.paths || defaults;
  for (const [key, input] of Object.entries(pathFields)) if (typeof paths[key] === "string") input.value = paths[key];
  const pipeline = defaults.pipeline || {};
  const training = pipeline.training || defaults.training || {};
  const integration = pipeline.integration || defaults.integration || {};
  $("#training-root").value = integration.training_root || defaults.training_root || "由服务端配置";
  $("#base-model-path").value = training.base_model_path || defaults.base_model_path || "";
  state.modelOutputBaseTemplate = training.output_root || defaults.train_output_root || "";
  $("#train-output-root").value = state.modelOutputBaseTemplate;
  $("#train-profile").value = training.profile || "chronos2-full";
  $("#execution-backend").value = integration.execution_mode || "docker_host";
  const evaluationOnly = integration.evaluation_only || {};
  $("#standalone-eval-backend").value = evaluationOnly.execution_mode || integration.execution_mode || "docker_host";
  configureStandaloneSlurmLaunchers(integration);
  const launchers = integration.backends?.slurm?.launchers || [];
  const sbatchSelect = $("#slurm-sbatch-path");
  sbatchSelect.replaceChildren();
  for (const launcher of launchers) {
    sbatchSelect.append(new Option(launcher.relative_path, launcher.relative_path));
  }
  if (!launchers.length) sbatchSelect.append(new Option("没有可用的 Studio sbatch", ""));
  const defaultSbatch = integration.backends?.slurm?.default_sbatch;
  if (defaultSbatch && [...sbatchSelect.options].some((option) => option.value === defaultSbatch)) sbatchSelect.value = defaultSbatch;
  $("#train-seed").value = training.seed ?? 42;
  $("#deepspeed-include").value = training.deepspeed_include || "localhost:0,1,2,3,4,5,6,7";
  $("#master-port").value = training.master_port ?? 19901;
  $("#keep-stage1").checked = Boolean(training.keep_stage1);
  $("#force-train").checked = Boolean(training.force_train);
  for (const stageName of ["stage1", "stage2"]) {
    const values = training[stageName] || {};
    for (const input of $$("[data-train-param]", $(`[data-train-stage="${stageName}"]`))) {
      if (values[input.dataset.trainParam] !== undefined) input.value = values[input.dataset.trainParam] ?? "";
    }
  }
  const evaluation = pipeline.evaluation || defaults.evaluation || {};
  $("#eval-project-root").value = integration.evaluation_root || evaluation.project_root || defaults.eval_project_root || "由服务端配置";
  state.evaluationOutputBaseTemplate = evaluation.output_root || defaults.eval_output_root || "";
  const suites = new Set(evaluation.benchmarks || ["tsrbench", "tinybenchmarks", "ts_haystack", "timeseriesexam"]);
  $$("#benchmark-suites input").forEach((input) => { input.checked = suites.has(input.value); });
  const evaluationFields = {
    "#eval-max-samples": "max_samples",
    "#haystack-split": "haystack_split",
    "#tiny-partition": "tiny_data_partition",
    "#tiny-seed": "tiny_partition_seed",
    "#tsr-prompt-mode": "tsr_prompt_mode",
    "#tsr-max-model-len": "tsr_max_model_len",
    "#tsr-max-new-tokens": "tsr_max_new_tokens",
    "#tsr-batch-size": "tsr_batch_size",
    "#tsr-request-chunk-size": "tsr_request_chunk_size",
    "#tiny-max-model-len": "tiny_max_model_len",
    "#tiny-request-chunk-size": "tiny_request_chunk_size",
    "#tiny-gpu-memory-utilization": "tiny_gpu_memory_utilization",
    "#haystack-max-model-len": "haystack_max_model_len",
    "#haystack-max-new-tokens": "haystack_max_new_tokens",
    "#haystack-batch-size": "haystack_batch_size",
    "#haystack-request-chunk-size": "haystack_request_chunk_size",
    "#exam-max-model-len": "exam_max_model_len",
    "#exam-max-new-tokens": "exam_max_new_tokens",
    "#exam-batch-size": "exam_batch_size",
    "#exam-request-chunk-size": "exam_request_chunk_size",
  };
  for (const [selector, key] of Object.entries(evaluationFields)) if (evaluation[key] !== undefined) $(selector).value = evaluation[key];
  $("#eval-offline").checked = evaluation.offline !== false;
  $("#force-eval").checked = Boolean(evaluation.force_eval);
  $("#tsrbench-root").value = evaluation.tsrbench_root || "由服务端配置";
  $("#tinybench-root").value = evaluation.tinybench_dataset_root || "由服务端配置";
  $("#haystack-root").value = evaluation.ts_haystack_root || "由服务端配置";
  $("#timeseriesexam-root").value = evaluation.timeseriesexam_root || "由服务端配置";
  $("#timeseriesexam-file").value = evaluation.timeseriesexam_data_file || "由服务端配置";
  refreshTrainingModeUI();
  $("#run-status").textContent = state.pipelineEnabled
    ? "先运行 Preflight；通过后启动当前训练方案。"
    : "请先按上方提示补齐当前执行后端配置。";
}

function handleRuleChange(event) {
  const input = event.target.closest("input[data-stage][data-source-rule][data-type]");
  if (!input) return;
  const rule = ensureSourceRule(input.dataset.stage, input.dataset.sourceRule);
  if (input.dataset.type === "sample-percent") {
    const value = Number(input.value);
    if (!validSamplePercent(value, input.value)) {
      showToast("随机抽样比例必须是 1–100 的有限数字，最多 6 位小数", "error");
      input.value = String(rule.samplePercent);
      input.focus();
      return;
    }
    rule.samplePercent = value;
  } else if (input.dataset.type === "abilities-all") {
    rule.abilities.clear();
    $$('input[data-type="abilities"]', input.closest(".source-rule-card")).forEach((item) => { item.checked = false; });
    input.checked = true;
  } else {
    const values = rule[input.dataset.type];
    if (input.checked) values.add(input.value); else values.delete(input.value);
    if (input.dataset.type === "abilities") {
      const all = $('input[data-type="abilities-all"]', input.closest(".source-rule-card"));
      if (all) all.checked = values.size === 0;
    }
  }
  const summary = $(".source-rule-summary b", input.closest(".source-rule-card"));
  if (summary) summary.textContent = sourceRuleSummary(rule);
  markRecipeDirty();
}

function handleSourceChange(event) {
  const input = event.target.closest("input[data-source][data-stage]");
  if (!input) return;
  const stage = state.stages[input.dataset.stage];
  if (input.checked) {
    stage.sources.add(input.dataset.source);
    ensureSourceRule(input.dataset.stage, input.dataset.source);
  } else {
    stage.sources.delete(input.dataset.source);
    stage.sourceRules.delete(input.dataset.source);
  }
  renderSources();
  renderRules();
  markRecipeDirty();
}

function bindEvents() {
  $$("[role=tab]").forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab));
    tab.addEventListener("keydown", handleTabKeys);
  });
  $$('[data-tab-link]').forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); activateTab(link.dataset.tabLink); }));
  $("#open-settings").addEventListener("click", () => openSettings(true));
  for (const selector of ["#close-settings", "#settings-backdrop"]) $(selector).addEventListener("click", () => openSettings(false));
  $("#scan-button").addEventListener("click", scanCatalog);
  $("#rebuild-sources").addEventListener("click", rebuildSources);
  $("#preview-button").addEventListener("click", () => calculatePreview().catch(() => {}));
  $("#dataset-search").addEventListener("input", applySourceSearch);
  $("#select-all").addEventListener("click", () => chooseSources("all"));
  $("#clear-sources").addEventListener("click", () => chooseSources("clear"));
  $("#source-list").addEventListener("change", handleSourceChange);
  $("#rules-section").addEventListener("change", handleRuleChange);
  // Keep the percentage summary and preview fingerprint in sync while the
  // user is typing.  Number inputs do not emit `change` until they lose
  // focus, which made it easy to click Preview while the old 100% value was
  // still held in state.
  $("#rules-section").addEventListener("input", (event) => {
    if (event.target.matches('input[data-type="sample-percent"]')) {
      handleRuleChange(event);
    }
  });
  Object.values(pathFields).forEach((input) => input.addEventListener("change", markRecipeDirty));
  $("#publish-button").addEventListener("click", publishVersion);
  $("#refresh-versions").addEventListener("click", () => refreshVersions());
  $("#version-list").addEventListener("click", (event) => { const button = event.target.closest("button[data-version-action]"); if (button) versionAction(button); });
  $("#version-name").addEventListener("input", updateDerivedPaths);
  $("#training-data-version").addEventListener("change", (event) => {
    state.selectedTrainingVersion = event.target.value || null;
    renderCurrentVersion();
  });
  $("#train-seed").addEventListener("input", updateDerivedPaths);
  $("#train-profile").addEventListener("change", refreshTrainingModeUI);
  $("#execution-backend").addEventListener("change", refreshTrainingModeUI);
  $("#slurm-sbatch-path").addEventListener("change", renderActionState);
  $("#standalone-model-paths").addEventListener("input", renderActionState);
  $("#standalone-eval-backend").addEventListener("change", renderActionState);
  $("#standalone-eval-sbatch-path").addEventListener("change", renderActionState);
  $("#base-model-path").addEventListener("input", () => {
    syncModelOutputScale();
    const pipeline = state.defaults?.pipeline || {};
    renderIntegrationStatus(pipeline, pipeline.integration || state.defaults?.integration || {});
    renderActionState();
  });
  $("#benchmark-suites").addEventListener("change", () => {
    updateDerivedPaths();
    renderActionState();
  });
  $("#standalone-eval-preflight").addEventListener("click", () => startStandaloneEvaluation(true, $("#standalone-eval-preflight")));
  $("#standalone-eval-submit").addEventListener("click", () => startStandaloneEvaluation(false, $("#standalone-eval-submit")));
  $$(".run-button").forEach((button) => button.addEventListener("click", () => startRun(button.dataset.runMode, button)));
  $("#overview-preflight").addEventListener("click", () => startRun("preflight", $("#overview-preflight")));
  $("#overview-run").addEventListener("click", () => startRun(effectiveRunMode(), $("#overview-run")));
  $("#refresh-jobs").addEventListener("click", () => refreshJobs());
  for (const root of [$("#job-list"), $("#recent-jobs")]) root.addEventListener("click", (event) => { const row = event.target.closest("[data-job-id]"); if (row) openJob(row.dataset.jobId); });
  $("#close-job-dialog").addEventListener("click", () => $("#job-dialog").close());
  $("#job-dialog").addEventListener("click", (event) => { const button = event.target.closest("button[data-copy]"); if (button) copyText(button.dataset.copy, button); });
  document.addEventListener("keydown", (event) => {
    const settings = $("#settings-panel");
    if (settings.hidden) return;
    if (event.key === "Escape") { openSettings(false); return; }
    if (event.key !== "Tab") return;
    const focusable = $$('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled)', settings);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
}

async function boot() {
  bindEvents();
  const initial = location.hash.slice(1);
  activateTab($( `[data-tab="${initial}"]`) ? initial : "overview", { updateHash: false });
  try {
    state.defaults = await api("/api/defaults");
    applyDefaults(state.defaults);
    initializeRules();
    setHealth("online", "服务正常");
    $("#scan-status").textContent = "可扫描";
    await Promise.all([refreshVersions({ quiet: true }), refreshJobs({ quiet: true })]);
  } catch (error) {
    setHealth("error", "服务不可用");
    $("#overview-status").textContent = error.message;
    showToast(error.message, "error");
  }
  renderActionState();
  window.setInterval(async () => {
    if (document.hidden) return;
    await refreshJobs({ quiet: true });
    if (state.activeJobId && $("#job-dialog").open) openJob(state.activeJobId);
  }, 5000);
}

boot();
