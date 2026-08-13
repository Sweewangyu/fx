"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Dataset = {
  name: string;
  family: string;
  split: string;
  training_role: string;
  rows: number;
  templates: number;
  compression_ratio: number;
  has_audit: boolean;
  has_annotations: boolean;
  index_ready: boolean;
  schema: "chatts" | "tsrbench";
  benchmark_task: string | null;
};

type SeriesChannel = {
  index: number;
  length: number;
  values: number[];
  stats: null | {
    min: number;
    max: number;
    mean: number;
    std: number;
    first: number;
    last: number;
  };
};

type BenchmarkMetadata = {
  task: string;
  major: string;
  domain: string | null;
  category: string | null;
  choices: string[] | Record<string, unknown> | null;
  series_names: string[];
  extra: Record<string, unknown>;
};

type TemplateSummary = {
  members: number;
  raw_prompts: number;
  answer_classes: number;
  first_index: number;
  last_index: number;
  answers: { value: string; count: number }[];
};

type RecordPayload = {
  dataset: string;
  dataset_total: number;
  index: number;
  schema: "chatts" | "tsrbench";
  input: string;
  output: string;
  series: SeriesChannel[];
  series_count: number;
  series_names: string[];
  choices: string[] | Record<string, unknown> | null;
  benchmark: BenchmarkMetadata | null;
  audit: Record<string, unknown> | null;
  annotation: Record<string, unknown> | null;
  taxonomy_template_id: string;
  quality_template_id: string;
  answer_class: string;
  issues: string[];
  normalized_template: string;
  template: TemplateSummary;
};

type Member = {
  index: number;
  input: string;
  output: string;
  answer_class: string;
  issues: string[];
  ability_label: string | null;
  quality: string | null;
  difficulty: string | null;
};

type MembersPayload = {
  total: number;
  offset: number;
  limit: number;
  members: Member[];
};

type Translation = Record<string, string>;

type TsrbenchStatus = {
  configured: boolean;
  found: boolean;
  root: string | null;
  checked_paths: string[];
  tasks_found: number;
  tasks_expected: number;
};

const CHANNEL_COLORS = [
  "#1f7a69",
  "#d97745",
  "#6857a8",
  "#b4475a",
  "#3d79b8",
  "#98812d",
  "#2e8b57",
  "#b268a2",
  "#47616f",
  "#a65f35",
  "#4f7f49",
  "#735f9c",
  "#ab4651",
  "#2e7680",
  "#8a6f32",
  "#536e9a",
];

const ISSUE_LABELS: Record<string, { title: string; description: string }> = {
  answer_leakage: {
    title: "答案泄漏",
    description: "问题文本中包含了最终答案约束，模型可能绕过时间序列。",
  },
  visual_grounding_mismatch: {
    title: "视觉措辞错位",
    description: "回答提到图、坐标轴或可视化，但模型输入只有数值序列。",
  },
  unverified_reasoning: {
    title: "推理未验证",
    description: "来源只验证最终标签，没有验证思维链是否由信号支持。",
  },
};

const QUALITY_LABELS: Record<string, string> = {
  unusable: "不可用",
  weak: "较差",
  acceptable: "可接受",
  good: "良好",
  excellent: "优秀",
};

const DIFFICULTY_LABELS: Record<string, string> = {
  very_easy: "很简单",
  easy: "简单",
  moderate: "中等",
  hard: "困难",
  very_hard: "很困难",
};

function formatNumber(value: number | undefined | null): string {
  return new Intl.NumberFormat("zh-CN").format(value ?? 0);
}

function shortText(value: string, max = 80): string {
  const clean = value.replace(/\s+/g, " ").trim();
  return clean.length > max ? `${clean.slice(0, max)}…` : clean;
}

function asString(value: unknown): string | null {
  if (value === null || value === undefined || value === "") return null;
  return String(value);
}

function nested(object: Record<string, unknown> | null, path: string): unknown {
  let current: unknown = object;
  for (const key of path.split(".")) {
    if (!current || typeof current !== "object" || !(key in current)) return null;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

function useApiBase() {
  const [apiBase, setApiBase] = useState(() => {
    if (typeof window === "undefined") return "http://localhost:8765";
    const saved = window.localStorage.getItem("tsqa-lens-api-base");
    const inferred = `${window.location.protocol}//${window.location.hostname}:8765`;
    return saved || inferred;
  });
  const save = useCallback((value: string) => {
    const normalized = value.trim().replace(/\/$/, "");
    window.localStorage.setItem("tsqa-lens-api-base", normalized);
    setApiBase(normalized);
  }, []);
  return { apiBase, saveApiBase: save };
}

async function apiJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.message || payload.error || `请求失败：${response.status}`);
  }
  return payload as T;
}

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`status-dot ${ok ? "status-dot--ok" : "status-dot--off"}`} aria-hidden="true" />;
}

function Pill({ children, tone = "neutral" }: { children: React.ReactNode; tone?: string }) {
  return <span className={`pill pill--${tone}`}>{children}</span>;
}

function SeriesChart({ series, names = [] }: { series: SeriesChannel[]; names?: string[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [visible, setVisible] = useState<number[]>(() =>
    series.slice(0, Math.min(6, series.length)).map((channel) => channel.index),
  );
  const [normalized, setNormalized] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const host = canvas.parentElement;
    if (!host) return;

    const draw = () => {
      const rect = host.getBoundingClientRect();
      const width = Math.max(320, rect.width);
      const height = Math.max(260, Math.min(440, width * 0.42));
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, width, height);

      const left = 52;
      const right = 18;
      const top = 18;
      const bottom = 34;
      const plotW = width - left - right;
      const plotH = height - top - bottom;
      const selected = series.filter((channel) => visible.includes(channel.index) && channel.values.length);

      ctx.strokeStyle = "rgba(35, 47, 43, 0.12)";
      ctx.lineWidth = 1;
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = "#7a827d";
      for (let i = 0; i <= 4; i += 1) {
        const y = top + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(width - right, y);
        ctx.stroke();
      }

      if (!selected.length) {
        ctx.fillStyle = "#6f7772";
        ctx.font = "14px system-ui, sans-serif";
        ctx.fillText("选择至少一个通道以显示曲线", left, top + 30);
        return;
      }

      let globalMin = Infinity;
      let globalMax = -Infinity;
      if (!normalized) {
        selected.forEach((channel) => {
          channel.values.forEach((value) => {
            globalMin = Math.min(globalMin, value);
            globalMax = Math.max(globalMax, value);
          });
        });
      } else {
        globalMin = -3;
        globalMax = 3;
      }
      if (globalMin === globalMax) globalMax = globalMin + 1;

      selected.forEach((channel) => {
        const values = normalized
          ? channel.values.map((value) => {
              const mean = channel.stats?.mean ?? 0;
              const std = channel.stats?.std || 1;
              return Math.max(-3, Math.min(3, (value - mean) / std));
            })
          : channel.values;
        ctx.beginPath();
        values.forEach((value, index) => {
          const x = left + (index / Math.max(1, values.length - 1)) * plotW;
          const y = top + (1 - (value - globalMin) / (globalMax - globalMin)) * plotH;
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = CHANNEL_COLORS[channel.index % CHANNEL_COLORS.length];
        ctx.lineWidth = selected.length > 6 ? 1.15 : 1.65;
        ctx.globalAlpha = selected.length > 8 ? 0.72 : 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1;
      });

      ctx.fillStyle = "#68716c";
      ctx.textAlign = "right";
      ctx.fillText(globalMax.toFixed(normalized ? 1 : 2), left - 8, top + 4);
      ctx.fillText(globalMin.toFixed(normalized ? 1 : 2), left - 8, top + plotH + 4);
      ctx.textAlign = "left";
      ctx.fillText("0", left, height - 10);
      const maxLength = Math.max(...selected.map((channel) => channel.length));
      ctx.textAlign = "right";
      ctx.fillText(formatNumber(maxLength - 1), width - right, height - 10);
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(host);
    return () => observer.disconnect();
  }, [series, visible, normalized]);

  if (!series.length) {
    return <div className="empty-state">这条记录没有可绘制的数值通道。</div>;
  }

  const allVisible = visible.length === series.length;
  return (
    <div className="chart-shell">
      <div className="chart-toolbar">
        <div className="channel-list" aria-label="时间序列通道">
          {series.map((channel) => (
            <button
              type="button"
              key={channel.index}
              className={`channel-chip ${visible.includes(channel.index) ? "channel-chip--active" : ""}`}
              onClick={() =>
                setVisible((current) =>
                  current.includes(channel.index)
                    ? current.filter((value) => value !== channel.index)
                    : [...current, channel.index],
                )
              }
            >
              <span style={{ background: CHANNEL_COLORS[channel.index % CHANNEL_COLORS.length] }} />
              {names[channel.index] || `通道 ${channel.index + 1}`}
              <small>{formatNumber(channel.length)}</small>
            </button>
          ))}
        </div>
        <div className="chart-actions">
          <button type="button" className="text-button" onClick={() => setVisible(allVisible ? [] : series.map((c) => c.index))}>
            {allVisible ? "隐藏全部" : "显示全部"}
          </button>
          <label className="switch-label">
            <input type="checkbox" checked={normalized} onChange={(event) => setNormalized(event.target.checked)} />
            标准化叠加
          </label>
        </div>
      </div>
      <canvas ref={canvasRef} aria-label="时间序列折线图" />
      <div className="series-stats-grid">
        {series.slice(0, 12).map((channel) => (
          <div className="series-stat" key={channel.index}>
            <span className="series-stat__dot" style={{ background: CHANNEL_COLORS[channel.index % CHANNEL_COLORS.length] }} />
            <div>
              <strong>{names[channel.index] || `通道 ${channel.index + 1}`}</strong>
              <span>
                μ {channel.stats?.mean.toFixed(3) ?? "—"} · σ {channel.stats?.std.toFixed(3) ?? "—"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TextPanel({
  eyebrow,
  title,
  original,
  translated,
  translating,
  onTranslate,
  accent,
}: {
  eyebrow: string;
  title: string;
  original: string;
  translated?: string;
  translating: boolean;
  onTranslate: () => void;
  accent: "question" | "answer";
}) {
  const [expanded, setExpanded] = useState(false);
  const long = original.length > 2400;
  return (
    <article className={`text-panel text-panel--${accent}`}>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
        <button type="button" className="translate-button" onClick={onTranslate} disabled={translating}>
          <span aria-hidden="true">文</span>
          {translating ? "翻译中…" : translated ? "重新翻译" : "翻译成中文"}
        </button>
      </div>
      <div className={`document-text ${!expanded && long ? "document-text--clamped" : ""}`}>{original || "（空）"}</div>
      {long && (
        <button type="button" className="expand-button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? "收起原文" : `展开全部 ${formatNumber(original.length)} 字符`}
        </button>
      )}
      {translated && (
        <div className="translation-block">
          <div className="translation-label"><span>中</span> Qwen 译文</div>
          <div className="document-text document-text--translated">{translated}</div>
        </div>
      )}
    </article>
  );
}

function MetadataGrid({ record }: { record: RecordPayload }) {
  const annotation = record.annotation;
  const audit = record.audit;
  const items = [
    ["数据格式", record.schema === "tsrbench" ? "TSRBench官方评测" : "ChatTS训练格式"],
    ["TSRBench任务", record.benchmark?.task],
    ["能力大类", record.benchmark?.major],
    ["能力维度", asString(nested(annotation, "ability_label")) || asString(nested(audit, "primary_label"))],
    ["质量", QUALITY_LABELS[asString(nested(annotation, "quality")) || ""] || asString(nested(annotation, "quality"))],
    ["难度", DIFFICULTY_LABELS[asString(nested(annotation, "difficulty")) || ""] || asString(nested(annotation, "difficulty")) || asString(nested(audit, "difficulty"))],
    ["任务", asString(nested(audit, "task")) || asString(nested(audit, "reasoning_subtype"))],
    ["题型", asString(nested(audit, "question_type"))],
    ["领域", record.benchmark?.domain || asString(nested(audit, "domain")) || asString(nested(audit, "dataset_name"))],
    ["答案来源", asString(nested(audit, "answer_source"))],
    ["验证状态", asString(nested(audit, "verifier.status"))],
    ["推理验证", asString(nested(audit, "verifier.reasoning_status"))],
    ["训练角色", asString(nested(audit, "training_role"))],
  ].filter((item) => item[1]);

  return (
    <div className="metadata-grid">
      {items.length ? items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      )) : <div className="metadata-empty">当前数据集没有audit或合并标签；原始QA仍可正常浏览。</div>}
    </div>
  );
}

function choiceEntries(choices: RecordPayload["choices"]): ReadonlyArray<readonly [string, string]> {
  if (!choices) return [];
  return Array.isArray(choices)
    ? choices.map((value, index) => ["ABCDEFG"[index] || String(index + 1), String(value)] as const)
    : Object.entries(choices).map(([key, value]) => [key, String(value)] as const);
}

function ChoicePanel({ choices, answer, translations }: { choices: RecordPayload["choices"]; answer: string; translations: Translation }) {
  if (!choices || (Array.isArray(choices) && !choices.length)) return null;
  const entries = choiceEntries(choices);
  const answerLetter = answer.match(/[A-G]/i)?.[0]?.toUpperCase();
  return (
    <section className="choice-panel">
      <div className="panel-heading"><div><span className="eyebrow">MULTIPLE CHOICE</span><h2>候选选项</h2></div>{answerLetter && <Pill tone="green">金标 {answerLetter}</Pill>}</div>
      <div className="choice-grid">
        {entries.map(([label, value]) => (
          <div className={label.toUpperCase() === answerLetter ? "choice-row choice-row--answer" : "choice-row"} key={label}>
            <strong>{label}</strong><div><span>{value}</span>{translations[`choice_${label}`] && <small>{translations[`choice_${label}`]}</small>}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function Home() {
  const { apiBase, saveApiBase } = useApiBase();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetName, setDatasetName] = useState("opentslm_tsqa");
  const [record, setRecord] = useState<RecordPayload | null>(null);
  const [members, setMembers] = useState<MembersPayload | null>(null);
  const [memberPage, setMemberPage] = useState(0);
  const [translations, setTranslations] = useState<Translation>({});
  const [translationBusy, setTranslationBusy] = useState<"input" | "output" | "both" | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [familyFilter, setFamilyFilter] = useState<"all" | "opentslm" | "tsrbench" | "other">("opentslm");
  const [jumpValue, setJumpValue] = useState("1");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [apiDraft, setApiDraft] = useState("http://localhost:8765");
  const [serverOk, setServerOk] = useState(false);
  const [qwenModel, setQwenModel] = useState("");
  const [tsrbenchStatus, setTsrbenchStatus] = useState<TsrbenchStatus | null>(null);
  const [activeTab, setActiveTab] = useState<"record" | "template" | "raw">("record");

  const loadDatasets = useCallback(async () => {
    try {
      const payload = await apiJson<{ datasets: Dataset[]; qwen: { model: string }; tsrbench: TsrbenchStatus }>(`${apiBase}/api/datasets`);
      setDatasets(payload.datasets);
      setQwenModel(payload.qwen.model || "");
      setTsrbenchStatus(payload.tsrbench);
      setServerOk(true);
      setError(null);
      setDatasetName((current) => payload.datasets.some((item) => item.name === current)
        ? current
        : payload.datasets[0]?.name || current);
    } catch (exception) {
      setServerOk(false);
      setError(exception instanceof Error ? exception.message : String(exception));
    }
  }, [apiBase]);

  const loadRecord = useCallback(async (dataset: string, index: number) => {
    setLoading(true);
    setError(null);
    try {
      const payload = await apiJson<RecordPayload>(
        `${apiBase}/api/record?dataset=${encodeURIComponent(dataset)}&index=${Math.max(0, index)}`,
      );
      setRecord(payload);
      setJumpValue(String(payload.index + 1));
      setTranslations({});
      setMemberPage(0);
      setServerOk(true);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadDatasets(), 0);
    return () => window.clearTimeout(timer);
  }, [loadDatasets]);

  useEffect(() => {
    if (datasets.some((item) => item.name === datasetName)) {
      const timer = window.setTimeout(() => void loadRecord(datasetName, 0), 0);
      return () => window.clearTimeout(timer);
    }
  }, [datasetName, datasets, loadRecord]);

  useEffect(() => {
    if (!record || activeTab !== "template") return;
    apiJson<MembersPayload>(
      `${apiBase}/api/template-members?dataset=${encodeURIComponent(record.dataset)}&template_id=${record.taxonomy_template_id}&offset=${memberPage * 12}&limit=12`,
    ).then(setMembers).catch((exception) => setError(exception instanceof Error ? exception.message : String(exception)));
  }, [record, activeTab, memberPage, apiBase]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!record || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if (event.key === "ArrowLeft") loadRecord(record.dataset, Math.max(0, record.index - 1));
      if (event.key === "ArrowRight") loadRecord(record.dataset, Math.min(record.dataset_total - 1, record.index + 1));
      if (event.key.toLowerCase() === "r") void randomRecord();
      if (event.key.toLowerCase() === "t") setActiveTab("template");
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const selectedDataset = datasets.find((item) => item.name === datasetName);
  const filteredDatasets = useMemo(() => datasets.filter((dataset) => {
    const familyMatch = familyFilter === "all"
      || (familyFilter === "opentslm" && dataset.name.startsWith("opentslm_"))
      || (familyFilter === "tsrbench" && dataset.schema === "tsrbench")
      || (familyFilter === "other" && !dataset.name.startsWith("opentslm_") && dataset.schema !== "tsrbench");
    return familyMatch && dataset.name.toLowerCase().includes(search.toLowerCase());
  }), [datasets, familyFilter, search]);

  async function randomRecord(issue?: string) {
    try {
      const suffix = issue ? `&issue=${encodeURIComponent(issue)}` : "";
      const payload = await apiJson<{ index: number }>(`${apiBase}/api/random?dataset=${encodeURIComponent(datasetName)}${suffix}`);
      await loadRecord(datasetName, payload.index);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    }
  }

  async function randomTemplateMember() {
    if (!record) return;
    try {
      const payload = await apiJson<{ index: number }>(
        `${apiBase}/api/random?dataset=${encodeURIComponent(record.dataset)}&template_id=${record.taxonomy_template_id}`,
      );
      await loadRecord(record.dataset, payload.index);
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    }
  }

  async function translate(which: "input" | "output" | "both") {
    if (!record) return;
    setTranslationBusy(which);
    setError(null);
    const texts: Record<string, string> = {};
    if (which === "input" || which === "both") {
      texts.input = record.input;
      choiceEntries(record.choices).forEach(([label, value]) => { texts[`choice_${label}`] = value; });
    }
    if (which === "output" || which === "both") texts.output = record.output;
    try {
      const payload = await apiJson<{ translations: Translation; cached: boolean }>(`${apiBase}/api/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts }),
      });
      setTranslations((current) => ({ ...current, ...payload.translations }));
    } catch (exception) {
      setError(exception instanceof Error ? exception.message : String(exception));
    } finally {
      setTranslationBusy(null);
    }
  }

  function submitJump(event: React.FormEvent) {
    event.preventDefault();
    if (!record) return;
    const index = Math.min(record.dataset_total, Math.max(1, Number.parseInt(jumpValue, 10) || 1)) - 1;
    void loadRecord(record.dataset, index);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><span>TS</span></div>
          <div><strong>TSQA Lens</strong><span>数据集审查台</span></div>
        </div>

        <div className="sidebar-section">
          <label className="search-box">
            <span aria-hidden="true">⌕</span>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="查找数据集" aria-label="查找数据集" />
          </label>
          <div className="segment-control" role="group" aria-label="数据集范围">
            <button className={familyFilter === "opentslm" ? "active" : ""} onClick={() => setFamilyFilter("opentslm")}>OpenTSLM</button>
            <button className={familyFilter === "tsrbench" ? "active" : ""} onClick={() => setFamilyFilter("tsrbench")}>TSRBench</button>
            <button className={familyFilter === "all" ? "active" : ""} onClick={() => setFamilyFilter("all")}>全部</button>
            <button className={familyFilter === "other" ? "active" : ""} onClick={() => setFamilyFilter("other")}>其他</button>
          </div>
        </div>

        <nav className="dataset-nav" aria-label="数据集">
          <div className="dataset-nav__title"><span>数据集</span><small>{filteredDatasets.length}</small></div>
          {familyFilter === "tsrbench" && filteredDatasets.length === 0 && tsrbenchStatus && (
            <div className="dataset-empty">
              <strong>{tsrbenchStatus.found ? "没有发现标准任务文件" : "TSRBench路径未找到"}</strong>
              <p>{tsrbenchStatus.found
                ? `已找到目录，但12个任务文件均未匹配：${tsrbenchStatus.root}`
                : "请设置 inspector_config.yaml 的 tsrbench_root，或在启动前设置 TSRBENCH_ROOT。"}</p>
              {!tsrbenchStatus.found && tsrbenchStatus.checked_paths[0] && <code>{tsrbenchStatus.checked_paths[0]}</code>}
            </div>
          )}
          {filteredDatasets.map((dataset) => (
            <button
              type="button"
              key={dataset.name}
              className={`dataset-item ${dataset.name === datasetName ? "dataset-item--active" : ""}`}
              onClick={() => setDatasetName(dataset.name)}
            >
              <div className="dataset-item__top">
                <span className="dataset-name">{dataset.name.replace("opentslm_", "").replace("tsrbench_", "")}</span>
                {dataset.name.startsWith("opentslm_") && <span className="source-tag">OpenTSLM</span>}
                {dataset.schema === "tsrbench" && <span className="source-tag source-tag--benchmark">TSRBench</span>}
              </div>
              <div className="dataset-item__meta">
                <span>{formatNumber(dataset.rows)} 条</span>
                <span>{formatNumber(dataset.templates)} 模板</span>
                <span>{dataset.compression_ratio ? `${dataset.compression_ratio.toFixed(dataset.compression_ratio > 100 ? 0 : 1)}×` : "—"}</span>
              </div>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button type="button" className="server-card" onClick={() => { setApiDraft(apiBase); setSettingsOpen(true); }}>
            <StatusDot ok={serverOk} />
            <div><strong>{serverOk ? "本地服务已连接" : "等待本地服务"}</strong><span>{qwenModel ? shortText(qwenModel.split("/").pop() || qwenModel, 28) : "配置Qwen翻译"}</span></div>
            <span aria-hidden="true">›</span>
          </button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="dataset-heading">
            <div className="breadcrumb"><span>数据审查</span><b>/</b><span>{selectedDataset?.family || "—"}</span></div>
            <h1>{datasetName}</h1>
            <div className="dataset-summary">
              <Pill tone={selectedDataset?.split === "dev" ? "purple" : "green"}>{selectedDataset?.split || "—"}</Pill>
              <span>{selectedDataset?.training_role || "—"}</span>
              <span>·</span>
              <span>{formatNumber(selectedDataset?.rows)} 条QA</span>
              <span>·</span>
              <span>{formatNumber(selectedDataset?.templates)} 个问题模板</span>
            </div>
          </div>

          <div className="record-nav">
            <form onSubmit={submitJump} className="jump-form">
              <span>第</span>
              <input value={jumpValue} onChange={(event) => setJumpValue(event.target.value)} inputMode="numeric" aria-label="记录序号" />
              <span>/ {formatNumber(record?.dataset_total || selectedDataset?.rows)}</span>
            </form>
            <button type="button" className="square-button" aria-label="上一条" disabled={!record || record.index === 0} onClick={() => record && loadRecord(record.dataset, record.index - 1)}>←</button>
            <button type="button" className="square-button" aria-label="下一条" disabled={!record || record.index >= record.dataset_total - 1} onClick={() => record && loadRecord(record.dataset, record.index + 1)}>→</button>
            <button type="button" className="primary-button" onClick={() => randomRecord()}><span aria-hidden="true">↝</span> 随机抽样</button>
          </div>
        </header>

        <div className="tabbar">
          <button className={activeTab === "record" ? "active" : ""} onClick={() => setActiveTab("record")}>记录详情</button>
          <button className={activeTab === "template" ? "active" : ""} onClick={() => setActiveTab("template")}>同模板成员 <span>{formatNumber(record?.template.members)}</span></button>
          <button className={activeTab === "raw" ? "active" : ""} onClick={() => setActiveTab("raw")}>完整元数据</button>
          <div className="tabbar-spacer" />
          {record?.issues.map((issue) => (
            <button type="button" className="issue-chip" key={issue} title={ISSUE_LABELS[issue]?.description} onClick={() => randomRecord(issue)}>
              <span aria-hidden="true">!</span> {ISSUE_LABELS[issue]?.title || issue}
            </button>
          ))}
          <button type="button" className="translate-all" onClick={() => translate("both")} disabled={!record || translationBusy !== null}>
            {translationBusy === "both" ? "Qwen翻译中…" : "中译问题与答案"}
          </button>
        </div>

        {error && (
          <div className="error-banner" role="alert">
            <strong>暂时无法完成</strong><span>{error}</span>
            <button type="button" onClick={() => setError(null)}>×</button>
          </div>
        )}

        <div className="content-scroll">
          {loading && (
            <div className="loading-panel">
              <div className="loading-orbit"><span /><span /><span /></div>
              <strong>{selectedDataset?.index_ready ? "正在读取记录" : "首次打开，正在建立快速索引"}</strong>
              <p>大型JSONL只索引一次，之后可立即跳转和随机抽样。</p>
            </div>
          )}

          {!loading && record && activeTab === "record" && (
            <div className="record-layout">
              <div className="record-main">
                {record.benchmark && (
                  <div className="benchmark-banner">
                    <div><span>官方评测集</span><strong>{record.benchmark.major} / {record.benchmark.task}</strong></div>
                    <div><span>领域</span><strong>{record.benchmark.domain || "未提供"}</strong></div>
                    <div><span>子类型</span><strong>{record.benchmark.category || "—"}</strong></div>
                  </div>
                )}
                <TextPanel
                  eyebrow="USER / INPUT"
                  title="问题与指令"
                  original={record.input}
                  translated={translations.input}
                  translating={translationBusy === "input" || translationBusy === "both"}
                  onTranslate={() => translate("input")}
                  accent="question"
                />
                <ChoicePanel choices={record.choices} answer={record.output} translations={translations} />
                <TextPanel
                  eyebrow="ASSISTANT / OUTPUT"
                  title="参考答案"
                  original={record.output}
                  translated={translations.output}
                  translating={translationBusy === "output" || translationBusy === "both"}
                  onTranslate={() => translate("output")}
                  accent="answer"
                />
              </div>

              <aside className="inspection-rail">
                <section className="rail-card">
                  <div className="rail-card__heading"><span>样本标签</span><small>#{record.index + 1}</small></div>
                  <MetadataGrid record={record} />
                </section>

                <section className="rail-card">
                  <div className="rail-card__heading"><span>审查提示</span><small>{record.issues.length || "—"}</small></div>
                  {record.issues.length ? (
                    <div className="issue-list">
                      {record.issues.map((issue) => (
                        <div className="issue-row" key={issue}>
                          <span aria-hidden="true">!</span>
                          <div><strong>{ISSUE_LABELS[issue]?.title || issue}</strong><p>{ISSUE_LABELS[issue]?.description || "需要人工复核。"}</p></div>
                        </div>
                      ))}
                    </div>
                  ) : <div className="clean-check"><span>✓</span><p>未命中当前自动风险规则。仍建议结合信号人工抽查。</p></div>}
                </section>

                <section className="rail-card template-quick-card">
                  <div className="rail-card__heading"><span>模板簇</span><small>{shortText(record.taxonomy_template_id, 10)}</small></div>
                  <div className="metric-triplet">
                    <div><strong>{formatNumber(record.template.members)}</strong><span>成员</span></div>
                    <div><strong>{formatNumber(record.template.raw_prompts)}</strong><span>原始问法</span></div>
                    <div><strong>{formatNumber(record.template.answer_classes)}</strong><span>答案类</span></div>
                  </div>
                  <div className="answer-distribution">
                    {record.template.answers.slice(0, 5).map((answer) => (
                      <div key={answer.value}>
                        <span title={answer.value}>{shortText(answer.value || "空答案", 24)}</span>
                        <b>{formatNumber(answer.count)}</b>
                      </div>
                    ))}
                  </div>
                  <div className="rail-actions">
                    <button type="button" onClick={() => setActiveTab("template")}>查看全部成员</button>
                    <button type="button" onClick={randomTemplateMember}>簇内随机一条</button>
                  </div>
                </section>
              </aside>

              <section className="series-section">
                <div className="section-heading">
                  <div><span className="eyebrow">NUMERIC MODALITY</span><h2>时间序列</h2></div>
                  <div className="section-meta"><span>{record.series_count} 通道</span><span>最长 {formatNumber(Math.max(0, ...record.series.map((item) => item.length)))} 点</span><span>长序列自动降采样显示</span></div>
                </div>
                <SeriesChart key={`${record.dataset}:${record.index}`} series={record.series} names={record.series_names} />
              </section>

              <section className="template-section">
                <div className="section-heading">
                  <div><span className="eyebrow">NORMALIZED PROMPT</span><h2>能力标注模板</h2></div>
                  <code>{record.taxonomy_template_id}</code>
                </div>
                <pre className="template-code">{record.normalized_template}</pre>
                {record.template.raw_prompts > 1 && (
                  <div className="template-warning"><span>注意</span> 这个模板包含 {formatNumber(record.template.raw_prompts)} 种原始问法，可能是数值归一化过宽；需要在同模板页核对窗口、预测长度或类别数字。</div>
                )}
              </section>
            </div>
          )}

          {!loading && record && activeTab === "template" && (
            <section className="members-view">
              <div className="members-hero">
                <div>
                  <span className="eyebrow">TEMPLATE CLUSTER</span>
                  <h2>这个“模板”内部是否真的相同？</h2>
                  <p>比较不同成员的原始问题、答案类别和风险标记。数值窗口、预测长度或睡眠阶段被合并时，原始问法数会大于1。</p>
                </div>
                <div className="members-hero__metrics">
                  <div><strong>{formatNumber(record.template.members)}</strong><span>QA成员</span></div>
                  <div><strong>{formatNumber(record.template.raw_prompts)}</strong><span>原始问题变体</span></div>
                  <div><strong>{formatNumber(record.template.answer_classes)}</strong><span>答案类别</span></div>
                </div>
              </div>
              <div className="members-controls">
                <button type="button" className="primary-button" onClick={randomTemplateMember}>↝ 随机查看簇内样本</button>
                <span>模板ID <code>{record.taxonomy_template_id}</code></span>
              </div>
              <div className="member-grid">
                {members?.members.map((member) => (
                  <button type="button" className={`member-card ${member.index === record.index ? "member-card--current" : ""}`} key={member.index} onClick={() => loadRecord(record.dataset, member.index)}>
                    <div className="member-card__head">
                      <span>#{member.index + 1}</span>
                      <div>{member.ability_label && <Pill tone="green">{member.ability_label}</Pill>}{member.quality && <Pill>{QUALITY_LABELS[member.quality] || member.quality}</Pill>}</div>
                    </div>
                    <p>{shortText(member.input, 210)}</p>
                    <div className="member-answer"><span>答案</span><strong>{shortText(member.answer_class, 55)}</strong></div>
                    {member.issues.length > 0 && <div className="member-issues">{member.issues.map((issue) => <span key={issue}>! {ISSUE_LABELS[issue]?.title || issue}</span>)}</div>}
                  </button>
                ))}
              </div>
              {members && members.total > members.limit && (
                <div className="pagination">
                  <button disabled={memberPage === 0} onClick={() => setMemberPage((page) => Math.max(0, page - 1))}>← 上一页</button>
                  <span>第 {memberPage + 1} / {Math.ceil(members.total / members.limit)} 页</span>
                  <button disabled={(memberPage + 1) * members.limit >= members.total} onClick={() => setMemberPage((page) => page + 1)}>下一页 →</button>
                </div>
              )}
            </section>
          )}

          {!loading && record && activeTab === "raw" && (
            <section className="raw-view">
              <div className="raw-view__intro">
                <span className="eyebrow">AUDIT TRAIL</span>
                <h2>完整元数据与标签</h2>
                <p>用于核对原始转换审计、质量难度标签与模板连接结果。这里不重复显示完整时间序列数组。</p>
              </div>
              <div className="raw-grid">
                <article><div className="raw-title"><span>合并标签</span><code>annotation</code></div><pre>{JSON.stringify(record.annotation, null, 2) || "null"}</pre></article>
                <article><div className="raw-title"><span>{record.benchmark ? "TSRBench元数据" : "来源审计"}</span><code>{record.benchmark ? "benchmark" : "audit"}</code></div><pre>{JSON.stringify(record.benchmark || record.audit, null, 2) || "null"}</pre></article>
                <article className="raw-grid__wide"><div className="raw-title"><span>派生字段</span><code>inspector</code></div><pre>{JSON.stringify({ taxonomy_template_id: record.taxonomy_template_id, quality_template_id: record.quality_template_id, answer_class: record.answer_class, issues: record.issues, series: record.series.map((channel) => ({ index: channel.index, length: channel.length, stats: channel.stats })) }, null, 2)}</pre></article>
              </div>
            </section>
          )}
        </div>
      </section>

      {settingsOpen && (
        <div className="modal-backdrop">
          <section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
            <div className="settings-modal__head"><div><span className="eyebrow">CONNECTION</span><h2 id="settings-title">本地数据服务</h2></div><button aria-label="关闭" onClick={() => setSettingsOpen(false)}>×</button></div>
            <p>前端只读取这个地址。Qwen地址、模型名和密钥保存在服务端配置中，不会发送到浏览器。</p>
            <label><span>API地址</span><input value={apiDraft} onChange={(event) => setApiDraft(event.target.value)} placeholder="http://localhost:8765" /></label>
            <div className="settings-status"><StatusDot ok={serverOk} /><span>{serverOk ? "已连接" : "未连接"}</span>{qwenModel && <code>{qwenModel}</code>}</div>
            <div className="settings-modal__actions"><button onClick={() => setSettingsOpen(false)}>取消</button><button className="primary-button" onClick={() => { saveApiBase(apiDraft); setSettingsOpen(false); }}>保存并连接</button></div>
          </section>
        </div>
      )}
    </main>
  );
}
