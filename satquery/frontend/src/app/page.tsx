"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";

// Types matching Backend Phase 1, Phase 10, Phase 12
interface ValidationResult {
  passed: boolean;
  message?: string;
  crs?: string;
  resolution?: number[];
  overlap?: boolean;
  crs_compatible?: boolean;
  ready_for_pairwise_analysis?: boolean;
  band_count?: number;
  width?: number;
  height?: number;
}

interface ExecutionTrace {
  task?: string;
  models_used?: string[];
  validation?: {
    status?: string;
    passed?: boolean;
    checks?: Array<{ check: string; passed: boolean; detail?: string }>;
    error?: string;
  };
  evidence?: Record<string, any>;
  execution_steps?: Array<{ id: string; name: string; status?: string }>;
  timestamp?: string;
  query?: string;
}

interface QueryResponse {
  answer: string;
  status: "success" | "failed" | "error";
  error?: string;
  execution_trace?: ExecutionTrace;
  trace?: ExecutionTrace;
  before_after?: {
    t1?: string;
    t2?: string;
    status?: string;
  };
  outputs?: {
    vqa?: { answer?: string; prediction?: string; confidence?: number; evidence?: Record<string, unknown> };
    change_vqa?: any;
    grounding?: { bbox?: [number, number, number, number] | null; confidence?: number; grounding_source?: string };
    change?: { change_detected?: boolean; change_ratio?: number; change_percentage?: number; changed_pixels?: number };
    fusion?: { confidence?: number; agreement?: string; dominant_source?: string };
    [key: string]: any;
  };
}

// Feature 2/3 shared metric card
function MetricCard({ label, value, suffix, accent }: { label: string; value: any; suffix?: string; accent?: string }) {
  const display = value === null || value === undefined ? "—" : `${value}${suffix || ""}`;
  return (
    <div className="hover-lift" style={{ position: "relative", overflow: "hidden", padding: "16px", borderRadius: "12px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-subtle)" }}>
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: "2px", background: accent || "#60a5fa", opacity: 0.75 }} />
      <div style={{ fontSize: "26px", fontWeight: 700, color: accent || "#60a5fa", fontFamily: "var(--font-geist-mono)" }} className="tnum">{display}</div>
      <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "4px" }}>{label}</div>
    </div>
  );
}

function BenchmarkView({ data, loading, error, onReload }: { data: any; loading: boolean; error: string | null; onReload: () => void }) {
  const m = data?.metrics || {};
  const bd = data?.breakdown || {};
  return (
    <section className="glass-panel animate-fade-in-up" style={{ padding: "24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "16px", marginBottom: "18px", flexWrap: "wrap" }}>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "4px", display: "flex", alignItems: "center", gap: "10px" }}><span className="icon-badge">📊</span> Offline Benchmark Dashboard</h2>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", maxWidth: "640px", lineHeight: 1.5 }}>
            {data?.description || "Bounded offline evaluation over local georeferenced validation fixtures. Every metric is computed from real specialist-model output on real pixels."}
          </p>
        </div>
        <button onClick={onReload} disabled={loading} className="btn-soft" style={{ padding: "8px 16px", borderRadius: "8px", background: "rgba(59,130,246,0.15)", border: "1px solid rgba(59,130,246,0.35)", color: "#60a5fa", fontSize: "13px", fontWeight: 600, cursor: loading ? "wait" : "pointer" }}>
          {loading ? "Running…" : "↻ Re-run"}
        </button>
      </div>

      {error && (
        <div style={{ padding: "12px 16px", borderRadius: "8px", background: "rgba(244,63,94,0.12)", border: "1px solid rgba(244,63,94,0.35)", color: "#fb7185", fontSize: "13px", marginBottom: "16px" }}>{error}</div>
      )}
      {loading && !data && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px" }} aria-busy="true" aria-label="Running bounded fixture evaluation">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: "78px" }} />
          ))}
        </div>
      )}

      {data && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "12px", marginBottom: "22px" }}>
            <MetricCard label="VQA accuracy" value={m.vqa_accuracy} suffix="%" accent="#34d399" />
            <MetricCard label="Captioning coverage" value={m.captioning_score} suffix="%" accent="#93b4f5" />
            <MetricCard label="Change-VQA accuracy" value={m.change_vqa_accuracy} suffix="%" accent="#34d399" />
            <MetricCard label="Grounding detection" value={m.grounding_detection_rate} suffix="%" accent="#60a5fa" />
            <MetricCard label="Composite score" value={m.composite_score} suffix="%" accent="#fbbf24" />
            <MetricCard label="Latency p50" value={m.latency_p50_ms} suffix=" ms" />
            <MetricCard label="Latency p95" value={m.latency_p95_ms} suffix=" ms" />
            <MetricCard label="Samples evaluated" value={m.samples_evaluated} />
          </div>

          <BenchmarkBreakdown bd={bd} />
        </>
      )}
    </section>
  );
}

function OkBadge({ ok }: { ok: boolean }) {
  return <span style={{ color: ok ? "var(--emerald-bright)" : "var(--rose-bright)", fontWeight: 700 }}>{ok ? "✓" : "✗"}</span>;
}

function BenchmarkBreakdown({ bd }: { bd: any }) {
  const th: React.CSSProperties = { textAlign: "left", padding: "6px 10px", color: "var(--text-muted)", fontWeight: 600, fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.04em" };
  const td: React.CSSProperties = { padding: "6px 10px", fontSize: "12px", color: "var(--text-secondary)", borderTop: "1px solid var(--border-subtle)" };
  const title: React.CSSProperties = { fontSize: "13px", fontWeight: 600, color: "var(--text-primary)", margin: "18px 0 8px" };
  return (
    <div>
      <div style={title}>Land-cover VQA</div>
      <table className="data-table" style={{ borderCollapse: "collapse" }}>
        <thead><tr><th style={th}>Fixture</th><th style={th}>Expected</th><th style={th}>Predicted</th><th style={th}>Conf</th><th style={th}>OK</th></tr></thead>
        <tbody>
          {(bd.vqa || []).map((r: any, i: number) => (
            <tr key={i}><td style={td}>{r.fixture}</td><td style={td}>{r.expected}</td><td style={td}>{r.predicted}</td><td style={td}>{r.confidence}</td><td style={td}><OkBadge ok={r.correct} /></td></tr>
          ))}
        </tbody>
      </table>

      <div style={title}>Change-VQA</div>
      <table className="data-table" style={{ borderCollapse: "collapse" }}>
        <thead><tr><th style={th}>T1 → T2</th><th style={th}>Expected change</th><th style={th}>Predicted</th><th style={th}>OK</th></tr></thead>
        <tbody>
          {(bd.change_vqa || []).map((r: any, i: number) => (
            <tr key={i}><td style={td}>{r.t1} → {r.t2}</td><td style={td}>{String(r.expected_change)}</td><td style={td}>{String(r.predicted_change)}</td><td style={td}><OkBadge ok={r.correct} /></td></tr>
          ))}
        </tbody>
      </table>

      <div style={title}>Grounding detection</div>
      <table className="data-table" style={{ borderCollapse: "collapse" }}>
        <thead><tr><th style={th}>Fixture</th><th style={th}>Phrase</th><th style={th}>Detected</th><th style={th}>Conf</th><th style={th}>BBox</th></tr></thead>
        <tbody>
          {(bd.grounding || []).map((r: any, i: number) => (
            <tr key={i}><td style={td}>{r.fixture}</td><td style={td}>{r.phrase}</td><td style={td}><OkBadge ok={r.detected} /></td><td style={td}>{r.confidence}</td><td style={{ ...td, fontFamily: "var(--font-geist-mono)" }}>{r.bbox ? `[${r.bbox.join(", ")}]` : "—"}</td></tr>
          ))}
        </tbody>
      </table>

      <div style={title}>Captioning coverage</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {(bd.captioning || []).map((r: any, i: number) => (
          <div key={i} style={{ padding: "10px 12px", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", marginBottom: "4px" }}>
              <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>{r.fixture} · expects “{r.expected_cover}”</span>
              <OkBadge ok={r.mentions_expected} />
            </div>
            <div style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: 1.5 }}>{r.caption}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RegistryView({ data, loading, error, onReload }: { data: any; loading: boolean; error: string | null; onReload: () => void }) {
  const models = data?.models || [];
  const tools = data?.tools || [];
  const taskTypes = data?.task_types || [];
  const params = data?.parameters || [];
  const stages = data?.pipeline_stages || [];
  const th: React.CSSProperties = { textAlign: "left", padding: "6px 10px", color: "var(--text-muted)", fontWeight: 600, fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.04em" };
  const td: React.CSSProperties = { padding: "8px 10px", fontSize: "12px", color: "var(--text-secondary)", borderTop: "1px solid var(--border-subtle)", verticalAlign: "top" };
  const title: React.CSSProperties = { fontSize: "13px", fontWeight: 600, color: "var(--text-primary)", margin: "22px 0 10px" };
  return (
    <section className="glass-panel animate-fade-in-up" style={{ padding: "24px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "16px", marginBottom: "8px", flexWrap: "wrap" }}>
        <div>
          <h2 style={{ fontSize: "18px", fontWeight: 700, marginBottom: "4px", display: "flex", alignItems: "center", gap: "10px" }}><span className="icon-badge">🧩</span> Agent Registry &amp; Parameters</h2>
          <p style={{ fontSize: "12px", color: "var(--text-secondary)", maxWidth: "640px", lineHeight: 1.5 }}>
            Every model, geospatial tool, supported task type and tunable threshold that governs a run — auditable alongside the execution trace.
          </p>
        </div>
        <button onClick={onReload} disabled={loading} className="btn-soft" style={{ padding: "8px 16px", borderRadius: "8px", background: "rgba(59,130,246,0.15)", border: "1px solid rgba(59,130,246,0.35)", color: "#60a5fa", fontSize: "13px", fontWeight: 600, cursor: loading ? "wait" : "pointer" }}>
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      {error && (
        <div style={{ padding: "12px 16px", borderRadius: "8px", background: "rgba(244,63,94,0.12)", border: "1px solid rgba(244,63,94,0.35)", color: "#fb7185", fontSize: "13px", marginTop: "12px" }}>{error}</div>
      )}
      {loading && !data && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))", gap: "12px", marginTop: "16px" }} aria-busy="true" aria-label="Loading registry">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: "96px" }} />
          ))}
        </div>
      )}

      {data && (
        <>
          {data.router && (
            <div style={{ marginTop: "16px", padding: "14px 16px", borderRadius: "12px", background: "rgba(59,130,246,0.10)", border: "1px solid rgba(59,130,246,0.30)" }}>
              <div style={{ fontSize: "14px", fontWeight: 700, color: "#93b4f5" }}>{data.router.name} · <span style={{ fontSize: "12px", fontWeight: 500, color: "var(--text-secondary)" }}>{data.router.phase} · {data.router.status}</span></div>
              <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>{data.router.description}</div>
            </div>
          )}

          <div style={title}>Pipeline stages</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", alignItems: "center" }}>
            {stages.map((s: any, i: number) => (
              <React.Fragment key={s.stage}>
                <div title={s.role} className="chip" style={{ padding: "6px 12px", borderRadius: "20px", background: "rgba(255,255,255,0.04)", border: "1px solid var(--border-subtle)", fontSize: "12px", color: "var(--text-secondary)", fontFamily: "var(--font-geist-mono)" }}>{s.stage}</div>
                {i < stages.length - 1 && <span style={{ color: "var(--text-muted)" }}>→</span>}
              </React.Fragment>
            ))}
          </div>

          <div style={title}>Models ({models.length})</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "12px" }}>
            {models.map((mo: any) => (
              <div key={mo.id} className="hover-lift" style={{ padding: "14px", borderRadius: "12px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-primary)" }}>{mo.name}</span>
                  <span style={{ fontSize: "11px", padding: "2px 8px", borderRadius: "10px", background: "rgba(59,130,246,0.15)", color: "#60a5fa", fontFamily: "var(--font-geist-mono)" }}>v{mo.version}</span>
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-geist-mono)", marginTop: "2px" }}>{mo.id}</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "4px", marginTop: "8px" }}>
                  {(mo.capabilities || []).map((c: string, i: number) => (
                    <span key={i} style={{ fontSize: "10px", padding: "2px 6px", borderRadius: "6px", background: "rgba(255,255,255,0.05)", color: "var(--text-secondary)" }}>{c}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div style={title}>Geospatial tools ({tools.length})</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "12px" }}>
            {tools.map((tl: any) => (
              <div key={tl.id} className="hover-lift" style={{ padding: "14px", borderRadius: "12px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ fontSize: "13px", fontWeight: 700, color: "var(--text-primary)" }}>{tl.name}</div>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-geist-mono)", marginTop: "2px" }}>{tl.id}</div>
                <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginTop: "8px", lineHeight: 1.5 }}>{tl.description}</div>
              </div>
            ))}
          </div>

          <div style={title}>Supported task types ({taskTypes.length})</div>
          <table className="data-table" style={{ borderCollapse: "collapse" }}>
            <thead><tr><th style={th}>Task</th><th style={th}>Mode</th><th style={th}>Inputs</th><th style={th}>Model</th></tr></thead>
            <tbody>
              {taskTypes.map((t: any) => (
                <tr key={t.id}>
                  <td style={td}><span style={{ color: "var(--text-primary)", fontWeight: 600 }}>{t.label}</span><br /><span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-geist-mono)" }}>{t.id}</span></td>
                  <td style={td}>{t.single_image ? "single-image" : "multi-image"}</td>
                  <td style={{ ...td, fontFamily: "var(--font-geist-mono)" }}>{(t.inputs || []).join(", ")}</td>
                  <td style={{ ...td, fontFamily: "var(--font-geist-mono)" }}>{t.model}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={title}>Tunable parameters ({params.length})</div>
          <table className="data-table" style={{ borderCollapse: "collapse" }}>
            <thead><tr><th style={th}>Parameter</th><th style={th}>Value</th><th style={th}>Stage</th><th style={th}>Description</th></tr></thead>
            <tbody>
              {params.map((p: any, i: number) => (
                <tr key={i}>
                  <td style={{ ...td, fontFamily: "var(--font-geist-mono)", color: "#60a5fa" }}>{p.name}</td>
                  <td style={{ ...td, fontFamily: "var(--font-geist-mono)", color: "var(--text-primary)" }}>{String(p.value)}</td>
                  <td style={td}>{p.stage}</td>
                  <td style={td}>{p.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}

export default function SatQueryApp() {
  // Navigation / Mode state
  const [activeTab, setActiveTab] = useState<"answer" | "beforeAfter" | "changeMask" | "evidence">("answer");
  const [traceTab, setTraceTab] = useState<"models" | "validation" | "evidence" | "steps" | "raw">("models");

  // File Upload State
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [previewUrlA, setPreviewUrlA] = useState<string | null>(null);
  const [previewUrlB, setPreviewUrlB] = useState<string | null>(null);
  
  // Validation state
  const [isValidatingPair, setIsValidatingPair] = useState<boolean>(false);
  const [pairValidation, setPairValidation] = useState<ValidationResult | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Question & Query State
  const [query, setQuery] = useState<string>("Has the built-up area increased between these two satellite scenes?");
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [executionResult, setExecutionResult] = useState<QueryResponse | null>(null);
  const [execError, setExecError] = useState<string | null>(null);
  // Answer-rendering language. Matches the backend contract exactly: "en" | "hi" | "hinglish".
  const [language, setLanguage] = useState<"en" | "hi" | "hinglish">("en");

  // Report Modal / Export State
  const [showReportModal, setShowReportModal] = useState<boolean>(false);
  const [reportFormat, setReportFormat] = useState<"pdf" | "json">("json");
  const [isExporting, setIsExporting] = useState<boolean>(false);

  // Connection State
  const [connectionStatus, setConnectionStatus] = useState<"connected" | "connecting" | "disconnected">("connected");

  // Feature 2 & 3: top-level view switch + benchmark dashboard + registry panel
  const [appView, setAppView] = useState<"workspace" | "benchmark" | "registry">("workspace");
  const [benchmark, setBenchmark] = useState<any | null>(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState<boolean>(false);
  const [benchmarkError, setBenchmarkError] = useState<string | null>(null);
  const [registry, setRegistry] = useState<any | null>(null);
  const [registryLoading, setRegistryLoading] = useState<boolean>(false);
  const [registryError, setRegistryError] = useState<string | null>(null);

  const fileInputRefA = useRef<HTMLInputElement>(null);
  const fileInputRefB = useRef<HTMLInputElement>(null);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Determine if upload is Benchmark Mode (PNG / JPEG) vs Georeferenced (GeoTIFF / TIFF)
  const isBenchmarkFile = (file: File | null): boolean => {
    if (!file) return false;
    const ext = file.name.toLowerCase();
    return ext.endsWith(".png") || ext.endsWith(".jpg") || ext.endsWith(".jpeg");
  };

  const isGeoTiffFile = (file: File | null): boolean => {
    if (!file) return false;
    const ext = file.name.toLowerCase();
    return ext.endsWith(".tif") || ext.endsWith(".tiff");
  };

  const isBenchmarkMode = isBenchmarkFile(fileA) || isBenchmarkFile(fileB);

  // Handle preview generation
  const handleFileChange = (file: File | null, slot: "A" | "B") => {
    if (slot === "A") {
      setFileA(file);
      if (file) {
        setPreviewUrlA(URL.createObjectURL(file));
      } else {
        setPreviewUrlA(null);
      }
    } else {
      setFileB(file);
      if (file) {
        setPreviewUrlB(URL.createObjectURL(file));
      } else {
        setPreviewUrlB(null);
      }
    }
    setPairValidation(null);
    setValidationError(null);
    setExecutionResult(null);
  };

  // Run Phase 1 check_compatibility on GeoTIFF uploads
  const validateUploadedPair = useCallback(async (fA: File, fB: File) => {
    // If either file is PNG/JPEG benchmark mode, skip Phase 1 georeferencing checks
    if (isBenchmarkFile(fA) || isBenchmarkFile(fB)) {
      setPairValidation({
        passed: true,
        message: "Benchmark Mode Active: public dataset evaluation (VRSBench/RSVQA/CDVQA). Georeferencing checks bypassed.",
        overlap: true,
        crs_compatible: true,
        ready_for_pairwise_analysis: true,
      });
      return;
    }

    setIsValidatingPair(true);
    setValidationError(null);

    try {
      const formData = new FormData();
      formData.append("file_a", fA);
      formData.append("file_b", fB);

      const res = await fetch(`${API_BASE}/validate-pair`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Pairwise validation failed" }));
        throw new Error(errData.detail || `HTTP ${res.status}: Validation rejected`);
      }

      const data = await res.json();
      const compat = data.compatibility || {};

      setPairValidation({
        passed: compat.ready_for_pairwise_analysis ?? false,
        crs_compatible: compat.crs_compatible,
        overlap: compat.spatial_overlap,
        ready_for_pairwise_analysis: compat.ready_for_pairwise_analysis,
        crs: data.meta_a?.crs,
        resolution: data.meta_a?.resolution,
        message: compat.ready_for_pairwise_analysis
          ? "CRS & Spatial Footprint Validated: Scenes aligned for bi-temporal change detection."
          : `Compatibility Warning: ${compat.spatial_overlap ? "" : "No spatial overlap. "}${compat.crs_compatible ? "" : "Incompatible CRS projections."}`,
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Validation connection error";
      setValidationError(msg);
      setPairValidation({
        passed: false,
        message: msg,
      });
    } finally {
      setIsValidatingPair(false);
    }
  }, [API_BASE]);
  // Execute analytical query against /query endpoint
  const handleExecuteQuery = async () => {
    if (!query.trim()) return;

    setIsExecuting(true);
    setExecError(null);
    setExecutionResult(null);
    setActiveTab("answer");

    try {
      const formData = new FormData();
      formData.append("query", query);
      formData.append("language", language);
      if (fileA) formData.append("files", fileA);
      if (fileB) formData.append("files", fileB);

      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Query execution failed" }));
        throw new Error(errData.detail || `HTTP ${res.status}: Execution error`);
      }

      const data = await res.json();
      setExecutionResult(data);
    } catch (err: unknown) {
      setExecError(err instanceof Error ? err.message : "Failed to execute query");
    } finally {
      setIsExecuting(false);
    }
  };

  // Dedicated single-image Scene Captioning via the /caption endpoint
  const handleCaptionScene = async () => {
    if (!fileA) {
      setExecError("Attach a single scene (Image A) to generate a description.");
      return;
    }
    setIsExecuting(true);
    setExecError(null);
    setExecutionResult(null);
    setActiveTab("answer");
    try {
      const formData = new FormData();
      formData.append("image", fileA);
      const res = await fetch(`${API_BASE}/caption`, { method: "POST", body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: "Captioning failed" }));
        throw new Error(errData.detail || `HTTP ${res.status}: Captioning error`);
      }
      const data = await res.json();
      const t = data.trace || {};
      const checks = (t.validation_checks || []).map((v: any) => ({ check: v.check, passed: v.passed, detail: v.detail }));
      const allPassed = checks.every((c: { passed: boolean }) => c.passed);
      const overall = t.confidence_summary?.overall;
      setExecutionResult({
        answer: typeof data.answer === "string" ? data.answer : String(data.answer),
        status: "success",
        execution_trace: {
          task: "captioning",
          models_used: (t.model_selections || []).map((m: any) => (m.version ? `${m.name} (${m.version})` : m.name)),
          validation: { status: allPassed ? "passed" : "failed", passed: allPassed, checks },
          evidence: { sources: t.evidence_sources || [] },
          execution_steps: t.execution_steps || [],
          timestamp: t.timestamp,
          query: t.query,
        },
        outputs: { caption: { confidence: overall } },
      });
    } catch (err: unknown) {
      setExecError(err instanceof Error ? err.message : "Failed to generate scene description");
    } finally {
      setIsExecuting(false);
    }
  };

  // Feature 2: load bounded offline benchmark metrics from the backend
  const loadBenchmark = useCallback(async () => {
    setBenchmarkLoading(true);
    setBenchmarkError(null);
    try {
      const res = await fetch(`${API_BASE}/benchmark`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: benchmark endpoint error`);
      setBenchmark(await res.json());
    } catch (err: unknown) {
      setBenchmarkError(err instanceof Error ? err.message : "Failed to load benchmark");
    } finally {
      setBenchmarkLoading(false);
    }
  }, [API_BASE]);

  // Feature 3: load the agent registry (models, tools, task types, parameters)
  const loadRegistry = useCallback(async () => {
    setRegistryLoading(true);
    setRegistryError(null);
    try {
      const res = await fetch(`${API_BASE}/registry`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: registry endpoint error`);
      setRegistry(await res.json());
    } catch (err: unknown) {
      setRegistryError(err instanceof Error ? err.message : "Failed to load registry");
    } finally {
      setRegistryLoading(false);
    }
  }, [API_BASE]);

  // Lazily fetch data the first time a view is opened
  useEffect(() => {
    if (appView === "benchmark" && !benchmark && !benchmarkLoading) loadBenchmark();
    if (appView === "registry" && !registry && !registryLoading) loadRegistry();
  }, [appView, benchmark, benchmarkLoading, registry, registryLoading, loadBenchmark, loadRegistry]);

  // Pre-flight: run /validate-pair compatibility as soon as BOTH scenes are attached.
  useEffect(() => {
    if (fileA && fileB) {
      validateUploadedPair(fileA, fileB);
    }
  }, [fileA, fileB, validateUploadedPair]);

  // Reflect the selected answer language on the live document (root layout stays "en" for SSR).
  useEffect(() => {
    const htmlLang: Record<typeof language, string> = { en: "en", hi: "hi", hinglish: "hi-Latn" };
    document.documentElement.lang = htmlLang[language] || "en";
  }, [language]);

  // Derive confidence from result
  const getConfidenceInfo = (conf: number | null | undefined) => {
    if (conf === null || conf === undefined || Number.isNaN(conf))
      return { bucket: "Confidence n/a", color: "#94a3b8", bg: "rgba(148, 163, 184, 0.12)", border: "rgba(148, 163, 184, 0.35)", icon: "⚪" };
    if (conf >= 0.85) return { bucket: "High Confidence", color: "#34d399", bg: "rgba(16, 185, 129, 0.12)", border: "rgba(16, 185, 129, 0.35)", icon: "🟢" };
    if (conf >= 0.60) return { bucket: "Medium Confidence", color: "#fbbf24", bg: "rgba(245, 158, 11, 0.12)", border: "rgba(245, 158, 11, 0.35)", icon: "🟡" };
    return { bucket: "Low Confidence", color: "#fb7185", bg: "rgba(244, 63, 94, 0.12)", border: "rgba(244, 63, 94, 0.35)", icon: "🔴" };
  };

  const trace = executionResult?.execution_trace || executionResult?.trace;
  const outputs = executionResult?.outputs || {};
  let currentConfidence: number | null = null;
  if (typeof outputs.vqa?.confidence === "number") currentConfidence = outputs.vqa.confidence;
  else if (typeof outputs.caption?.confidence === "number") currentConfidence = outputs.caption.confidence;
  else if (typeof outputs.grounding?.confidence === "number") currentConfidence = outputs.grounding.confidence;
  else if (typeof outputs.fusion?.confidence === "number") currentConfidence = outputs.fusion.confidence;
  // No fabricated fallback: when the backend reports no confidence, it stays null and renders as "n/a".

  const confInfo = getConfidenceInfo(currentConfidence);

  // Execute is gated on a real uploaded scene: a query cannot run without an image.
  const executeDisabled = isExecuting || !query.trim() || !fileA;

  const handleDownloadReport = async () => {
    if (!executionResult) return;
    setIsExporting(true);
    try {
      const report = {
        query,
        answer: executionResult.answer,
        confidence: getConfidenceInfo(currentConfidence),
        trace: executionResult.execution_trace || executionResult.trace,
        outputs: executionResult.outputs,
      };

      if (reportFormat === "json") {
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(report, null, 2));
        const downloadAnchorNode = document.createElement('a');
        downloadAnchorNode.setAttribute("href", dataStr);
        downloadAnchorNode.setAttribute("download", "satquery_report.json");
        document.body.appendChild(downloadAnchorNode);
        downloadAnchorNode.click();
        downloadAnchorNode.remove();
      } else {
        // Real client-side PDF (jsPDF) built from the actual assembled report data.
        const { jsPDF } = await import("jspdf");
        const doc = new jsPDF({ unit: "pt", format: "a4" });
        const marginX = 48;
        const maxW = doc.internal.pageSize.getWidth() - marginX * 2;
        const pageH = doc.internal.pageSize.getHeight();
        let y = 56;
        const ensure = (h: number) => { if (y + h > pageH - 48) { doc.addPage(); y = 56; } };
        const write = (text: string, size: number, bold: boolean, color: [number, number, number] = [20, 24, 33]) => {
          doc.setFont("helvetica", bold ? "bold" : "normal");
          doc.setFontSize(size);
          doc.setTextColor(color[0], color[1], color[2]);
          for (const line of doc.splitTextToSize(text || "—", maxW) as string[]) {
            ensure(size + 6);
            doc.text(line, marginX, y);
            y += size + 6;
          }
        };
        const gap = (h = 10) => { y += h; };

        write("SatQuery — Earth Observation Analysis Report", 18, true);
        write(new Date().toLocaleString(), 9, false, [120, 128, 140]);
        gap();
        const t: any = report.trace || {};
        write("Query", 12, true, [59, 110, 220]); write(report.query, 11, false); gap();
        write(`Language: ${language}`, 10, false, [90, 98, 110]); gap(4);
        write("Answer", 12, true, [59, 110, 220]); write(String(report.answer ?? "—"), 11, false); gap();
        write("Confidence", 12, true, [59, 110, 220]);
        write(`${report.confidence.bucket}${currentConfidence != null ? ` — ${(currentConfidence * 100).toFixed(1)}%` : ""}`, 11, false);
        gap();
        const models: string[] = t.models_used || (t.model_selections || []).map((m: any) => m.version ? `${m.name} (${m.version})` : m.name);
        if (models?.length) { write("Models Invoked", 12, true, [59, 110, 220]); models.forEach((m) => write(`• ${m}`, 10, false)); gap(); }
        const val = t.validation; if (val) { write("Validation", 12, true, [59, 110, 220]); write(`Status: ${val.status ?? (val.passed ? "passed" : "failed")}`, 10, false); (val.checks || []).forEach((c: any) => write(`• ${c.check}: ${c.passed ? "PASS" : "FAIL"}${c.detail ? ` — ${c.detail}` : ""}`, 9, false)); gap(); }
        const steps: any[] = t.execution_steps || []; if (steps.length) { write("Execution Trace", 12, true, [59, 110, 220]); steps.forEach((s: any, i: number) => write(`${i + 1}. ${typeof s === "string" ? s : (s.step || s.action || JSON.stringify(s))}`, 9, false)); gap(); }
        write("Evidence & Outputs", 12, true, [59, 110, 220]);
        write(JSON.stringify(report.outputs ?? {}, null, 2), 8, false, [70, 78, 90]);
        doc.save("satquery_report.pdf");
      }
    } catch (e) {
      console.error("Export failed", e);
    } finally {
      setIsExporting(false);
      setShowReportModal(false);
    }
  };

  return (
    <div className="app-shell">
      {/* ===== Sidebar navigation ===== */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="sidebar-logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="7.5" />
              <path d="M4.5 12h15" />
              <path d="M12 4.5c3 2.4 3 12.6 0 15" />
              <path d="M12 4.5c-3 2.4-3 12.6 0 15" />
            </svg>
          </span>
          <div style={{ minWidth: 0 }}>
            <span className="gradient-text" style={{ fontSize: "17px", fontWeight: 700, letterSpacing: "-0.01em", display: "block" }}>SatQuery</span>
            <div style={{ fontSize: "11px", color: "var(--text-secondary)" }}>Geospatial Intelligence</div>
          </div>
        </div>

        <div className="sidebar-section-label">Console</div>
        {([
          { id: "workspace", label: "Workspace" },
          { id: "benchmark", label: "Benchmarks" },
          { id: "registry", label: "Registry" },
        ] as const).map((v) => (
          <button
            key={v.id}
            onClick={() => setAppView(v.id)}
            className={`sidebar-item${appView === v.id ? " active" : ""}`}
          >
            <span className="si-icon">
              {v.id === "workspace" ? (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>
              ) : v.id === "benchmark" ? (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/><rect x="7" y="11" width="3" height="6"/><rect x="12" y="7" width="3" height="10"/><rect x="17" y="13" width="3" height="4"/></svg>
              ) : (
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>
              )}
            </span>
            <span>{v.label}</span>
          </button>
        ))}
        <div className="sidebar-spacer" />

        <div className="sidebar-foot">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 10px", borderRadius: "8px", background: connectionStatus === "connected" ? "rgba(34,197,94,0.10)" : connectionStatus === "connecting" ? "rgba(245,158,11,0.10)" : "rgba(244,63,94,0.10)", border: `1px solid ${connectionStatus === "connected" ? "rgba(34,197,94,0.30)" : connectionStatus === "connecting" ? "rgba(245,158,11,0.30)" : "rgba(244,63,94,0.30)"}` }}>
            <span className={connectionStatus === "connected" ? "animate-pulse-emerald" : connectionStatus === "connecting" ? "animate-pulse-cyan" : ""} style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", backgroundColor: connectionStatus === "connected" ? "var(--emerald-bright)" : connectionStatus === "connecting" ? "var(--amber-bright)" : "var(--rose-bright)" }} />
            <span style={{ fontSize: "12px", fontWeight: 600, color: connectionStatus === "connected" ? "#34d399" : connectionStatus === "connecting" ? "#fbbf24" : "#fb7185" }}>
              {connectionStatus === "connected" ? "Backend Connected" : connectionStatus === "connecting" ? "Connecting…" : "Disconnected"}
            </span>
          </div>
          <a href={`${API_BASE}/docs`} target="_blank" rel="noopener noreferrer" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "6px", padding: "8px 10px", borderRadius: "8px", border: "1px solid var(--border-subtle)", background: "rgba(255,255,255,0.02)", color: "var(--text-secondary)", textDecoration: "none", fontSize: "12px", fontWeight: 500 }}>
            <span>Swagger API Docs</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline points="15 3 21 3 21 9" /><line x1="10" y1="14" x2="21" y2="3" /></svg>
          </a>
          <div style={{ fontSize: "10.5px", color: "var(--text-muted)", padding: "0 2px", fontFamily: "var(--font-geist-mono)" }}>v0.1.0 · monorepo build</div>
        </div>
      </aside>

      {/* ===== Main content column ===== */}
      <main style={{ padding: "28px 32px 56px", maxWidth: "1200px", width: "100%" }}>
        <div className="page-head">
          <div>
            <h1>{appView === "workspace" ? "Analysis Workspace" : appView === "benchmark" ? "Benchmark Dashboard" : "Agent Registry"}</h1>
            <div className="page-sub">
              {appView === "workspace"
                ? "Upload georeferenced scenes, query in natural language, and audit every model invocation."
                : appView === "benchmark"
                ? "Bounded offline evaluation over local validation fixtures — real pixels, real metrics."
                : "Every model, tool, task type and tunable threshold that governs an analytical run."}
            </div>
          </div>
        </div>


      {/* Feature 2: Benchmark Dashboard */}
      {appView === "benchmark" && (
        <BenchmarkView data={benchmark} loading={benchmarkLoading} error={benchmarkError} onReload={loadBenchmark} />
      )}

      {/* Feature 3: Registry & Parameters */}
      {appView === "registry" && (
        <RegistryView data={registry} loading={registryLoading} error={registryError} onReload={loadRegistry} />
      )}

      {/* Main Grid: Left (Upload + Question) & Right (Results + Execution Trace) */}
      {appView === "workspace" && (
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))", gap: "24px", alignItems: "start" }}>
        
        {/* LEFT COLUMN: Upload Panel & Question Box */}
        <div className="animate-fade-in-up" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* UPLOAD PANEL */}
          <section className="glass-panel" style={{ padding: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="icon-badge">📂</span>
                <h2 style={{ fontSize: "15px", fontWeight: 600 }}>Raster Scene Upload Panel</h2>
              </div>
              <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                Primary: GeoTIFF (.tif) | Secondary: PNG/JPG (Benchmark)
              </span>
            </div>

            {/* Benchmark Warning Banner when PNG/JPEG detected */}
            {isBenchmarkMode && (
              <div
                style={{
                  padding: "10px 14px",
                  borderRadius: "8px",
                  background: "rgba(245, 158, 11, 0.08)",
                  border: "1px solid rgba(245, 158, 11, 0.25)",
                  color: "var(--amber-bright)",
                  fontSize: "12px",
                  marginBottom: "16px",
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                }}
              >
                <span style={{ fontSize: "16px" }}>ℹ️</span>
                <div>
                  <strong>benchmark mode, no georeferencing</strong> — Evaluates public benchmark datasets (VRSBench / RSVQA / CDVQA). Geospatial CRS/extent compatibility checks are bypassed.
                </div>
              </div>
            )}

            {/* Dual Upload Slots: Scene A (T1 / Optical) & Scene B (T2 / SAR) */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px", marginBottom: "16px" }}>
              
              {/* Scene A */}
              <div
                onClick={() => fileInputRefA.current?.click()}
                className="upload-zone"
                style={{
                  border: "1px dashed rgba(59, 130, 246, 0.3)",
                  borderRadius: "10px",
                  padding: "16px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: fileA ? "rgba(59, 130, 246, 0.05)" : "rgba(10, 13, 19, 0.5)",
                  transition: "all 0.2s ease",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                <input
                  type="file"
                  ref={fileInputRefA}
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  style={{ display: "none" }}
                  onChange={(e) => handleFileChange(e.target.files?.[0] || null, "A")}
                />
                {previewUrlA && isBenchmarkFile(fileA) ? (
                  <img
                    src={previewUrlA}
                    alt="Scene A Preview"
                    style={{ width: "100%", height: "100px", objectFit: "cover", borderRadius: "6px", marginBottom: "8px" }}
                  />
                ) : (
                  <div style={{ fontSize: "28px", marginBottom: "6px" }}>{fileA ? "🗺️" : "📤"}</div>
                )}
                <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>
                  {fileA ? fileA.name : "Upload Scene A (T1)"}
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                  {fileA ? `${(fileA.size / 1024).toFixed(0)} KB • ${isGeoTiffFile(fileA) ? "GeoTIFF" : "Benchmark Image"}` : "Click to select GeoTIFF or Image"}
                </div>
                {fileA && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleFileChange(null, "A");
                    }}
                    style={{
                      position: "absolute",
                      top: "6px",
                      right: "6px",
                      background: "rgba(244, 63, 94, 0.2)",
                      border: "none",
                      borderRadius: "50%",
                      width: "20px",
                      height: "20px",
                      color: "var(--rose-bright)",
                      cursor: "pointer",
                      fontSize: "10px",
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>

              {/* Scene B */}
              <div
                onClick={() => fileInputRefB.current?.click()}
                className="upload-zone"
                style={{
                  border: "1px dashed rgba(59, 130, 246, 0.3)",
                  borderRadius: "10px",
                  padding: "16px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: fileB ? "rgba(59, 130, 246, 0.05)" : "rgba(10, 13, 19, 0.5)",
                  transition: "all 0.2s ease",
                  position: "relative",
                  overflow: "hidden",
                }}
              >
                <input
                  type="file"
                  ref={fileInputRefB}
                  accept=".tif,.tiff,.png,.jpg,.jpeg"
                  style={{ display: "none" }}
                  onChange={(e) => handleFileChange(e.target.files?.[0] || null, "B")}
                />
                {previewUrlB && isBenchmarkFile(fileB) ? (
                  <img
                    src={previewUrlB}
                    alt="Scene B Preview"
                    style={{ width: "100%", height: "100px", objectFit: "cover", borderRadius: "6px", marginBottom: "8px" }}
                  />
                ) : (
                  <div style={{ fontSize: "28px", marginBottom: "6px" }}>{fileB ? "🗺️" : "📤"}</div>
                )}
                <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>
                  {fileB ? fileB.name : "Upload Scene B (T2 / SAR)"}
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                  {fileB ? `${(fileB.size / 1024).toFixed(0)} KB • ${isGeoTiffFile(fileB) ? "GeoTIFF" : "Benchmark Image"}` : "For change detection / fusion"}
                </div>
                {fileB && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleFileChange(null, "B");
                    }}
                    style={{
                      position: "absolute",
                      top: "6px",
                      right: "6px",
                      background: "rgba(244, 63, 94, 0.2)",
                      border: "none",
                      borderRadius: "50%",
                      width: "20px",
                      height: "20px",
                      color: "var(--rose-bright)",
                      cursor: "pointer",
                      fontSize: "10px",
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>
            </div>

            {/* Validation Feedback Display */}
            {isValidatingPair && (
              <div style={{ padding: "10px 14px", borderRadius: "8px", background: "rgba(59, 130, 246, 0.08)", fontSize: "12px", color: "#60a5fa", display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="animate-spin-slow">⚙️</span>
                <span>Executing Phase 1 check_compatibility (CRS, spatial overlap, resolution)...</span>
              </div>
            )}

            {pairValidation && !isValidatingPair && (
              <div
                style={{
                  padding: "12px 14px",
                  borderRadius: "8px",
                  background: pairValidation.passed ? "rgba(16, 185, 129, 0.08)" : "rgba(244, 63, 94, 0.08)",
                  border: `1px solid ${pairValidation.passed ? "rgba(16, 185, 129, 0.25)" : "rgba(244, 63, 94, 0.25)"}`,
                  fontSize: "12px",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 600, color: pairValidation.passed ? "var(--emerald-bright)" : "var(--rose-bright)", marginBottom: "4px" }}>
                  <span>{pairValidation.passed ? "✓" : "⚠"}</span>
                  <span>{pairValidation.passed ? "Pre-Flight Compatibility Passed" : "Pair Compatibility Error"}</span>
                </div>
                <div style={{ color: "var(--text-secondary)", lineHeight: 1.4 }}>{pairValidation.message}</div>
                {pairValidation.crs && (
                  <div style={{ marginTop: "6px", display: "flex", gap: "14px", fontSize: "11px", color: "var(--text-muted)", fontFamily: "var(--font-geist-mono)" }}>
                    <span>CRS: {pairValidation.crs}</span>
                    <span>Overlap: {pairValidation.overlap ? "YES" : "NO"}</span>
                  </div>
                )}
              </div>
            )}

            {validationError && (
              <div style={{ marginTop: "8px", padding: "10px 14px", borderRadius: "8px", background: "rgba(244, 63, 94, 0.1)", border: "1px solid rgba(244, 63, 94, 0.25)", color: "var(--rose-bright)", fontSize: "12px" }}>
                {validationError}
              </div>
            )}
          </section>

          {/* QUESTION BOX */}
          <section className="glass-panel" style={{ padding: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="icon-badge">💬</span>
                <h2 style={{ fontSize: "15px", fontWeight: 600 }}>Natural Language Query Box</h2>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--text-muted)" }}>
                  <span aria-hidden>🌐</span>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value as "en" | "hi" | "hinglish")}
                    aria-label="Answer language"
                    title="Language the answer is rendered in (query understanding is cross-lingual)"
                    style={{
                      background: "#0a0d13",
                      color: "#ffffff",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "6px",
                      padding: "4px 8px",
                      fontSize: "12px",
                      fontFamily: "inherit",
                      outline: "none",
                      cursor: "pointer",
                    }}
                  >
                    <option value="en">English</option>
                    <option value="hi">हिन्दी (Hindi)</option>
                    <option value="hinglish">Hinglish</option>
                  </select>
                </label>
                <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>Auto-routed via Execution Planner</span>
              </div>
            </div>

            {/* Quick-Prompt Suggestions */}
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "14px" }}>
              {[
                { label: "Change: Built-Up Increase", q: "Has the built-up area increased between these two satellite scenes?" },
                { label: "VQA: Land Cover Class", q: "What type of land cover is visible across this scene?" },
                { label: "Grounding: Locate Vessels", q: "Locate maritime vessels or harbor infrastructure in the scene" },
                { label: "SAR Fusion: Flood Extent", q: "Evaluate water extent and flood boundaries fusing Optical and SAR data" },
              ].map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(item.q)}
                  className="chip"
                  style={{
                    fontSize: "11px",
                    padding: "5px 12px",
                    borderRadius: "20px",
                    background: "rgba(255, 255, 255, 0.04)",
                    border: "1px solid var(--border-subtle)",
                    color: "var(--text-secondary)",
                  }}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Textarea Input */}
            <textarea
              className="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask an analytical question about the satellite scene(s)..."
              rows={3}
              style={{
                width: "100%",
                padding: "12px 14px",
                borderRadius: "8px",
                background: "#0a0d13",
                border: "1px solid var(--border-subtle)",
                color: "#ffffff",
                fontSize: "14px",
                outline: "none",
                resize: "vertical",
                marginBottom: "14px",
                fontFamily: "inherit",
              }}
            />

            {/* Submit Action */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                {fileA ? "1 image attached" : "No image attached"} {fileB ? "+ 1 pair image" : ""}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <button
                onClick={handleCaptionScene}
                disabled={isExecuting || !fileA}
                title="Generate a natural-language description of Image A (single-image captioning task)"
                className="btn-soft"
                style={{
                  padding: "10px 18px",
                  borderRadius: "8px",
                  background: isExecuting || !fileA ? "rgba(255,255,255,0.05)" : "rgba(59, 130, 246, 0.15)",
                  color: isExecuting || !fileA ? "var(--text-muted)" : "#93b4f5",
                  border: `1px solid ${isExecuting || !fileA ? "rgba(255,255,255,0.1)" : "rgba(59, 130, 246, 0.4)"}`,
                  fontWeight: 600,
                  fontSize: "13px",
                  cursor: isExecuting || !fileA ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                <span>📝</span>
                <span>Describe Scene</span>
              </button>
              <button
                onClick={handleExecuteQuery}
                disabled={executeDisabled}
                className="btn-primary"
                title={!fileA ? "Attach a satellite scene (Image A) before running a query" : undefined}
                style={{
                  padding: "10px 22px",
                  borderRadius: "8px",
                  background: executeDisabled ? "rgba(59, 130, 246, 0.2)" : "linear-gradient(135deg, #3b82f6, #2563eb)",
                  color: "#ffffff",
                  border: "none",
                  fontWeight: 600,
                  fontSize: "13px",
                  cursor: executeDisabled ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  boxShadow: isExecuting ? "none" : "0 4px 14px rgba(59, 130, 246, 0.35)",
                }}
              >
                {isExecuting ? (
                  <>
                    <span className="animate-spin-slow">⚙️</span>
                    <span>Analyzing Scene...</span>
                  </>
                ) : (
                  <>
                    <span>Execute Query</span>
                    <span>→</span>
                  </>
                )}
              </button>
              </div>
            </div>

            {execError && (
              <div style={{ marginTop: "12px", padding: "10px 14px", borderRadius: "8px", background: "rgba(244, 63, 94, 0.1)", border: "1px solid rgba(244, 63, 94, 0.25)", color: "var(--rose-bright)", fontSize: "12px" }}>
                {execError}
              </div>
            )}
          </section>

        </div>

        {/* RIGHT COLUMN: Result Panel & Execution Trace Panel */}
        <div className="animate-fade-in-up" style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* RESULT PANEL (Answer + Confidence + Tabs + Download Report) */}
          <section className="glass-panel" style={{ padding: "20px" }}>
            
            {/* Header: Title + Download Report Action */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="icon-badge">🎯</span>
                <h2 style={{ fontSize: "15px", fontWeight: 600 }}>Analytical Results & Evidence</h2>
              </div>

              {/* Download Report Button with Tooltip */}
              <div style={{ position: "relative" }} title={!executionResult ? "Execute a query first to generate and export an auditable report." : "Export auditable report bundle (PDF or JSON)"}>
                <button
                  onClick={() => setShowReportModal(true)}
                  disabled={!executionResult || isExecuting}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "6px",
                    padding: "8px 14px",
                    borderRadius: "8px",
                    background: !executionResult || isExecuting ? "rgba(255, 255, 255, 0.05)" : "rgba(59, 130, 246, 0.12)",
                    border: `1px solid ${!executionResult || isExecuting ? "rgba(255, 255, 255, 0.1)" : "rgba(59, 130, 246, 0.35)"}`,
                    color: !executionResult || isExecuting ? "var(--text-muted)" : "var(--cyan-bright)",
                    fontSize: "12px",
                    fontWeight: 600,
                    cursor: !executionResult || isExecuting ? "not-allowed" : "pointer",
                    transition: "all 0.2s ease",
                  }}
                >
                  <span>📥</span>
                  <span>Download Report</span>
                </button>
              </div>
            </div>

            {/* Answer Display & Confidence Badge */}
            {executionResult ? (
              <div
                className="answer-reveal"
                style={{
                  padding: "16px",
                  borderRadius: "10px",
                  background: "rgba(10, 13, 19, 0.75)",
                  border: "1px solid var(--border-subtle)",
                  marginBottom: "18px",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px", gap: "12px" }}>
                  <span style={{ fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", fontWeight: 600 }}>
                    Audited Answer
                  </span>
                  
                  {/* Confidence / Agreement Badge */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                      padding: "4px 10px",
                      borderRadius: "9999px",
                      background: confInfo.bg,
                      border: `1px solid ${confInfo.border}`,
                      color: confInfo.color,
                      fontSize: "12px",
                      fontWeight: 600,
                    }}
                  >
                    <span>{confInfo.icon}</span>
                    <span>{currentConfidence !== null ? `${(currentConfidence * 100).toFixed(0)}% (${confInfo.bucket})` : confInfo.bucket}</span>
                  </div>
                </div>

                <div style={{ fontSize: "15px", lineHeight: 1.5, color: "#f8fafc", fontWeight: 500 }}>
                  {executionResult.answer}
                </div>
              </div>
            ) : (
              <div
                style={{
                  padding: "36px 20px",
                  textAlign: "center",
                  borderRadius: "10px",
                  background: "rgba(10, 13, 19, 0.4)",
                  border: "1px dashed var(--border-subtle)",
                  color: "var(--text-muted)",
                  fontSize: "13px",
                  marginBottom: "18px",
                }}
              >
                No query executed yet. Upload raster scenes and ask a question to view model predictions and visual evidence.
              </div>
            )}

            {/* Evidence Tabs: Answer | Before / After | Change Mask | Evidence */}
            <div style={{ display: "flex", borderBottom: "1px solid var(--border-subtle)", marginBottom: "16px", gap: "4px" }}>
              {[
                { id: "answer", label: "Overview" },
                { id: "beforeAfter", label: "Before / After (T1/T2)" },
                { id: "changeMask", label: "Change Mask" },
                { id: "evidence", label: "Visual Evidence (BBox)" },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as typeof activeTab)}
                  className="tab-btn"
                  style={{
                    padding: "8px 14px",
                    background: "transparent",
                    border: "none",
                    borderBottom: activeTab === tab.id ? "2px solid var(--cyan-bright)" : "2px solid transparent",
                    color: activeTab === tab.id ? "var(--cyan-bright)" : "var(--text-secondary)",
                    fontWeight: activeTab === tab.id ? 600 : 500,
                    fontSize: "12px",
                    cursor: "pointer",
                    transition: "all 0.15s ease",
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Contents */}
            <div style={{ minHeight: "180px" }}>
              {/* Tab 1: Overview */}
              {activeTab === "answer" && (
                <div style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                  {executionResult ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 12px", background: "rgba(255,255,255,0.02)", borderRadius: "6px" }}>
                        <span>Target Task:</span>
                        <strong style={{ color: "var(--text-primary)" }}>{executionResult.execution_trace?.task || "VQA / Change"}</strong>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 12px", background: "rgba(255,255,255,0.02)", borderRadius: "6px" }}>
                        <span>Status:</span>
                        <span style={{ color: executionResult.status === "success" ? "var(--emerald-bright)" : "var(--rose-bright)", fontWeight: 600 }}>
                          {executionResult.status.toUpperCase()}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "8px 12px", background: "rgba(255,255,255,0.02)", borderRadius: "6px" }}>
                        <span>Models Invocations:</span>
                        <span style={{ color: "var(--cyan-bright)", fontFamily: "var(--font-geist-mono)", fontSize: "12px" }}>
                          {(executionResult.execution_trace?.models_used && executionResult.execution_trace.models_used.length > 0)
                            ? executionResult.execution_trace.models_used.join(", ")
                            : (executionResult.status === "failed" || executionResult.status === "error" ? "None (Failed before execution)" : "vqa_engine")}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <p>Execute an analysis to view summary statistics.</p>
                  )}
                </div>
              )}

              {/* Tab 2: Before / After Thumbnails */}
              {activeTab === "beforeAfter" && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px" }}>
                  <div style={{ textAlign: "center", padding: "12px", background: "rgba(0,0,0,0.3)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>Scene T1 (Baseline)</div>
                    {executionResult?.before_after?.t1 || (previewUrlA && isBenchmarkFile(fileA)) ? (
                      <img src={executionResult?.before_after?.t1 || previewUrlA!} alt="Scene T1" style={{ width: "100%", height: "140px", objectFit: "cover", borderRadius: "6px" }} />
                    ) : (
                      <div style={{ height: "140px", display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(59, 130, 246, 0.05)", borderRadius: "6px", color: "var(--text-muted)", fontSize: "12px" }}>
                        {fileA ? "GeoTIFF Raw Raster Loaded" : "No T1 Scene Uploaded"}
                      </div>
                    )}
                  </div>
                  <div style={{ textAlign: "center", padding: "12px", background: "rgba(0,0,0,0.3)", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
                    <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>Scene T2 (Post-Event / SAR)</div>
                    {executionResult?.before_after?.t2 || (previewUrlB && isBenchmarkFile(fileB)) ? (
                      <img src={executionResult?.before_after?.t2 || previewUrlB!} alt="Scene T2" style={{ width: "100%", height: "140px", objectFit: "cover", borderRadius: "6px" }} />
                    ) : (
                      <div style={{ height: "140px", display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(59, 130, 246, 0.05)", borderRadius: "6px", color: "var(--text-muted)", fontSize: "12px" }}>
                        {fileB ? "GeoTIFF Raw Raster Loaded" : "No T2 Scene Uploaded"}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Tab 3: Change Mask */}
              {activeTab === "changeMask" && (() => {
                const evData = executionResult?.execution_trace?.evidence || executionResult?.outputs?.change_vqa || executionResult?.outputs?.change || {};
                const isChanged = evData.change_detected ?? (evData.changed_pixels ? evData.changed_pixels > 0 : false);
                const pct = evData.change_percentage ?? (evData.change_ratio ? evData.change_ratio * 100 : 0);
                return (
                  <div style={{ padding: "16px", background: "rgba(0,0,0,0.3)", borderRadius: "8px", textAlign: "center" }}>
                    <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "10px" }}>
                      Bi-Temporal Difference & Normalized Change Field
                    </div>
                    <div
                      style={{
                        height: "140px",
                        borderRadius: "6px",
                        background: isChanged
                          ? "radial-gradient(circle, rgba(239, 68, 68, 0.35) 20%, rgba(15, 23, 42, 0.8) 70%)"
                          : "radial-gradient(circle, rgba(16, 185, 129, 0.2) 20%, rgba(15, 23, 42, 0.8) 70%)",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        border: `1px solid ${isChanged ? "rgba(239, 68, 68, 0.3)" : "rgba(16, 185, 129, 0.3)"}`,
                      }}
                    >
                      <span style={{ fontSize: "24px", marginBottom: "4px" }}>🛰️</span>
                      <span style={{ fontSize: "13px", color: isChanged ? "#f87171" : "#34d399", fontWeight: 600 }}>
                        Change Detected: {isChanged ? "YES (Spatial Variance Found)" : "NO SIGNIFICANT CHANGE (0.0% Variance)"}
                      </span>
                      <span style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                        Change Ratio: {pct.toFixed(1)}% of surface area ({evData.changed_pixels ?? 0} changed pixels)
                      </span>
                    </div>
                  </div>
                );
              })()}

              {/* Tab 4: Visual Evidence Overlays */}
              {activeTab === "evidence" && (() => {
                const bbox = executionResult?.outputs?.grounding?.bbox || executionResult?.execution_trace?.evidence?.bbox || executionResult?.outputs?.change_vqa?.bbox;
                const conf = executionResult?.outputs?.grounding?.confidence ?? executionResult?.execution_trace?.evidence?.confidence ?? null;
                const gSource = executionResult?.outputs?.grounding?.grounding_source || executionResult?.outputs?.change_vqa?.grounding_source || "owlvit_model";
                return (
                  <div style={{ padding: "14px", background: "rgba(0,0,0,0.3)", borderRadius: "8px" }}>
                    <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
                      Grounded Object Bounding Box & Coordinates:
                    </div>
                    {bbox && Array.isArray(bbox) && bbox.length === 4 ? (
                      <div style={{ padding: "12px", borderRadius: "6px", background: "rgba(59, 130, 246, 0.08)", border: "1px solid rgba(59, 130, 246, 0.3)" }}>
                        <div style={{ fontSize: "12px", color: "var(--cyan-bright)", fontWeight: 600, marginBottom: "4px" }}>
                          Bounding Box: [{bbox.join(", ")}]
                        </div>
                        <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "2px" }}>
                          Confidence: {conf !== null && conf !== undefined ? `${(conf * 100).toFixed(1)}%` : "n/a"}
                        </div>
                        <div style={{ fontSize: "11px", color: gSource === "owlvit_model" ? "#34d399" : "#f59e0b", fontWeight: 600 }}>
                          Source: {gSource === "owlvit_model" ? "OwlViT Model (google/owlvit-base-patch32)" : (gSource === "unavailable" ? "Model Unavailable" : "Change-Mask Fallback")}
                        </div>
                      </div>
                    ) : (gSource === "unavailable" ? (
                      <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px", background: "rgba(239, 68, 68, 0.05)", borderRadius: "6px", border: "1px dashed rgba(239, 68, 68, 0.3)" }}>
                        <span style={{color: "#ef4444", fontWeight: 600}}>Model Unavailable</span><br />
                        <span style={{opacity: 0.8, display: "inline-block", marginTop: "4px"}}>No object detected. The grounding model could not be loaded or executed.</span>
                      </div>
                    ) : conf === 0.0 ? (
                      <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px", background: "rgba(245, 158, 11, 0.05)", borderRadius: "6px", border: "1px dashed rgba(245, 158, 11, 0.3)" }}>
                        <span style={{color: "#f59e0b", fontWeight: 600}}>No Object Detected</span><br />
                        <span style={{opacity: 0.8, display: "inline-block", marginTop: "4px"}}>The model executed but found no matching targets in the scene.</span>
                      </div>
                    ) : (
                      <div style={{ padding: "20px", textAlign: "center", color: "var(--text-muted)", fontSize: "12px" }}>
                        No spatial bounding box targeted for this specific query.
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          </section>

          {/* EXECUTION TRACE PANEL (Phase 12 Auditable Execution Trace) */}
          <section className="glass-panel" style={{ padding: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px", flexWrap: "wrap", gap: "10px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span className="icon-badge">🔍</span>
                <h2 style={{ fontSize: "15px", fontWeight: 600 }}>Phase 12 Auditable Execution Trace</h2>
              </div>
              <span
                style={{
                  fontSize: "11px",
                  padding: "2px 8px",
                  borderRadius: "4px",
                  background: "rgba(16, 185, 129, 0.1)",
                  color: "var(--emerald-bright)",
                  border: "1px solid rgba(16, 185, 129, 0.25)",
                }}
              >
                Verified Runtime Proof
              </span>
            </div>

            {/* Trace Sub-Tabs */}
            <div style={{ display: "flex", gap: "6px", marginBottom: "14px", flexWrap: "wrap" }}>
              {[
                { id: "models", label: "Models Used" },
                { id: "validation", label: "Validation Checks" },
                { id: "evidence", label: "Evidence Sources" },
                { id: "raw", label: "Raw JSON" },
              ].map((subTab) => (
                <button
                  key={subTab.id}
                  onClick={() => setTraceTab(subTab.id as typeof traceTab)}
                  className="tab-btn"
                  style={{
                    padding: "4px 10px",
                    borderRadius: "6px",
                    background: traceTab === subTab.id ? "rgba(59, 130, 246, 0.15)" : "rgba(255, 255, 255, 0.04)",
                    border: `1px solid ${traceTab === subTab.id ? "rgba(59, 130, 246, 0.4)" : "var(--border-subtle)"}`,
                    color: traceTab === subTab.id ? "var(--cyan-bright)" : "var(--text-secondary)",
                    fontSize: "11px",
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {subTab.label}
                </button>
              ))}
            </div>

            {/* Sub-Tab Viewers */}
            <div style={{ background: "#0a0d13", borderRadius: "8px", padding: "14px", border: "1px solid var(--border-subtle)", minHeight: "140px" }}>
              {/* Models Used */}
              {traceTab === "models" && (
                <div>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
                    Actual specialist model invocations verified during task graph execution:
                  </div>
                  {executionResult?.execution_trace?.models_used?.length ? (
                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                      {executionResult.execution_trace.models_used.map((m, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "6px 12px",
                            borderRadius: "6px",
                            background: "rgba(59, 130, 246, 0.1)",
                            border: "1px solid rgba(59, 130, 246, 0.25)",
                            color: "#60a5fa",
                            fontSize: "12px",
                            fontFamily: "var(--font-geist-mono)",
                          }}
                        >
                          ✓ {m}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                      Execute a query to trace live model calls.
                    </div>
                  )}
                </div>
              )}

              {/* Validation Checks */}
              {traceTab === "validation" && (
                <div>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
                    Pre-flight validation gates (CRS, geometry integrity, band counts):
                  </div>
                  {executionResult?.execution_trace?.validation ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      {(() => {
                        const val = executionResult.execution_trace!.validation!;
                        const ok = val.passed ?? (val.status === "passed");
                        return (
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", padding: "4px 8px", background: "rgba(255,255,255,0.04)", borderRadius: "4px" }}>
                            <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>
                              Overall{val.status ? ` (${val.status})` : ""}:
                            </span>
                            <span style={{ color: ok ? "var(--emerald-bright)" : "var(--rose-bright)", fontWeight: 700 }}>
                              {ok ? "PASSED" : "FAILED"}
                            </span>
                          </div>
                        );
                      })()}
                      {(executionResult.execution_trace.validation.checks ?? []).map((c, idx) => (
                        <div key={idx} style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", padding: "4px 8px", background: "rgba(255,255,255,0.02)", borderRadius: "4px" }}>
                          <span style={{ color: "var(--text-secondary)" }}>{c.check}:</span>
                          <span style={{ color: c.passed ? "var(--emerald-bright)" : "var(--rose-bright)", fontWeight: 600 }}>
                            {c.passed ? "PASSED" : "FAILED"} {c.detail ? `(${c.detail})` : ""}
                          </span>
                        </div>
                      ))}
                      {executionResult.execution_trace.validation.error && (
                        <div style={{ fontSize: "11px", color: "var(--rose-bright)", padding: "4px 8px" }}>
                          {executionResult.execution_trace.validation.error}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                      {pairValidation ? "Pre-flight validation recorded from Upload Panel." : "No validation logs recorded."}
                    </div>
                  )}
                </div>
              )}

              {/* Evidence Sources */}
              {traceTab === "evidence" && (
                <div>
                  <div style={{ fontSize: "11px", color: "var(--text-muted)", marginBottom: "8px" }}>
                    Evidence collected across task graph nodes:
                  </div>
                  {executionResult?.execution_trace?.evidence ? (
                    <pre style={{ fontSize: "11px", color: "#94a3b8", maxHeight: "180px", overflow: "auto" }}>
                      {JSON.stringify(executionResult.execution_trace.evidence, null, 2)}
                    </pre>
                  ) : (
                    <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                      Evidence will appear once query completes.
                    </div>
                  )}
                </div>
              )}

              {/* Raw JSON */}
              {traceTab === "raw" && (
                <pre style={{ fontSize: "11px", color: "#60a5fa", maxHeight: "200px", overflow: "auto" }}>
                  {JSON.stringify(executionResult?.execution_trace || { status: "ready", waiting_for_execution: true }, null, 2)}
                </pre>
              )}
            </div>
          </section>

        </div>

      </div>
      )}

      {/* DOWNLOAD REPORT MODAL */}
      {showReportModal && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0, 0, 0, 0.75)",
            backdropFilter: "blur(8px)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 9999,
            padding: "20px",
          }}
          onClick={() => setShowReportModal(false)}
        >
          <div
            className="glass-panel"
            style={{
              padding: "24px",
              maxWidth: "500px",
              width: "100%",
              background: "#0c1324",
              border: "1px solid rgba(59, 130, 246, 0.3)",
              boxShadow: "0 20px 50px rgba(0,0,0,0.8)",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "18px" }}>📥</span>
                <h3 style={{ fontSize: "16px", fontWeight: 700 }}>Export Audit Report</h3>
              </div>
              <button
                onClick={() => setShowReportModal(false)}
                style={{ background: "transparent", border: "none", color: "var(--text-muted)", fontSize: "16px", cursor: "pointer" }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "18px", lineHeight: 1.5 }}>
              Bundle the query, answer, confidence score (🟢/🟡/🔴), visual evidence, and complete Phase 12 execution trace into an auditable document.
            </p>

            {/* Format Selection */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "20px" }}>
              <div
                onClick={() => setReportFormat("json")}
                style={{
                  padding: "14px",
                  borderRadius: "8px",
                  border: `1px solid ${reportFormat === "json" ? "var(--cyan-bright)" : "var(--border-subtle)"}`,
                  background: reportFormat === "json" ? "rgba(59, 130, 246, 0.12)" : "rgba(255, 255, 255, 0.02)",
                  cursor: "pointer",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "20px", marginBottom: "4px" }}>📋</div>
                <div style={{ fontSize: "13px", fontWeight: 600, color: reportFormat === "json" ? "var(--cyan-bright)" : "var(--text-primary)" }}>
                  JSON Bundle
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                  Machine-readable audit file
                </div>
              </div>

              <div
                onClick={() => setReportFormat("pdf")}
                style={{
                  padding: "14px",
                  borderRadius: "8px",
                  border: `1px solid ${reportFormat === "pdf" ? "var(--cyan-bright)" : "var(--border-subtle)"}`,
                  background: reportFormat === "pdf" ? "rgba(59, 130, 246, 0.12)" : "rgba(255, 255, 255, 0.02)",
                  cursor: "pointer",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: "20px", marginBottom: "4px" }}>📄</div>
                <div style={{ fontSize: "13px", fontWeight: 600, color: reportFormat === "pdf" ? "var(--cyan-bright)" : "var(--text-primary)" }}>
                  PDF Document
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>
                  Printable formatted audit report
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
              <button
                onClick={() => setShowReportModal(false)}
                style={{
                  padding: "8px 16px",
                  borderRadius: "6px",
                  background: "transparent",
                  border: "1px solid var(--border-subtle)",
                  color: "var(--text-secondary)",
                  fontSize: "13px",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleDownloadReport}
                disabled={isExporting}
                style={{
                  padding: "8px 20px",
                  borderRadius: "6px",
                  background: "linear-gradient(135deg, #3b82f6, #2563eb)",
                  border: "none",
                  color: "#ffffff",
                  fontSize: "13px",
                  fontWeight: 600,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                {isExporting ? "Exporting..." : `Download ${reportFormat.toUpperCase()}`}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Docker Compose Quickstart Footer */}
      <footer
        className="glass-panel"
        style={{
          padding: "24px 28px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "16px",
          marginTop: "40px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              padding: "6px 12px",
              background: "rgba(59, 130, 246, 0.1)",
              borderRadius: "6px",
              color: "#60a5fa",
              fontSize: "12px",
              fontFamily: "var(--font-geist-mono)",
            }}
          >
            docker compose up --build
          </div>
          <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
            Builds and runs frontend:3000 & backend:8000
          </span>
        </div>

        <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
          Dependencies: fastapi, uvicorn, rasterio, pyproj, python-multipart, pytest, httpx
        </div>
      </footer>
      </main>
    </div>
  );
}
