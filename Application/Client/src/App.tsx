import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Download,
  FileUp,
  History,
  Plus,
  ShieldCheck,
} from "lucide-react";
import { z } from "zod";
import {
  analysisSchema,
  healthSchema,
  request,
  samplesSchema,
  type Analysis,
} from "./api";
import Viewer from "./Viewer";

const modalities = ["flair", "t1", "t1ce", "t2", "seg"];
export default function App() {
  const [health, setHealth] = useState<z.infer<typeof healthSchema> | null>(
    null,
  );
  const [samples, setSamples] = useState<z.infer<typeof samplesSchema>>([]);
  const [history, setHistory] = useState<Analysis[]>([]);
  const [selected, setSelected] = useState<Analysis | null>(null);
  const [intake, setIntake] = useState(true);
  const [tab, setTab] = useState<"sample" | "upload">("sample");
  const [sample, setSample] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    const [nextHealth, nextSamples, nextHistory] = await Promise.all([
      request("/health", healthSchema),
      request("/samples", samplesSchema),
      request("/analyses", z.array(analysisSchema)),
    ]);
    setHealth(nextHealth);
    setSamples(nextSamples);
    setHistory(nextHistory);
    setSample((previous) => previous || nextSamples[0]?.id || "");
    setSelected((previous) =>
      previous
        ? (nextHistory.find((item) => item.id === previous.id) ?? previous)
        : null,
    );
  }, []);
  useEffect(() => {
    let active = true;
    const update = () => {
      refresh().catch((reason: unknown) => {
        if (active)
          setError(
            reason instanceof Error
              ? reason.message
              : "Cannot connect to server",
          );
      });
    };
    update();
    const timer = window.setInterval(update, 3000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [refresh]);
  async function start(form: HTMLFormElement) {
    setBusy(true);
    setError("");
    try {
      const formData = new FormData(form);
      const reference = formData.get("seg");
      if (reference instanceof File && !reference.name && reference.size === 0) {
        formData.delete("seg");
      }
      const result = await request(
        tab === "sample" ? "/analyses/sample" : "/analyses/upload",
        analysisSchema,
        tab === "sample"
          ? {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ sample_id: sample }),
            }
          : { method: "POST", body: formData },
      );
      setSelected(result);
      setIntake(false);
      await refresh();
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "Analysis could not be started",
      );
    } finally {
      setBusy(false);
    }
  }
  const ready = health?.models.status === "ready";
  const report = selected?.report;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="NeuroScope home">
          <div className="brand-icon">
            <BrainCircuit size={25} />
          </div>
          <div>
            NeuroScope<span>MRI RESEARCH WORKSTATION</span>
          </div>
        </a>
        <div className="workspace-label">WORKSPACE</div>
        <button
          className={`nav-item ${intake ? "active" : ""}`}
          onClick={() => setIntake(true)}
        >
          <Plus size={18} /> New analysis
        </button>
        <div className="history-heading">
          <span>
            <History size={15} /> RECENT ANALYSES
          </span>
          <span>{history.length}</span>
        </div>
        <nav className="history-list" aria-label="Analysis history">
          {history.length === 0 && (
            <p className="muted empty-history">
              Your analyses will appear here.
            </p>
          )}
          {history.map((item) => (
            <button
              key={item.id}
              className={`history-item ${!intake && selected?.id === item.id ? "selected" : ""}`}
              onClick={() => {
                setSelected(item);
                setIntake(false);
              }}
            >
              <span className={`status-dot ${item.status}`} />
              <span>
                <strong>{item.source.replace("BraTS2021_", "BraTS · ")}</strong>
                <small>
                  {new Date(item.created_at).toLocaleDateString()} ·{" "}
                  {item.status}
                </small>
              </span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <ShieldCheck size={18} />
          <div>
            Local by design<small>Scans stay on this machine</small>
          </div>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <span>
            <span className="muted">Workspace</span>
            <span className="slash">/</span>
            {intake ? "New analysis" : "Case analysis"}
          </span>
          <span className="research-badge">RESEARCH USE ONLY</span>
        </header>
        <div className="content">
          <div className="page-heading">
            <div className="eyebrow">BRAIN MRI · MULTIMODAL ANALYSIS</div>
            <h1>
              {intake
                ? "A clearer view of the brain."
                : selected?.source.replace("BraTS2021_", "BraTS · ")}
            </h1>
            <p>
              {intake
                ? "Explore tumor detection, segmentation, and model explanations in one place."
                : "Inspect the scan, compare regions, and explore the model’s prediction."}
            </p>
          </div>
          <div
            className={`model-status ${health?.models.status === "error" ? "model-error" : ""}`}
          >
            <span className={`status-dot ${ready ? "completed" : "running"}`} />
            <span>
              {ready
                ? `Models ready · ${health.models.device?.toUpperCase()}`
                : health?.models.status === "error"
                  ? "Models unavailable"
                  : "Connecting & loading models"}
            </span>
            <span className="model-names">EfficientNet-B0 + SegResNet</span>
          </div>
          {health?.models.error && (
            <p className="error-banner" role="alert">
              {health.models.error}
            </p>
          )}
          {error && (
            <div className="error-banner" role="alert">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          {intake ? (
            <>
              <div className="intake-grid">
                <section className="panel intake">
                  <div className="panel-title">
                    <span>
                      <FileUp size={18} /> Start an analysis
                    </span>
                    <span className="muted">01 / INPUT</span>
                  </div>
                  <div className="intake-body">
                    <div className="tabs">
                      <button
                        aria-pressed={tab === "sample"}
                        onClick={() => setTab("sample")}
                      >
                        Sample case
                      </button>
                      <button
                        aria-pressed={tab === "upload"}
                        onClick={() => setTab("upload")}
                      >
                        Upload scans
                      </button>
                    </div>
                    <form
                      onSubmit={(event) => {
                        event.preventDefault();
                        void start(event.currentTarget);
                      }}
                    >
                      {tab === "sample" ? (
                        <>
                          <h2>Explore a prepared case</h2>
                          <p className="muted">
                            Run the trained models on a local BraTS sample, with
                            expert labels for comparison.
                          </p>
                          <label className="field-label" htmlFor="sample">
                            SAMPLE CASE
                          </label>
                          <select
                            id="sample"
                            value={sample}
                            onChange={(event) => setSample(event.target.value)}
                            disabled={busy}
                          >
                            {samples.length === 0 && (
                              <option value="">No local samples found</option>
                            )}
                            {samples.map((item) => (
                              <option key={item.id} value={item.id}>
                                {item.id}
                                {item.has_reference
                                  ? " · expert reference included"
                                  : ""}
                              </option>
                            ))}
                          </select>
                          <div className="case-info">
                            <BrainCircuit size={30} />
                            <div>
                              <strong>
                                Four modalities. One complete picture.
                              </strong>
                              <span>FLAIR · T1 · T1ce · T2</span>
                            </div>
                            <CheckCircle2 size={20} />
                          </div>
                        </>
                      ) : (
                        <>
                          <h2>Bring your own scans</h2>
                          <p className="muted">
                            Use co-registered, skull-stripped, preprocessed 3D
                            NIfTI volumes with matching shape and affine. Up to
                            128 MiB per file.
                          </p>
                          <div className="file-grid">
                            {modalities.map((name) => (
                              <label key={name} className="upload-field">
                                <span>
                                  {name === "seg"
                                    ? "Expert labels (optional)"
                                    : name.toUpperCase()}
                                </span>
                                <input
                                  name={name}
                                  type="file"
                                  accept=".nii,.nii.gz"
                                  required={name !== "seg"}
                                  disabled={busy}
                                />
                              </label>
                            ))}
                          </div>
                        </>
                      )}
                      <button
                        className="primary"
                        type="submit"
                        disabled={
                          busy || !ready || (tab === "sample" && !sample)
                        }
                      >
                        {busy ? (
                          <>
                            <span className="spinner" /> Preparing case…
                          </>
                        ) : (
                          <>
                            Run analysis <ArrowRight size={17} />
                          </>
                        )}
                      </button>
                      <p className="form-note">
                        Detection and segmentation run locally. Processing may
                        take several minutes.
                      </p>
                    </form>
                  </div>
                </section>
                <aside className="intro-card">
                  <div className="brain-illustration">
                    <BrainCircuit size={115} strokeWidth={0.8} />
                    <span className="orbit one" />
                    <span className="orbit two" />
                  </div>
                  <div className="eyebrow">FROM SCAN TO INSIGHT</div>
                  <h2>See beyond the image.</h2>
                  <p>
                    Navigate MRI slices, examine tumor regions, and see where
                    the detection model focuses.
                  </p>
                  <div className="feature">
                    <Activity size={17} />
                    <span>Slice-level tumor detection</span>
                  </div>
                  <div className="feature">
                    <BrainCircuit size={17} />
                    <span>Whole tumor, core & enhancing regions</span>
                  </div>
                  <div className="feature">
                    <Activity size={17} />
                    <span>Axial Grad-CAM explanations</span>
                  </div>
                </aside>
              </div>
              <div className="research-note">
                <ShieldCheck size={20} />
                <p>
                  <strong>Built for exploration, not diagnosis.</strong> Model
                  outputs are research predictions and require expert
                  interpretation. This application is not a medical device.
                </p>
              </div>
            </>
          ) : (
            selected && (
              <>
                <div className="case-meta">
                  <span>{selected.metadata.shape.join(" × ")} voxels</span>
                  <span>
                    {selected.metadata.has_reference
                      ? "Expert reference available"
                      : "No expert reference"}
                  </span>
                  <span className={`status-pill ${selected.status}`}>
                    {selected.status === "running" ? (
                      <span className="spinner" />
                    ) : (
                      <Clock3 size={13} />
                    )}
                    {selected.status}
                  </span>
                </div>
                {selected.error && (
                  <p className="error-banner" role="alert">
                    {selected.error}
                  </p>
                )}
                {["queued", "running"].includes(selected.status) && (
                  <div className="progress-banner" role="status">
                    <span className="spinner" />
                    <div>
                      <strong>
                        {selected.status === "queued"
                          ? "Case queued"
                          : "Analyzing your scan"}
                      </strong>
                      <span>
                        Detection and 3D segmentation are running locally. You
                        can explore the scan while you wait.
                      </span>
                    </div>
                  </div>
                )}
                <div className="analysis-grid">
                  <Viewer key={selected.id} analysis={selected} />
                  <aside className="results">
                    <section className="panel">
                      <div className="panel-title">
                        <span>
                          <Activity size={16} /> Analysis findings
                        </span>
                      </div>
                      <div className="results-body">
                        {report ? (
                          <>
                            <div className="eyebrow">
                              MAXIMUM SLICE PROBABILITY
                            </div>
                            <div className="probability">
                              {(report.detection.max_probability * 100).toFixed(
                                1,
                              )}
                              <span>%</span>
                            </div>
                            <div className="probability-bar">
                              <i
                                style={{
                                  width: `${report.detection.max_probability * 100}%`,
                                }}
                              />
                            </div>
                            <p className="detection-label">
                              {report.detection.tumor_detected
                                ? "Tumor signal detected"
                                : "Below detection threshold"}
                            </p>
                            <p className="muted">
                              Threshold{" "}
                              {(report.detection.threshold * 100).toFixed(1)}% ·{" "}
                              {report.detection.n_slices_checked} slices checked
                            </p>
                            <hr />
                            <div className="eyebrow">REGION VOLUMES</div>
                            {(
                              ["WT", "TC", "ET"] satisfies Array<
                                keyof typeof report.localization
                              >
                            ).map((key) => (
                              <div className="measurement" key={key}>
                                <span>
                                  <i
                                    className={
                                      key === "WT"
                                        ? "edema"
                                        : key === "TC"
                                          ? "core"
                                          : "enhancing"
                                    }
                                  />
                                  <strong>{key}</strong>
                                  <small>
                                    {key === "WT"
                                      ? "Whole tumor"
                                      : key === "TC"
                                        ? "Tumor core"
                                        : "Enhancing tumor"}
                                  </small>
                                </span>
                                <b>
                                  {report.localization[key].volume_cm3.toFixed(
                                    2,
                                  )}{" "}
                                  <small>cm³</small>
                                </b>
                              </div>
                            ))}
                            <p className="muted small">
                              Regions overlap: ET is within TC, and TC is within
                              WT.
                            </p>
                            <hr />
                            <div className="eyebrow">LOCALIZATION</div>
                            <p className="summary">{report.summary}</p>
                            <div className="download-links">
                              <a
                                href={`/api/analyses/${selected.id}/download/report`}
                              >
                                <Download size={15} /> Report JSON
                              </a>
                              <a
                                href={`/api/analyses/${selected.id}/download/mask`}
                              >
                                <Download size={15} /> NIfTI mask
                              </a>
                            </div>
                          </>
                        ) : (
                          <div className="awaiting">
                            <BrainCircuit size={32} />
                            <h3>
                              {selected.status === "failed"
                                ? "Analysis failed"
                                : "Findings will appear here"}
                            </h3>
                            <p>
                              Volumes and localization are available after a
                              successful analysis.
                            </p>
                          </div>
                        )}
                      </div>
                    </section>
                    <p className="result-note">
                      Research predictions require expert interpretation.
                      Probability describes detector confidence, not a clinical
                      diagnosis.
                    </p>
                  </aside>
                </div>
              </>
            )
          )}
          <footer>
            NeuroScope<span>Local inference · Research use only</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
