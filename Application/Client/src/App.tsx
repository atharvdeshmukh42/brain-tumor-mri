import { useCallback, useEffect, useState } from "react";
import { Download, Plus, ScanLine, X } from "lucide-react";
import { z } from "zod";
import {
  analysisSchema,
  healthSchema,
  request,
  samplesSchema,
  type Analysis,
} from "./api";
import Viewer from "./Viewer";

const uploadFields = [
  ["flair", "FLAIR"],
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
  ["seg", "Expert reference"],
] as const;

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
      if (
        reference instanceof File &&
        !reference.name &&
        reference.size === 0
      ) {
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
  const sampleCase = samples.find((item) => item.id === sample);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="NeuroScope home">
          <span className="brand-icon">
            <ScanLine size={20} />
          </span>
          <span>
            NeuroScope<small>MRI workstation</small>
          </span>
        </a>
        <button
          className={`nav-item ${intake ? "active" : ""}`}
          aria-current={intake ? "page" : undefined}
          onClick={() => setIntake(true)}
        >
          <Plus size={17} /> New analysis
        </button>
        <div className="history-heading">
          <h2>Analysis history</h2>
          <span className="mono">{history.length}</span>
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
              aria-current={
                !intake && selected?.id === item.id ? "page" : undefined
              }
              onClick={() => {
                setSelected(item);
                setIntake(false);
              }}
            >
              <span className={`status-dot ${item.status}`} />
              <span>
                <strong>{item.source.replace("BraTS2021_", "BraTS ")}</strong>
                <small>
                  <time dateTime={item.created_at}>
                    {new Date(item.created_at).toLocaleString(undefined, {
                      month: "short",
                      day: "numeric",
                      hour: "numeric",
                      minute: "2-digit",
                    })}
                  </time>{" "}
                  <span aria-hidden="true">/</span> {item.status}
                </small>
              </span>
            </button>
          ))}
        </nav>
      </aside>
      <main>
        <div className="content">
          <header className="page-heading">
            <h1>
              {intake
                ? "New analysis"
                : selected?.source.replace("BraTS2021_", "BraTS ")}
            </h1>
            <div className="model-status">
              <span
                className={`status-dot ${ready ? "completed" : health?.models.status === "error" ? "failed" : "running"}`}
              />
              <span>
                {ready
                  ? "Models ready"
                  : health?.models.status === "error"
                    ? "Models unavailable"
                    : "Loading models"}
              </span>
              {ready && health.models.device && (
                <span className="device mono">
                  {health.models.device.toUpperCase()}
                </span>
              )}
            </div>
          </header>
          {health?.models.error && (
            <p className="error-banner" role="alert">
              {health.models.error}
            </p>
          )}
          {error && (
            <div className="error-banner" role="alert">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}
          {intake ? (
            <section className="panel intake">
              <div className="tabs" role="group" aria-label="Scan source">
                <button
                  aria-pressed={tab === "sample"}
                  onClick={() => setTab("sample")}
                  disabled={busy}
                >
                  Sample case
                </button>
                <button
                  aria-pressed={tab === "upload"}
                  onClick={() => setTab("upload")}
                  disabled={busy}
                >
                  Upload scans
                </button>
              </div>
              <div className="intake-body">
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void start(event.currentTarget);
                  }}
                >
                  {tab === "sample" ? (
                    <>
                      <label className="field-label" htmlFor="sample">
                        BraTS 2021 case
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
                          </option>
                        ))}
                      </select>
                      {sampleCase && (
                        <dl className="case-info">
                          <div>
                            <dt>Sequences</dt>
                            <dd>FLAIR, T1, T1ce, T2</dd>
                          </div>
                          <div>
                            <dt>Expert reference</dt>
                            <dd>
                              {sampleCase.has_reference
                                ? "Included"
                                : "Not available"}
                            </dd>
                          </div>
                        </dl>
                      )}
                    </>
                  ) : (
                    <>
                      <p className="upload-note">
                        Use aligned, skull-stripped 3D NIfTI files with matching
                        dimensions and affines in millimeters. .nii or .nii.gz,
                        up to 128 MiB per file.
                      </p>
                      <div className="file-grid">
                        {uploadFields.map(([name, label]) => (
                          <label key={name} className="upload-field">
                            <span>
                              {label}{" "}
                              {name === "seg" && <small>Optional</small>}
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
                  <div className="form-actions">
                    <button
                      className="primary"
                      type="submit"
                      disabled={busy || !ready || (tab === "sample" && !sample)}
                    >
                      {busy ? (
                        <>
                          <span className="spinner" /> Preparing case…
                        </>
                      ) : (
                        "Run analysis"
                      )}
                    </button>
                    <p>Analysis may take several minutes.</p>
                  </div>
                </form>
              </div>
            </section>
          ) : (
            selected && (
              <>
                <div className="case-meta">
                  <span className="mono">
                    {selected.metadata.shape.join(" × ")} voxels
                  </span>
                  <span>
                    {selected.metadata.has_reference
                      ? "Expert reference available"
                      : "No expert reference"}
                  </span>
                  <span
                    className={`status-pill ${selected.status}`}
                    role="status"
                  >
                    {selected.status === "running" ? (
                      <span className="spinner" />
                    ) : (
                      <span className={`status-dot ${selected.status}`} />
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
                        You can explore the scan while detection and
                        segmentation finish.
                      </span>
                    </div>
                  </div>
                )}
                <div className="analysis-grid">
                  <Viewer key={selected.id} analysis={selected} />
                  <aside className="panel">
                    <div className="panel-title">
                      <h2>Findings</h2>
                    </div>
                    <div className="results-body">
                      {report ? (
                        <>
                          <h3>Detection</h3>
                          <div className="probability">
                            {(report.detection.max_probability * 100).toFixed(
                              1,
                            )}
                            <span>%</span>
                          </div>
                          <p className="muted">
                            Peak model probability across slices
                          </p>
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
                          <h3>Region volumes</h3>
                          {(
                            ["WT", "TC", "ET"] satisfies Array<
                              keyof typeof report.localization
                            >
                          ).map((key) => (
                            <div className="measurement" key={key}>
                              <span>
                                <small>
                                  {key === "WT"
                                    ? "Whole tumor"
                                    : key === "TC"
                                      ? "Tumor core"
                                      : "Enhancing tumor"}
                                </small>
                                <span className="region-code mono">{key}</span>
                              </span>
                              <b>
                                {report.localization[key].volume_cm3.toFixed(2)}{" "}
                                <small>cm³</small>
                              </b>
                            </div>
                          ))}
                          <p className="muted small">
                            Regions overlap: ET is within TC, and TC is within
                            WT.
                          </p>
                          <hr />
                          <h3>Localization</h3>
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
                  </aside>
                </div>
              </>
            )
          )}
        </div>
      </main>
    </div>
  );
}
