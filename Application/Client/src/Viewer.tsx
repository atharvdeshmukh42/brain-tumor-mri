import { useEffect, useState } from "react";
import { Layers, ScanLine } from "lucide-react";
import { z } from "zod";
import type { Analysis } from "./api";
const planeSchema = z.enum(["axial", "coronal", "sagittal"]);
const modalitySchema = z.enum(["flair", "t1", "t1ce", "t2"]);
const overlaySchema = z.enum(["none", "predicted", "reference"]);

export default function Viewer({ analysis }: { analysis: Analysis }) {
  const [plane, setPlane] = useState<z.infer<typeof planeSchema>>("axial");
  const [modality, setModality] =
    useState<z.infer<typeof modalitySchema>>("flair");
  const [index, setIndex] = useState(
    Math.floor(analysis.metadata.shape[2] / 2),
  );
  const [overlay, setOverlay] = useState<z.infer<typeof overlaySchema>>("none");
  const [opacity, setOpacity] = useState(0.55);
  const [cam, setCam] = useState(false);
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const completed = analysis.status === "completed";
  const axis = plane === "axial" ? 2 : plane === "coronal" ? 1 : 0;
  const max = analysis.metadata.shape[axis] - 1;
  const safeIndex = Math.min(index, max);
  useEffect(() => {
    const controller = new AbortController();
    let objectUrl = "";
    setLoading(true);
    setError("");
    const timer = window.setTimeout(() => {
      const query = new URLSearchParams({
        modality,
        plane,
        index: String(safeIndex),
        overlay,
        opacity: String(opacity),
        gradcam: String(cam && plane === "axial"),
      });
      fetch(`/api/analyses/${analysis.id}/slice?${query}`, {
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok) {
            const result = z
              .object({ detail: z.string() })
              .safeParse(await response.json());
            throw new Error(
              result.success ? result.data.detail : "Unable to load slice",
            );
          }
          objectUrl = URL.createObjectURL(await response.blob());
          if (!controller.signal.aborted) {
            setUrl(objectUrl);
            setLoading(false);
          }
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted) {
            setError(
              reason instanceof Error ? reason.message : "Unable to load slice",
            );
            setLoading(false);
          }
        });
    }, 100);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [analysis.id, modality, plane, safeIndex, overlay, opacity, cam]);
  return (
    <section className="panel viewer">
      <div className="panel-title">
        <span>
          <ScanLine size={17} /> MRI explorer
        </span>
        <span className="muted">Original voxel grid</span>
      </div>
      <div className="viewer-toolbar">
        <div className="segmented" aria-label="MRI modality">
          {modalitySchema.options.map((value) => (
            <button
              key={value}
              aria-pressed={modality === value}
              onClick={() => setModality(value)}
            >
              {value.toUpperCase()}
            </button>
          ))}
        </div>
        <label className="sr-only" htmlFor="plane">
          Plane
        </label>
        <select
          id="plane"
          value={plane}
          onChange={(event) => {
            const value = planeSchema.parse(event.target.value);
            setPlane(value);
            setIndex(
              Math.floor(
                analysis.metadata.shape[
                  value === "axial" ? 2 : value === "coronal" ? 1 : 0
                ] / 2,
              ),
            );
            setCam(false);
          }}
        >
          {planeSchema.options.map((value) => (
            <option key={value}>{value}</option>
          ))}
        </select>
      </div>
      <div className="scan-canvas" aria-busy={loading}>
        <span className="scan-caption">
          {modality.toUpperCase()} / {plane.toUpperCase()}
        </span>
        {url && !error && (
          <img
            src={url}
            alt={`${modality.toUpperCase()} ${plane} MRI slice ${safeIndex + 1}, ${overlay} overlay${cam ? ", Grad-CAM" : ""}`}
            style={{ opacity: loading ? 0.35 : 1 }}
          />
        )}
        <span className="scan-grid">
          {analysis.metadata.shape.join(" × ")}
          <br />
          Voxel axes {analysis.metadata.orientation.join(" / ")}
        </span>
        {loading && (
          <div className="scan-message" role="status">
            <span className="spinner" />
            {cam ? "Computing explanation…" : "Loading slice…"}
          </div>
        )}
        {error && (
          <div className="scan-message error" role="alert">
            {error}
          </div>
        )}
      </div>
      <div className="slice-control">
        <label htmlFor="slice">
          Slice <strong>{safeIndex + 1}</strong>
          <span className="muted"> / {max + 1}</span>
        </label>
        <input
          id="slice"
          type="range"
          min="0"
          max={max}
          value={safeIndex}
          onChange={(event) => setIndex(Number(event.target.value))}
        />
        <div className="muted">Use arrow keys to move through slices</div>
      </div>
      <div className="overlay-controls">
        <label>
          <Layers size={15} /> Overlay
          <select
            aria-label="Overlay"
            value={overlay}
            onChange={(event) =>
              setOverlay(overlaySchema.parse(event.target.value))
            }
          >
            <option value="none">None</option>
            <option value="predicted" disabled={!completed}>
              Prediction
            </option>
            <option
              value="reference"
              disabled={!analysis.metadata.has_reference}
            >
              Expert reference
            </option>
          </select>
        </label>
        <label>
          Opacity{" "}
          <input
            aria-label="Overlay opacity"
            type="range"
            min="0"
            max="1"
            step=".05"
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={cam}
            disabled={!completed || plane !== "axial"}
            onChange={(event) => setCam(event.target.checked)}
          />{" "}
          Grad-CAM <span className="muted">axial</span>
        </label>
      </div>
      <div className="legend">
        <span>
          <i className="edema" />
          Edema (2)
        </span>
        <span>
          <i className="core" />
          Necrotic core (1)
        </span>
        <span>
          <i className="enhancing" />
          Enhancing (4)
        </span>
      </div>
      {cam && (
        <p className="viewer-note">
          Grad-CAM highlights detector attention for this axial slice. It does
          not define tumor boundaries.
        </p>
      )}
    </section>
  );
}
