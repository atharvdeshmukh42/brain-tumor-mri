---
name: verify
description: Verify the NeuroScope MRI React interface and FastAPI service with real local models and sample scans.
---

# Verify NeuroScope

## Launch

Run from the repository root. Follow Application/README.md setup first. Start exactly one backend process on a free port. Use an isolated MRI_STORAGE_DIR for an independent run. A second API must never share the active instance's storage because startup recovers interrupted jobs.

```sh
MRI_DEVICE=cpu MRI_STORAGE_DIR=/tmp/neuroscope-verify Application/Server/.venv/bin/python -m uvicorn main:app --app-dir Application/Server --host 127.0.0.1 --port 8010
VITE_API_TARGET=http://127.0.0.1:8010 npm run dev --prefix Application/Client -- --port 5173 --strictPort
```

Wait for /api/health to report models.status ready. Loading or an error is not readiness.

## Doctor

```sh
curl --fail http://127.0.0.1:8010/api/health
```

Check that this process owns port 8010 and reports the intended device. Do not stop an unrelated service to claim a port.

## Drive

Run API behavior tests and the production build.

```sh
Application/Server/.venv/bin/python -m pytest Application/Server/tests -q
npm run build --prefix Application/Client
```

In a separate process with its own storage, run the real inference smoke. Never run it against the active server's storage.

```sh
MRI_DEVICE=cpu MRI_STORAGE_DIR=/tmp/neuroscope-smoke Application/Server/.venv/bin/python Application/Server/smoke.py --sample BraTS2021_00495
```

Drive http://127.0.0.1:5173 with browser automation. Click Run analysis, wait for .status-pill.completed, select Prediction using the Overlay label, change Plane to coronal and confirm Grad-CAM is disabled, then choose axial and enable Grad-CAM. Wait for the MRI explorer image to load. Check both download links. Reload and select the analysis through Analysis history. Switch to a 390 pixel viewport and check for horizontal overflow. Also choose Upload scans and verify the four required file fields and optional reference field.

## Evidence

Save screenshots, console errors, API results, and generated masks under .audit/verify/. Confirm a downloaded mask has the original dimensions and labels 0, 1, 2, 4. Check completed SQLite records and report files in the isolated storage. Use real model outputs. A successful build alone does not prove the UI or inference works.

## Cleanup

Stop only the server and frontend processes this run started. Remove only the scratch storage this run created. Keep .audit/verify evidence. Do not delete Application/Models, Application/SampleCases, or the user's analysis history.

## Helpers

Application/Server/smoke.py exercises model loading, sample submission, completion, output downloads, all three overlay planes, and Grad-CAM through the real API. Its timeout defaults to 1800 seconds.
