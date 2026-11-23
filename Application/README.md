# NeuroScope MRI research workstation

Local detection, 3D segmentation, localization, and axial Grad-CAM using the trained EfficientNet-B0 and SegResNet models in `Code/brats`. This is a research application, not a diagnostic medical device. Inputs must already be co-registered, skull-stripped, preprocessed four-modality MRI volumes. The app does not perform registration or skull stripping.

## Setup

Use Python 3.11 and Node.js 20.19+ (or 22.12+). Run from the repository root.

```sh
python3.11 -m venv Application/Server/.venv
Application/Server/.venv/bin/python -m pip install -r Application/Server/requirements.lock.txt
npm ci --prefix Application/Client
cp Application/.env.example Application/.env
```

Replace `python3.11` with the path to your Python 3.11 executable if needed. The repository includes both model checkpoints and two [BraTS sample cases](SampleCases/README.md) in the layout below. You can start the app without downloading data or retraining. Keep `Code/brats/` alongside `Application/` at the repository root because the backend imports it.

```text
Application/
  Models/
    effb0/{best.pt,config.json,model_card.json,summary.json}
    segresnet/{best.pt,config.json,model_card.json,summary.json}
  SampleCases/
    BraTS2021_00495/BraTS2021_00495_{flair,t1,t1ce,t2,seg}.nii.gz
    BraTS2021_00621/BraTS2021_00621_{flair,t1,t1ce,t2,seg}.nii.gz
```

`{...}` denotes separate files, not a literal filename. Alternatively set absolute asset directories in `Application/.env`. `MRI_DEVICE=auto` chooses CUDA, then Apple MPS, then CPU. For an unsupported accelerator operation or insufficient device memory, restart with `MRI_DEVICE=cpu` and create a new analysis. An explicitly unavailable device produces a clear health error; the server never silently changes a requested device. Only load trusted local model checkpoints.

## Start

Terminal 1, from the repository root:

```sh
Application/Server/.venv/bin/python -m uvicorn main:app --app-dir Application/Server --host 127.0.0.1 --port 8000
```

Terminal 2, from the repository root:

```sh
npm run dev --prefix Application/Client
```

Open http://127.0.0.1:5173. Wait for the model status to become ready, then choose a sample or upload `.nii` / `.nii.gz` files. Uploads require FLAIR, T1, T1ce, T2 and optionally a reference segmentation with labels 0, 1, 2, 4. All files must share their 3D shape and affine, declare millimeter spatial units, and contain finite data. Limits are 128 MiB compressed per file, 256 MiB expanded per file, and 32 million voxels per volume. Processing can take several minutes on CPU. Use a single API process, without `--workers` or `--reload`, so one worker owns the models and job queue.

The Vite development server proxies `/api` to port 8000. CORS permits only `localhost:5173` and `127.0.0.1:5173`. Both services bind to loopback. This local application has no authentication and is not intended for network deployment.

If port 8000 is already occupied, start the backend with `--port 8010` and the frontend with `VITE_API_TARGET=http://127.0.0.1:8010 npm run dev --prefix Application/Client`. Keep the frontend on port 5173.

## Viewing and outputs

The viewer displays original voxel axes, not canonical anatomical reorientation. Plane names assume the usual BraTS grid; the displayed NIfTI orientation codes identify each stored axis. The slice slider supports keyboard arrows. Overlay colors show label 2 edema, label 1 necrotic core, and label 4 enhancing tumor. WT contains TC, and TC contains ET, so region volumes are overlapping measurements.

Predicted masks and reports become available after completion. Grad-CAM uses the detection model for the selected axial slice and is supported for the model's 240 × 240 BraTS canvas. It shows model attention, not tumor boundaries. The displayed probability is the maximum detector probability across checked slices; it is not patient-level clinical risk. Segmentation runs independently of the detection threshold.

SQLite history, uploaded inputs, masks, reports, and cached explanations live in `Application/Server/storage` by default. Each job has a generated UUID directory. Restarted queued/running jobs become failed with an interruption explanation; completed jobs remain viewable. No automatic deletion occurs. Close the server before backing up the storage directory. Keep it private because it contains scan data. JSON reports omit original absolute input paths.

## Verification

```sh
Application/Server/.venv/bin/python -m pytest Application/Server/tests -q
npm run build --prefix Application/Client
Application/Server/.venv/bin/python Application/Server/smoke.py --sample BraTS2021_00495
```

Regular tests cover API validation, geometry, non-finite values, gzip expansion limits, safe filenames, empty optional upload fields, millimeter units, full-volume expert comparisons, queue saturation, cancelled explanations, CORS, durable history/restart recovery, PNG planes, and output downloads. They use a non-inference queue adapter and never load model weights. The separate smoke command loads the real models, processes a sample, verifies report/mask downloads, renders predicted overlays in all three planes, and computes real Grad-CAM. It persists an analysis and prints its report; default timeout is 1800 seconds.

## API

Interactive schema at http://127.0.0.1:8000/docs.

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Model state, device, initialization error |
| `GET /api/samples` | Discover available complete sample cases |
| `POST /api/analyses/sample` | JSON `{"sample_id":"BraTS2021_00495"}` |
| `POST /api/analyses/upload` | Multipart `flair`, `t1`, `t1ce`, `t2`, optional `seg` |
| `GET /api/analyses` | Latest 200 persisted analyses |
| `GET /api/analyses/{id}` | Status, metadata, report, error |
| `GET /api/analyses/{id}/slice` | PNG query `modality`, `plane`, `index` (zero-based), `overlay`, `opacity`, `gradcam` |
| `GET /api/analyses/{id}/download/report` | JSON report |
| `GET /api/analyses/{id}/download/mask` | Original-grid BraTS label NIfTI |

Analysis states are `queued`, `running`, `completed`, and `failed`. Model loading errors and failed analyses remain explicit. Cached symlinks adapt flat model folders to the existing shared package's `checkpoints/best.pt` layout without modifying the original assets or training code.
