# Verification record

Verified on 2026-11-23 with Python 3.11.14 on Apple Silicon and Node.js 25.3.0.

- `python -m pytest Application/Server/tests -q`: 5 passed. The Starlette test client emits an upstream AnyIO deprecation warning.
- `npm run build --prefix Application/Client`: TypeScript and Vite production build passed.
- `ruff check Application/Server/*.py Application/Server/tests/*.py`: passed.
- Real CPU smoke against `BraTS2021_00495`: completed. Job `2ec7a994-fda3-4aab-9570-9e533b05d9e2` produced a JSON report and original-grid NIfTI mask. The script verified both downloads, predicted-overlay PNGs in axial/coronal/sagittal planes, and a real axial Grad-CAM PNG.
- Live Chromium verification against a separate copy of the persisted real inference outputs: sample intake, all five upload fields, history selection, predicted overlay, coronal plane, disabled non-axial Grad-CAM, and history after reload passed. No page exceptions or mobile horizontal overflow occurred. Desktop and 390-pixel mobile screenshots were visually inspected.

The real smoke used the supplied model and sample directories at the original desktop location through `MRI_MODELS_DIR` and `MRI_SAMPLES_DIR`. No original assets or shared training code changed. The command is reproducible with the paths configured in `Application/.env`:

```sh
MRI_DEVICE=cpu Application/Server/.venv/bin/python Application/Server/smoke.py --sample BraTS2021_00495
```

The first concurrent verification exposed a store recovery defect. Recovery now runs only when the API lifespan starts, rather than whenever a Store instance opens. The history test proves opening another Store preserves queued work and explicit startup recovery marks it interrupted. The full real smoke then passed.

MPS/CUDA inference was not exercised. The second sample was discovered but not analyzed during this verification. Uploaded inputs were exercised through the real API validation suite; full model inference was exercised using the supplied sample. CPU inference is resource intensive. Use one backend process and allow several minutes.
