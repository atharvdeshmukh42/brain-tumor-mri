# Verification record

Verified on 2026-11-23 with Python 3.11 and Node.js 25.3.0 on Apple Silicon.

- The locked backend environment passes all 10 API tests. Tests cover history/restart recovery, request validation, uploads without a reference, finite data, matching geometry, millimeter units, gzip limits, safe filenames, queue saturation, cancelled explanations, full-volume reference comparisons, overlays, and downloads.
- TypeScript and the Vite production build pass with Vite 6.4.6 and Zod 4.6.7. `npm audit` reports zero vulnerabilities.
- Ruff check and format check pass for Application/Server.
- Both model checkpoints load with strict architecture matching. Both supplied cases have aligned effective NIfTI affines, finite data, and valid BraTS reference labels.
- Real CPU inference completed for both BraTS2021_00495 and BraTS2021_00621. The API smoke verified JSON and mask downloads, predicted PNG overlays in three planes, and actual axial Grad-CAM. Masks retain the original 240 × 240 × 155 grid.
- A real Chrome interaction submitted a sample, waited for completion, selected a prediction overlay, changed planes, verified that non-axial Grad-CAM is disabled, enabled axial Grad-CAM, and reopened a completed analysis after reload. No page exceptions occurred.
- A second real browser interaction uploaded the four required scans without an optional reference and completed inference successfully.
- Desktop and 390-pixel mobile screenshots were inspected. The MRI image loads and the mobile layout has no horizontal overflow.

An independent GPT-6.2 Sol review reproduced edge cases in optional uploads, NIfTI units, reference comparison outside the brain crop, full queues, and request cancellation. Each has a focused regression check and a fix. The comment pass found three redundant comments/docstrings, which were removed. The reviewer could inspect local decision records but could not audit the parent runtime transcript.

Run the commands in README.md or use the project verification skill at ../.agents/skills/verify/SKILL.md. Real smoke tests must use their own MRI_STORAGE_DIR if another API process is running. Starting another API on the same storage recovers active jobs as interrupted.

MPS and CUDA inference remain unverified. The verified execution mode is MRI_DEVICE=cpu. Dependency deprecation warnings in the Python test environment and upstream tree-shaking comment warnings during the frontend build do not fail the checks.

The delivery snapshot was tested using only staged files. The frontend installed from the lockfile and built successfully. The shared Python and API suites passed 37 tests; 7 optional parity tests skipped because the archived pipeline is outside the delivery. Both packaged samples passed input validation, and a CPU smoke run on `BraTS2021_00495` verified inference, three overlay planes, Grad-CAM, and both downloads using the packaged checkpoints.
