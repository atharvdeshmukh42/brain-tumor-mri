# Brain tumor detection, segmentation, localization and XAI (BraTS 2021)

Four Google Colab notebooks train and evaluate the models; everything they produce is saved to Google Drive.
You then download the trained weights and run them offline on the laptop.

| Task | Model | Notebook |
|---|---|---|
| Detection (is there tumor on this slice?) | **EfficientNet-B0**, ImageNet-pretrained, 2D axial slices | `02_detection_training.ipynb` |
| Segmentation (which voxels are WT / TC / ET?) | **SegResNet**, 3D residual U-Net-style network (BraTS 2018 winner) | `03_segmentation_localization.ipynb` |
| Localization (where, how big, which side?) | plain measurement code on the predicted mask | `03_segmentation_localization.ipynb` |
| Explainability | Grad-CAM, Grad-CAM++, occlusion, test-time-augmentation uncertainty | `04_xai.ipynb` |

## Folder contents

```
Code/
  01_data_preparation.ipynb           download, preprocess, split patients, 23 dataset graphs        (CPU runtime)
  02_detection_training.ipynb         train + evaluate EfficientNet-B0                                (T4 GPU)
  03_segmentation_localization.ipynb  train + evaluate SegResNet, localization of every patient       (T4 GPU)
  04_xai.ipynb                        heatmaps and uncertainty                                        (T4 GPU)
  predict_case.py                     run both trained models on one patient, offline (start of the app)
  requirements.txt                    packages for the laptop
  brats/                              the shared code, used by every notebook AND by the app
    config.py        constants: modalities, labels, regions, normalization settings
    fileio.py        safe writing to Google Drive (atomic writes, retries, verified copies, disk checks)
    env.py           Colab / Drive / GPU / seeds, project folders, reproducibility logs
    notebook.py      notebook start-up (installs missing packages)
    raw_data.py      Kaggle download, archive extraction
    case.py          one patient: NIfTI loading, cropping, normalization, the PreparedCase container
    dataset.py       the whole dataset: resumable sharded preprocessing, statistics, patient split, staging
    metrics.py       Dice, IoU, HD95, detection metrics, threshold choice, calibration
    localization.py  tumor position, size, volume, hemisphere
    training.py      mixed precision, learning-rate schedule, training epoch, resumable training runs
    detection.py     EfficientNet inputs, augmentation, datasets, prediction
    segmentation.py  SegResNet / U-Net, training patches, loss, sliding-window prediction, TTA
    evaluation.py    segmentation + localization evaluation of every patient, run comparison
    xai.py           Grad-CAM, Grad-CAM++, occlusion, entropy
    inference.py     both models on new scans -> report (used by predict_case.py and the app)
    plots.py         every chart helper, one visual style (PNG + PDF + CSV)
  tests/                              automated tests (run on the laptop, see section 7)
```

The shared package reads data prepared with the original version of the pipeline. The optional parity tests use the
original files if available at `../Code_v1_before_refactor/`; the archive is not part of the delivery.

---

## 1. One-time setup

1. In Google Drive create the folder **`MyDrive/BrainTumor/code/`** and upload into it **everything in this `Code`
   folder**: the 4 notebooks, `predict_case.py`, `requirements.txt` and the **whole `brats` folder** (drag the folder
   itself, so Drive has `code/brats/config.py`, `code/brats/case.py`, ...). The `tests` folder is not needed on Drive.
   If an older version is already there, delete `brats_utils.py` and `brats_plots.py` from `code/` (they are no longer used).
2. Kaggle access (only notebook 01 needs it): kaggle.com → profile → *Settings* → *API*.
   - **Create New Token** downloads `kaggle.json`: add the Colab secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`
     (key icon in the left Colab sidebar, switch on *Notebook access*), or upload the file to `MyDrive/BrainTumor/kaggle.json`.
   - A token starting with `KGAT_` goes into the Colab secret `KAGGLE_API_TOKEN`.
3. Open each notebook from Drive (double-click → *Open with Google Colaboratory*).

## 2. Run order (final run)

The notebooks are already set for the final run: `QUICK_RUN = False`, EfficientNet-B0 (`effb0_v1`) and SegResNet
(`segresnet_v1`, learning rate 5e-4). Nothing needs to be edited. Choose the runtime, then **Runtime → Run all**.

| Step | Notebook | Runtime | Approx. time | Check |
|---|---|---|---|---|
| 1 | 01 data | **CPU** | 1–2 h (includes the one-time ~13 GB download) | ends with `patients_ok` ≈ 1251 |
| 2 | 02 detection | **T4 GPU** | 5–7 h (early stopping may end it sooner) | validation AUC above 0.9 after a few epochs |
| 3 | 03 segmentation | **T4 GPU** | 10–12 h, over several sessions | epoch 1 finishes (no "CUDA out of memory"); mean validation Dice above 0.3 by epoch 10 |
| 4 | 04 XAI | **T4 GPU** | 20–40 min | `Saved .../summary.json` |

Times are estimates from the quick run on a free Colab T4 (2 CPU cores); the notebooks print the real time per epoch
and the time left.

**Smoke test:** set `QUICK_RUN = True` in a notebook to run it in a few minutes on 40 patients. Quick runs write to
`data_quick/` and `runs/*/<name>_quick/` and never mix with the real results.

## 3. If Colab disconnects

Nothing is lost except the epoch (or the 100-patient shard) that was running.

- Notebook 01 marks each finished shard on Drive and skips it next time.
- Notebooks 02 and 03 save the full training state (`last.pt`) after **every epoch**.
- Notebook 03's final evaluation saves each patient as soon as it is done.

To continue: open the same notebook, keep the same `RUN_NAME`, and choose **Runtime → Run all**. It prints
`RESUMING: epoch N was finished`.

- To train another model, change `MODEL_NAME` **and** `RUN_NAME` (e.g. `unet3d` / `unet3d_v1` for a comparison).
  Runs never overwrite each other, and both training notebooks end with a comparison graph of all finished runs.
- Do not change `EPOCHS` in the middle of a run: the learning-rate schedule depends on it.
- Do not upload new code to Drive while a run is in progress; a resumed run would use the new code.
- Keep the browser tab open (free Colab disconnects idle sessions and caps sessions at about 12 hours) and run the
  last cell (`env.flush_drive()`) before closing a notebook: it waits until every file is uploaded to Drive.

## 4. What is saved on Drive

```
MyDrive/BrainTumor/
  code/                      this folder
  data/
    raw/                     the original Kaggle download (kept, never downloaded again)
    preprocessed/            shard_000.tar ... (+ .done.json markers, per-shard statistics)
    meta/                    cases.csv, slices.csv, split.csv, case_list.csv, dataset_summary.json, ...
    figures/                 23 dataset graphs (PNG + PDF + CSV)
    logs/                    environment / package versions
  runs/
    detection/effb0_v1/
      checkpoints/           best.pt (for the app), last.pt (full state), epoch_001.pt, ...
      predictions/           every train/val/test slice with its probability, per-patient tables, metrics
      figures/               training curves, confusion matrices, ROC, PR, calibration, examples, ...
      history.csv  config.json  summary.json  model_card.json  logs/  code_snapshot/  export/
    segmentation/segresnet_v1/
      checkpoints/           best.pt, last.pt, epoch_XXX.pt
      predictions/val|test/  masks/*.nii.gz, probs/*.npz, metrics/*.json (+ localization per patient)
      figures/  history.csv  config.json  summary.json  model_card.json  logs/  code_snapshot/  export/
    xai/xai_v1/              heatmaps, uncertainty maps, quantitative CSVs, figures/
```

How the data is stored: each patient is cropped to the brain and kept as the original integer intensities (lossless,
compressed, about 5–10 MB per patient). The z-score normalization is applied on the fly, by the same function during
training and in the app. Patients are split **before** anything is trained, so slices of one patient never appear in
two splits.

## 5. Using the models offline on the laptop

Download only what the app needs. **Do not download `data/`**: it is tens of GB.

```
runs/detection/effb0_v1/          checkpoints/best.pt, config.json, model_card.json, summary.json
runs/segmentation/segresnet_v1/   checkpoints/best.pt, config.json, model_card.json, summary.json
code/                             brats/ folder, predict_case.py, requirements.txt
```

Keep the same folder layout (`runs/detection/effb0_v1/checkpoints/best.pt`, ...). For testing, a few patients from
`data/raw/` are enough (each patient is 4 NIfTI files plus a label file).

Install the packages once:

```bash
pip install -r requirements.txt
```

Run one patient (on Apple Silicon it uses the Mac GPU automatically):

```bash
python predict_case.py --flair P_flair.nii.gz --t1 P_t1.nii.gz --t1ce P_t1ce.nii.gz --t2 P_t2.nii.gz --det-run runs/detection/effb0_v1 --seg-run runs/segmentation/segresnet_v1 --out results/P
```

It writes `results/P/report.json` (detection result, tumor slices, position / size / volume per region),
`results/P/tumor_mask.nii.gz` (predicted mask in the original scan grid, BraTS labels 1 / 2 / 4) and
`results/P/slice_probabilities.json`.

The app should use `brats.inference.TumorAnalyzer` the same way `predict_case.py` does: load it once, then call
`analyze(files, out_dir)` for each patient. It expects the BraTS format: 4 co-registered, skull-stripped modalities.
Raw hospital scans would first need skull stripping and registration, which is outside this project's scope.

## 6. Settings you might change

| Notebook | Setting | Default | Notes |
|---|---|---|---|
| 02 | `MODEL_NAME` | `efficientnet_b0` | also `efficientnet_b2`, `efficientnetv2_s`, `efficientnetv2_m`, `resnet18`, `resnet50` |
| 02 | `EPOCHS`, `EARLY_STOP_PATIENCE` | 20, 6 | early stopping watches validation AUC |
| 02 | `THRESHOLD_METHOD` | `f1` | chosen on validation, kept between 0.05 and 0.95 |
| 03 | `MODEL_NAME` | `segresnet` | also `unet3d` (use `LR = 1e-3`), `unet3d_large`, `dynunet` (much slower) |
| 03 | `EPOCHS`, `LR` | 100, 5e-4 | if mean validation Dice is still near 0 at epoch 10, stop and lower `LR` |
| 03 | `BATCH_SIZE` | 2 | set 1 (with a new `RUN_NAME`) if the GPU runs out of memory |
| 03 | `ET_MIN_VOXELS` | 0 | e.g. 500 removes tiny false enhancing-tumor blobs (a common BraTS trick) |
| 04 | `SEG_RUN_NAME` | `segresnet_v1` | must match the run of notebook 03 |
| 04 | `N_UNCERTAINTY_CASES` | 20 | each patient is predicted 9 times |

## 7. Tests (on the laptop)

```bash
cd Code
pip install -r requirements.txt
python -m pytest tests
```

The tests use small synthetic patients, so no real data is needed. They check preprocessing, storage, metrics with
known answers, localization, model shapes, the training loop and resuming after a disconnect. The optional parity
tests skip if the original pipeline archive is absent.

## 8. Troubleshooting

| Message | Fix |
|---|---|
| `The "brats" folder was not found in .../code` | Upload the `brats` folder (not just its files) into `MyDrive/BrainTumor/code/` |
| `Kaggle credentials were not found` | See step 1.2 |
| `Kaggle download failed ... 403` | Open the dataset page on kaggle.com while logged in, accept the terms, run again |
| `Not enough local disk` | Notebook 01: use a CPU runtime. Notebooks 02–04 need about 15 GB free |
| `No GPU` | *Runtime → Change runtime type → T4 GPU* |
| `pip changed ... Restart session` | *Runtime → Restart session*, then *Run all* |
| `... is missing - run notebook 01 (with the same QUICK_RUN setting)` | Notebooks 02–04 read the data of the matching quick/full run |
| `Run "X" was trained with ...` | You changed the model but kept the name. Pick a new `RUN_NAME` |
| CUDA out of memory (03) | `BATCH_SIZE = 1` (or `ROI = (112, 112, 112)`) with a new `RUN_NAME`; set the same name in notebook 04 |
| Code changes on Drive are not picked up | *Runtime → Restart session*, then *Run all* (Python keeps imported code in memory) |
