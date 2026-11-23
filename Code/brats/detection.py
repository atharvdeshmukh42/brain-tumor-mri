"""
Tumor detection: an ImageNet-pretrained EfficientNet that looks at one axial slice (4 modalities as 4 channels)
and outputs one logit: is there tumor on this slice?

Inputs are full 240 x 240 axial slices: the brain crop is put back at its original position, so the model sees
the brain where it really is in the scan.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .case import PreparedCase, normalize_image, region_masks
from .config import DETECTION_CANVAS
from .fileio import load_json
from .training import WorkerRandom, autocast, load_model_weights

# Short project names -> timm model names.
DETECTION_MODELS = {
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_b1': 'efficientnet_b1',
    'efficientnet_b2': 'efficientnet_b2',
    'efficientnet_b3': 'efficientnet_b3',
    'efficientnetv2_s': 'tf_efficientnetv2_s',
    'efficientnetv2_m': 'tf_efficientnetv2_m',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50',
}


def build_detection_model(name: str = 'efficientnet_b0', in_chans: int = 4, pretrained: bool = True,
                          drop_rate: float = 0.2) -> tuple[torch.nn.Module, bool]:
    """Binary tumor / no-tumor classifier with one output logit. Returns (model, imagenet_weights_loaded)."""
    import timm
    timm_name = DETECTION_MODELS.get(name, name)
    if pretrained:
        try:
            model = timm.create_model(timm_name, pretrained=True, in_chans=in_chans, num_classes=1,
                                      drop_rate=drop_rate)
            return model, True
        except Exception as error:      # no internet / hub problem: train from scratch instead of failing
            print(f'WARNING: could not load ImageNet weights for {timm_name} ({error}). Training from scratch.')
    model = timm.create_model(timm_name, pretrained=False, in_chans=in_chans, num_classes=1, drop_rate=drop_rate)
    return model, False


# -----------------------------------------------------------------------------
# Model inputs
# -----------------------------------------------------------------------------
def _center_fit(array: np.ndarray, size: int) -> np.ndarray:
    """Center-crop or zero-pad the last two axes to size x size."""
    h, w = array.shape[-2:]
    out = np.zeros(array.shape[:-2] + (size, size), dtype=array.dtype)
    keep_h, keep_w = min(h, size), min(w, size)
    src_h, src_w = (h - keep_h) // 2, (w - keep_w) // 2
    dst_h, dst_w = (size - keep_h) // 2, (size - keep_w) // 2
    out[..., dst_h:dst_h + keep_h, dst_w:dst_w + keep_w] = array[..., src_h:src_h + keep_h, src_w:src_w + keep_w]
    return out


def to_canvas(slices: np.ndarray, bbox: np.ndarray, orig_shape, size: int = DETECTION_CANVAS) -> np.ndarray:
    """Place cropped slices (K, C, x, y) back at their position in the full axial slice (K, C, size, size)."""
    k, c = slices.shape[:2]
    h, w = int(orig_shape[0]), int(orig_shape[1])
    canvas = np.zeros((k, c, h, w), dtype=np.float32)
    x0, x1, y0, y1 = (int(v) for v in bbox[:4])
    canvas[:, :, x0:x1, y0:y1] = slices
    if (h, w) != (size, size):
        canvas = _center_fit(canvas, size)
    return canvas


def detection_inputs(case: PreparedCase, z_indices) -> np.ndarray:
    """Normalized model inputs (K, 4, 240, 240) for the crop-slice indices z_indices of a case."""
    z_indices = np.asarray(z_indices, dtype=np.int64)
    raw = case.image[..., z_indices]                                          # (C, x, y, K)
    normalized = normalize_image(raw, case.norm_mean, case.norm_std)
    return to_canvas(np.moveaxis(normalized, -1, 0), case.bbox, case.orig_shape)


def detection_masks(case: PreparedCase, z_indices, region: str = 'WT') -> np.ndarray:
    """Expert tumor masks (K, 240, 240), aligned with detection_inputs."""
    z_indices = np.asarray(z_indices, dtype=np.int64)
    mask = region_masks(case.labels[..., z_indices])[region].astype(np.float32)      # (x, y, K)
    return to_canvas(np.moveaxis(mask, -1, 0)[:, None], case.bbox, case.orig_shape)[:, 0] > 0.5


def augment_slices(slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random flips, intensity scale/shift, noise and small shifts for (K, C, H, W) slices (modified in place)."""
    for i in range(slices.shape[0]):
        s = slices[i]
        if rng.random() < 0.5:
            s = s[:, ::-1, :]
        if rng.random() < 0.5:
            s = s[:, :, ::-1]
        brain = np.any(s != 0, axis=0)
        if rng.random() < 0.8:
            scale = rng.uniform(0.9, 1.1, size=(s.shape[0], 1, 1)).astype(np.float32)
            shift = rng.uniform(-0.1, 0.1, size=(s.shape[0], 1, 1)).astype(np.float32)
            s = np.where(brain[None], s * scale + shift, 0.0)
        if rng.random() < 0.2:
            s = np.where(brain[None], s + rng.normal(0, 0.05, size=s.shape).astype(np.float32), 0.0)
        if rng.random() < 0.5:
            dx, dy = rng.integers(-8, 9, size=2)
            s = np.roll(s, (int(dx), int(dy)), axis=(1, 2))
        slices[i] = s
    return slices


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def detection_slice_table(slices_df: pd.DataFrame, case_ids, min_brain_px: int = 500,
                          pos_min_px: int = 1) -> pd.DataFrame:
    """Eligible axial slices (enough brain visible) of the given patients, with their tumor / no-tumor label."""
    table = slices_df[slices_df['case_id'].isin(set(case_ids)) & (slices_df['brain_px'] >= min_brain_px)].copy()
    table['label'] = (table['wt_px'] >= pos_min_px).astype(np.int64)
    return table.reset_index(drop=True)


class DetectionDataset(torch.utils.data.Dataset):
    """One item = one patient. Training: `slices_per_case` random slices; evaluation: all eligible slices.

    Each item is (inputs float16 (K, 4, 240, 240), labels float32 (K,), info dict of per-slice arrays).
    Loading one patient file per item (instead of one per slice) keeps the disk reads low.
    """

    INFO_COLUMNS = ('z', 'z_orig', 'label', 'wt_px', 'brain_px')

    def __init__(self, table: pd.DataFrame, cases_dir: Path, slices_per_case: int | None = None,
                 repeats: int = 1, augment: bool = False):
        self.cases_dir = Path(cases_dir)
        self.slices_per_case = slices_per_case
        self.repeats = int(repeats)
        self.augment = augment
        self.slices_by_case = OrderedDict(
            (case_id, {column: group[column].to_numpy() for column in self.INFO_COLUMNS})
            for case_id, group in table.groupby('case_id', sort=True))
        self.case_ids = list(self.slices_by_case)
        self._random = WorkerRandom()

    def __len__(self) -> int:
        return len(self.case_ids) * self.repeats

    def __getitem__(self, index: int):
        case_id = self.case_ids[index % len(self.case_ids)]
        slices = self.slices_by_case[case_id]
        picked = np.arange(len(slices['z']))
        if self.slices_per_case is not None:
            picked = self._random.get().choice(picked, size=self.slices_per_case,
                                               replace=len(picked) < self.slices_per_case)
        case = PreparedCase.load(self.cases_dir / f'{case_id}.npz')
        inputs = detection_inputs(case, slices['z'][picked])
        if self.augment:
            inputs = augment_slices(inputs, self._random.get())
        info = {'case_id': np.array([case_id] * len(picked)), 'z': slices['z'][picked],
                'z_orig': slices['z_orig'][picked], 'wt_px': slices['wt_px'][picked],
                'brain_px': slices['brain_px'][picked]}
        return (torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float16)),
                torch.from_numpy(slices['label'][picked].astype(np.float32)), info)


def collate_patients(batch):
    """Concatenate the slices of several patients into one batch of slices."""
    inputs = torch.cat([item[0] for item in batch])
    labels = torch.cat([item[1] for item in batch])
    info = {key: np.concatenate([item[2][key] for item in batch]) for key in batch[0][2]}
    return inputs, labels, info


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, inputs: torch.Tensor | np.ndarray, device: torch.device,
                          amp: bool = True, chunk: int = 128) -> np.ndarray:
    """Tumor probability for each slice of inputs (K, 4, H, W)."""
    model.eval()
    if not torch.is_tensor(inputs):
        inputs = torch.from_numpy(np.asarray(inputs))
    probs = []
    for start in range(0, len(inputs), chunk):
        batch = inputs[start:start + chunk].to(device, non_blocking=True).float()
        with autocast(device, amp):
            logits = model(batch).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)


def predict_loader(model: torch.nn.Module, loader, device: torch.device, amp: bool = True,
                   desc: str = 'predict') -> pd.DataFrame:
    """One row per slice the loader yields: case_id, z, z_orig, wt_px, brain_px, label, prob."""
    tables = []
    for inputs, labels, info in tqdm(loader, desc=desc, leave=False):
        table = pd.DataFrame(dict(info))
        table['label'] = labels.numpy().astype(int)
        table['prob'] = predict_probabilities(model, inputs, device, amp)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def load_detector(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Trained detection model (eval mode) and its model card from a run folder."""
    weights = load_model_weights(Path(run_dir) / 'checkpoints' / 'best.pt', device)
    model, _ = build_detection_model(weights['model_name'], in_chans=4, pretrained=False)
    model.load_state_dict(weights['model'])
    return model.to(device).eval(), load_json(Path(run_dir) / 'model_card.json')


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def patient_detection_summary(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Per patient: is the tumor found, and do the predicted tumor slices match the true ones?"""
    rows = []
    for case_id, slices in predictions.groupby('case_id'):
        predicted_positive = slices['prob'] >= threshold
        true_z = set(slices.loc[slices['label'] == 1, 'z_orig'])
        pred_z = set(slices.loc[predicted_positive, 'z_orig'])
        union = true_z | pred_z
        rows.append({'case_id': case_id, 'n_slices': len(slices), 'n_true_tumor_slices': len(true_z),
                     'n_pred_tumor_slices': len(pred_z), 'max_prob': float(slices['prob'].max()),
                     'tumor_present_true': bool(true_z), 'tumor_present_pred': bool(pred_z),
                     'slice_accuracy': float((predicted_positive.astype(int) == slices['label']).mean()),
                     'slice_set_iou': len(true_z & pred_z) / len(union) if union else 1.0})
    return pd.DataFrame(rows)


def compare_detection_runs(detection_runs_dir: Path, quick_run: bool) -> pd.DataFrame:
    """One row per finished detection run (same quick/full mode) with its test metrics, size and speed."""
    rows = []
    for summary_path in sorted(Path(detection_runs_dir).glob('*/summary.json')):
        summary = load_json(summary_path)
        if summary.get('quick_run') != quick_run:
            continue
        test = summary['metrics']['test']
        rows.append({'run': summary['run_name'], 'model': summary['model_name'],
                     'params_M': summary['parameters'] / 1e6, 'test_auc': test['auc'], 'test_f1': test['f1'],
                     'test_accuracy': test['accuracy'], 'test_sensitivity': test['sensitivity'],
                     'test_specificity': test['specificity'], 'train_hours': summary['training_hours'],
                     'gpu_ms_per_slice': summary['inference'].get('cuda_ms_per_slice', np.nan),
                     'cpu_ms_per_slice': summary['inference'].get('cpu_ms_per_slice', np.nan)})
    return pd.DataFrame(rows)
