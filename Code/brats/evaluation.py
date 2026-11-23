"""
Evaluation of the segmentation model (notebook 03): the resumable per-patient segmentation + localization
evaluation, split summaries, and the comparison table of finished segmentation runs.

(Detection evaluation helpers live in detection.py, so the detection notebook does not need MONAI.)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .case import region_masks, save_label_nifti, uncrop
from .config import REGIONS
from .fileio import load_json, save_json, save_npz
from .localization import describe_location, localization_errors, localize_mask
from .metrics import label_confusion, segmentation_metrics
from .segmentation import SegCaseDataset, predict_volume, probs_to_labels, single_item_collate
from .training import make_loader


# -----------------------------------------------------------------------------
# Segmentation + localization
# -----------------------------------------------------------------------------
def evaluate_segmentation_split(model: torch.nn.Module, case_ids: list[str], cases_dir: Path, out_dir: Path,
                                device: torch.device, *, roi, sw_batch: int, overlap: float, amp: bool,
                                threshold: float, et_min_voxels: int, save_probabilities: bool,
                                num_workers: int, split_name: str) -> None:
    """Predict, measure and localize every patient; each patient is saved as soon as it is done (resumable).

    Writes for each patient in out_dir:
        masks/<id>_pred.nii.gz   predicted labels in the original scan grid (BraTS values 1, 2, 4)
        probs/<id>_probs.npz     region probabilities inside the crop (if save_probabilities)
        metrics/<id>.json        Dice / IoU / HD95 / ..., voxel confusion, localization of prediction and truth
    """
    out_dir = Path(out_dir)
    todo = [case_id for case_id in case_ids if not (out_dir / 'metrics' / f'{case_id}.json').is_file()]
    print(f'{split_name}: {len(case_ids) - len(todo)} done earlier, {len(todo)} to do')
    if not todo:
        return
    loader = make_loader(SegCaseDataset(todo, cases_dir), 1, False, num_workers, device, single_item_collate)
    for sample in tqdm(loader, total=len(todo), desc=split_name):
        start = time.time()
        probs = predict_volume(model, sample.image, device, roi, sw_batch, overlap, amp)
        inference_seconds = time.time() - start
        pred_labels = probs_to_labels(probs, threshold, et_min_voxels)
        true_labels = sample.labels
        pred_masks, true_masks = region_masks(pred_labels), region_masks(true_labels)
        loc_pred = {r: localize_mask(pred_masks[r], sample.affine, sample.spacing, sample.brain, sample.crop_origin)
                    for r in REGIONS}
        loc_true = {r: localize_mask(true_masks[r], sample.affine, sample.spacing, sample.brain, sample.crop_origin)
                    for r in REGIONS}
        save_label_nifti(uncrop(pred_labels, sample.bbox, sample.orig_shape), sample.affine,
                         out_dir / 'masks' / f'{sample.case_id}_pred.nii.gz')
        if save_probabilities:
            save_npz(out_dir / 'probs' / f'{sample.case_id}_probs.npz', probs=probs.astype(np.float16),
                     bbox=sample.bbox, orig_shape=sample.orig_shape, regions=np.array(REGIONS))
        save_json({'case_id': sample.case_id, 'split': split_name,
                   'metrics': segmentation_metrics(pred_labels, true_labels, sample.spacing),
                   'voxel_confusion': label_confusion(pred_labels, true_labels, sample.brain).tolist(),
                   'loc_pred': loc_pred, 'loc_gt': loc_true,
                   'loc_errors': {r: localization_errors(loc_pred[r], loc_true[r]) for r in REGIONS},
                   'report_pred': describe_location(loc_pred['WT']), 'report_gt': describe_location(loc_true['WT']),
                   'inference_seconds': inference_seconds},
                  out_dir / 'metrics' / f'{sample.case_id}.json')


LOCATION_FIELDS = ('volume_cm3', 'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm', 'extent_x_mm', 'extent_y_mm',
                   'extent_z_mm', 'side', 'ap', 'si', 'n_components')


def collect_segmentation_split(out_dir: Path, split_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Table of all evaluated patients of a split (one row each) and the summed 4 x 4 voxel confusion."""
    rows, confusion = [], np.zeros((4, 4), np.int64)
    for metrics_path in sorted((Path(out_dir) / 'metrics').glob('*.json')):
        result = load_json(metrics_path)
        row = {'case_id': result['case_id'], 'split': split_name, **result['metrics'],
               'inference_seconds': result['inference_seconds'],
               'report_pred': result['report_pred'], 'report_gt': result['report_gt']}
        for region in REGIONS:
            for key, value in result['loc_errors'][region].items():
                row[f'{key}_{region}'] = value
            for key in LOCATION_FIELDS:
                row[f'pred_{key}_{region}'] = result['loc_pred'][region].get(key)
                row[f'gt_{key}_{region}'] = result['loc_gt'][region].get(key)
        rows.append(row)
        confusion += np.array(result['voxel_confusion'])
    return pd.DataFrame(rows), confusion


def summarize_segmentation(results: pd.DataFrame) -> dict[str, float]:
    """Mean / median of every metric per region, localization errors, and the mean Dice over the 3 regions."""
    out = {}
    for region in REGIONS:
        for metric in ('dice', 'iou', 'hd95', 'sensitivity', 'precision'):
            values = results[f'{metric}_{region}'].astype(float)
            out[f'{metric}_{region}_mean'], out[f'{metric}_{region}_median'] = float(values.mean()), float(values.median())
        out[f'centroid_error_mm_{region}_median'] = float(results[f'centroid_error_mm_{region}'].astype(float).median())
        out[f'bbox_iou_{region}_mean'] = float(results[f'bbox_iou_{region}'].astype(float).mean())
    out['mean_dice'] = float(np.mean([out[f'dice_{region}_mean'] for region in REGIONS]))
    out['side_agreement_WT'] = float(pd.to_numeric(results['side_agree_WT'], errors='coerce').mean())
    return out


def compare_segmentation_runs(segmentation_runs_dir: Path, quick_run: bool) -> pd.DataFrame:
    """One row per finished segmentation run (same quick/full mode) with its test Dice, HD95, size and speed."""
    rows = []
    for summary_path in sorted(Path(segmentation_runs_dir).glob('*/summary.json')):
        summary = load_json(summary_path)
        if summary.get('quick_run') != quick_run or 'test' not in summary.get('results', {}):
            continue
        test = summary['results']['test']
        rows.append({'run': summary['run_name'], 'model': summary['model_name'],
                     'params_M': summary['parameters'] / 1e6,
                     **{f'dice_{region}': test[f'dice_{region}_mean'] for region in REGIONS},
                     'mean_dice': test['mean_dice'], 'hd95_WT': test['hd95_WT_mean'],
                     'train_hours': summary['training_hours'],
                     'gpu_s_per_case': summary['inference'].get('cuda_seconds_per_case_mean', np.nan),
                     'cpu_s_per_case': summary['inference'].get('cpu_seconds_one_case', np.nan)})
    return pd.DataFrame(rows)
