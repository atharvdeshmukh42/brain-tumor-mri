"""Evaluation metrics: segmentation overlap and distance, detection metrics, threshold choice, calibration."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .case import region_masks
from .config import HD95_EMPTY_PENALTY_MM, REGIONS


# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------
def dice_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """Dice overlap of two boolean masks; 1.0 when both are empty."""
    pred_sum, truth_sum = int(pred.sum()), int(truth.sum())
    if pred_sum + truth_sum == 0:
        return 1.0
    return 2.0 * int(np.logical_and(pred, truth).sum()) / (pred_sum + truth_sum)


def iou_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """Intersection over union (Jaccard) of two boolean masks; 1.0 when both are empty."""
    union = int(np.logical_or(pred, truth).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(pred, truth).sum()) / union


def hd95(pred: np.ndarray, truth: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff distance between the mask surfaces, in mm (BraTS convention for empty masks)."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        return HD95_EMPTY_PENALTY_MM
    # Work on the bounding box of both masks (+2 voxels) - much faster than the whole volume.
    occupied = np.argwhere(pred | truth)
    low = np.maximum(occupied.min(0) - 2, 0)
    high = occupied.max(0) + 3
    box = tuple(slice(a, b) for a, b in zip(low, high))
    pred, truth = pred[box], truth[box]
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=structure, border_value=0)
    truth_surface = truth ^ ndimage.binary_erosion(truth, structure=structure, border_value=0)
    dist_to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    return float(max(np.percentile(dist_to_truth[pred_surface], 95), np.percentile(dist_to_pred[truth_surface], 95)))


def segmentation_metrics(pred_labels: np.ndarray, true_labels: np.ndarray, spacing) -> OrderedDict:
    """Dice, IoU, sensitivity, precision, HD95 and volumes for WT / TC / ET."""
    voxel_cm3 = float(np.prod(spacing)) / 1000.0
    pred_masks, true_masks = region_masks(pred_labels), region_masks(true_labels)
    out = OrderedDict()
    for region in REGIONS:
        pred, truth = pred_masks[region], true_masks[region]
        true_positives = int(np.logical_and(pred, truth).sum())
        out[f'dice_{region}'] = dice_score(pred, truth)
        out[f'iou_{region}'] = iou_score(pred, truth)
        out[f'sensitivity_{region}'] = (true_positives / truth.sum() if truth.any()
                                        else (1.0 if not pred.any() else 0.0))
        out[f'precision_{region}'] = (true_positives / pred.sum() if pred.any()
                                      else (1.0 if not truth.any() else 0.0))
        out[f'hd95_{region}'] = hd95(pred, truth, spacing)
        out[f'vol_pred_{region}_cm3'] = float(pred.sum()) * voxel_cm3
        out[f'vol_gt_{region}_cm3'] = float(truth.sum()) * voxel_cm3
        out[f'gt_empty_{region}'] = bool(not truth.any())
    return out


def label_confusion(pred_labels: np.ndarray, true_labels: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """4 x 4 voxel counts (rows = true label, columns = predicted label), optionally inside mask."""
    if mask is not None:
        pred_labels, true_labels = pred_labels[mask], true_labels[mask]
    pairs = true_labels.astype(np.int64).ravel() * 4 + pred_labels.astype(np.int64).ravel()
    return np.bincount(pairs, minlength=16).reshape(4, 4)


# -----------------------------------------------------------------------------
# Detection (binary: tumor / no tumor)
# -----------------------------------------------------------------------------
def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> OrderedDict:
    """Confusion counts, accuracy, sensitivity, specificity, precision, F1, AUC and average precision."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float('nan')
    specificity = tn / (tn + fp) if tn + fp else float('nan')
    precision = tp / (tp + fp) if tp + fp else float('nan')
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (tp and precision + sensitivity) else 0.0
    out = OrderedDict(threshold=float(threshold), n=int(len(y)), positives=int(y.sum()),
                      tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / max(len(y), 1),
                      sensitivity=sensitivity, specificity=specificity, precision=precision, f1=f1,
                      balanced_accuracy=np.nanmean([sensitivity, specificity]))
    try:
        out['auc'] = float(roc_auc_score(y, prob))
        out['average_precision'] = float(average_precision_score(y, prob))
    except ValueError:          # only one class present
        out['auc'] = float('nan')
        out['average_precision'] = float('nan')
    return out


def weighted_bce(y: np.ndarray, prob: np.ndarray, pos_weight: float = 1.0, eps: float = 1e-7) -> float:
    """Binary cross-entropy from probabilities, with the same positive-class weight as the training loss.

    Matches torch.nn.BCEWithLogitsLoss(pos_weight=...), so training and validation loss curves are comparable.
    """
    y = np.asarray(y, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(pos_weight * y * np.log(prob) + (1 - y) * np.log(1 - prob)))


def choose_threshold(y: np.ndarray, prob: np.ndarray, method: str = 'f1',
                     min_threshold: float = 0.05, max_threshold: float = 0.95) -> float:
    """Pick the decision threshold on the VALIDATION set (never on test).

    method 'f1' maximises F1, 'youden' maximises sensitivity + specificity - 1. The result is kept inside
    [min_threshold, max_threshold]: an almost untrained model can otherwise produce absurd values such as 0.002.
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return 0.5
    if method == 'youden':
        fpr, tpr, thresholds = roc_curve(y, prob)
        best = float(thresholds[np.argmax(tpr - fpr)])
    elif method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y, prob)
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        best = float(thresholds[int(np.argmax(f1))])
    else:
        raise ValueError(f'Unknown threshold method "{method}" (use "f1" or "youden").')
    return float(np.clip(best, min_threshold, max_threshold))


def calibration_table(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Reliability table (mean predicted probability vs observed frequency per bin) and the expected calibration error."""
    y, prob = np.asarray(y), np.asarray(prob)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_index = np.clip(np.digitize(prob, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        rows.append({'bin_lo': edges[b], 'bin_hi': edges[b + 1], 'n': int(in_bin.sum()),
                     'mean_prob': float(np.mean(prob[in_bin])) if in_bin.any() else np.nan,
                     'frac_pos': float(np.mean(y[in_bin])) if in_bin.any() else np.nan})
    table = pd.DataFrame(rows)
    ece = float(np.nansum(table['n'] / max(len(y), 1) * np.abs(table['mean_prob'] - table['frac_pos'])))
    return table, ece
