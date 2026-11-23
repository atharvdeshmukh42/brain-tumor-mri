"""
Run both trained models on new scans: detection, segmentation, localization and a report.

This is the code path of the offline application (and of predict_case.py). It expects BraTS-style input:
4 co-registered, skull-stripped modalities (FLAIR, T1, T1ce, T2) as NIfTI files.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .case import PreparedCase, prepare_case_from_files, region_masks, save_label_nifti, uncrop
from .config import REGIONS
from .detection import detection_inputs, load_detector, predict_probabilities
from .fileio import save_json
from .localization import describe_location, localize_mask
from .metrics import segmentation_metrics
from .segmentation import load_segmenter, predict_volume, probs_to_labels


def detect_tumor_slices(model: torch.nn.Module, card: dict, case: PreparedCase, device: torch.device,
                        batch: int = 32) -> dict:
    """Tumor probability for every axial slice that shows enough brain, and the slices above the threshold."""
    brain = case.brain_mask()
    z0 = case.crop_origin[2]
    z_indices = [z for z in range(brain.shape[2]) if brain[:, :, z].sum() >= card['min_brain_pixels']]
    probs = np.concatenate([
        predict_probabilities(model, detection_inputs(case, z_indices[start:start + batch]), device, amp=False)
        for start in range(0, len(z_indices), batch)]) if z_indices else np.zeros(0)
    tumor_slices = [int(z + z0) for z, p in zip(z_indices, probs) if p >= card['threshold']]
    return {'tumor_detected': bool(tumor_slices), 'max_probability': float(probs.max()) if probs.size else 0.0,
            'threshold': card['threshold'], 'n_slices_checked': len(z_indices), 'tumor_slices': tumor_slices,
            'slice_probabilities': {int(z + z0): float(p) for z, p in zip(z_indices, probs)}}


def segment_tumor(model: torch.nn.Module, card: dict, case: PreparedCase, device: torch.device
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Region probabilities (3, x, y, z) and internal labels (x, y, z) inside the brain crop."""
    probs = predict_volume(model, case.normalized(), device, roi=card['roi'], sw_batch=2,
                           overlap=card['sw_overlap'], amp=False)
    return probs, probs_to_labels(probs, card['threshold'], card['et_min_voxels'])


class TumorAnalyzer:
    """Both trained models, loaded once; call analyze() for each patient."""

    def __init__(self, detection_run: Path, segmentation_run: Path, device: torch.device):
        self.device = device
        self.detector, self.detector_card = load_detector(detection_run, device)
        self.segmenter, self.segmenter_card = load_segmenter(segmentation_run, device)

    def analyze(self, files: dict[str, Path], out_dir: Path | None = None) -> dict:
        """Full analysis of one patient. With out_dir, also writes report.json, slice_probabilities.json and
        tumor_mask.nii.gz (predicted labels in the original scan grid, BraTS values 1 / 2 / 4)."""
        start = time.time()
        case = prepare_case_from_files(files, require_seg=False)
        detection = detect_tumor_slices(self.detector, self.detector_card, case, self.device)
        _, labels = segment_tumor(self.segmenter, self.segmenter_card, case, self.device)
        masks = region_masks(labels)
        brain = case.brain_mask()
        localization = {region: localize_mask(masks[region], case.affine, case.spacing, brain, case.crop_origin)
                        for region in REGIONS}
        report = {
            'input': {key: str(path) for key, path in files.items()},
            'device': str(self.device),
            'detection': {key: value for key, value in detection.items() if key != 'slice_probabilities'},
            'localization': localization,
            'summary': describe_location(localization['WT']),
            'seconds': round(time.time() - start, 1),
        }
        if case.has_labels:
            report['comparison_with_expert'] = segmentation_metrics(labels, case.labels, case.spacing)
        if out_dir is not None:
            out_dir = Path(out_dir)
            save_label_nifti(uncrop(labels, case.bbox, case.orig_shape), case.affine, out_dir / 'tumor_mask.nii.gz')
            save_json(report, out_dir / 'report.json')
            save_json(detection['slice_probabilities'], out_dir / 'slice_probabilities.json')
        return report
