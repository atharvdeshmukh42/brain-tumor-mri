"""Shared test fixtures: small synthetic BraTS-like patients (no real data needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))


def synthetic_patient(shape=(64, 72, 48), seed: int = 0, with_tumor: bool = True):
    """A skull-stripped 'brain' (ellipsoid of integer intensities) with a nested tumor: ET inside NCR inside ED.

    Returns image (4, X, Y, Z) float32, internal labels (X, Y, Z) uint8, affine (4, 4), spacing (3,).
    """
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    centre = np.array(shape) / 2
    brain = ((x - centre[0]) / (shape[0] * 0.38)) ** 2 + ((y - centre[1]) / (shape[1] * 0.4)) ** 2 \
        + ((z - centre[2]) / (shape[2] * 0.4)) ** 2 <= 1
    labels = np.zeros(shape, np.uint8)
    if with_tumor:
        tumor_centre = centre + np.array([6, -4, 3])
        distance = np.sqrt((x - tumor_centre[0]) ** 2 + (y - tumor_centre[1]) ** 2 + (z - tumor_centre[2]) ** 2)
        labels[(distance <= 9) & brain] = 2      # edema
        labels[(distance <= 6) & brain] = 1      # necrotic core
        labels[(distance <= 3) & brain] = 3      # enhancing tumor
    image = np.zeros((4,) + shape, np.float32)
    for c, level in enumerate((400, 300, 500, 600)):
        channel = rng.normal(level, 40, size=shape).round()
        channel += 250 * (labels > 0) + 150 * (labels == 3)
        image[c] = np.where(brain, np.clip(channel, 1, None), 0)
    affine = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine[:3, 3] = [90.0, 126.0, -72.0]
    spacing = np.array([1.0, 1.0, 1.0], np.float32)
    return image, labels, affine, spacing


@pytest.fixture
def patient():
    return synthetic_patient()


@pytest.fixture
def nifti_files(tmp_path):
    """The synthetic patient written as BraTS-style NIfTI files (labels stored as 0, 1, 2, 4)."""
    import nibabel as nib
    image, labels, affine, _ = synthetic_patient()
    files = {}
    for c, modality in enumerate(('flair', 't1', 't1ce', 't2')):
        files[modality] = tmp_path / f'P_{modality}.nii.gz'
        nib.save(nib.Nifti1Image(image[c].astype(np.int16), affine), files[modality])
    brats_labels = np.array([0, 1, 2, 4], np.uint8)[labels]
    files['seg'] = tmp_path / 'P_seg.nii.gz'
    nib.save(nib.Nifti1Image(brats_labels, affine), files['seg'])
    return files
