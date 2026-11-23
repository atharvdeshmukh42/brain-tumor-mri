"""
One patient: loading the NIfTI files, labels, cropping to the brain, normalization.

`prepare_case` is used for training data AND by the offline app on new scans, so both see identical inputs.

Coordinate conventions
    Arrays are indexed (X, Y, Z) exactly as stored in the NIfTI files (BraTS: 240 x 240 x 155, 1 mm voxels).
    A "crop" is the brain bounding box (x0, x1, y0, y1, z0, z1) in that original grid.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (CROP_MARGIN, INTERNAL_TO_BRATS_LABEL, LABEL_EDEMA, LABEL_ENHANCING, LABEL_NECROTIC,
                     MODALITIES, NORM_CLIP, REGIONS)
from .fileio import atomic_replace, save_npz, with_retries

BBox = np.ndarray   # int32 array (x0, x1, y0, y1, z0, z1); the upper bounds are exclusive


# -----------------------------------------------------------------------------
# Loading NIfTI files
# -----------------------------------------------------------------------------
def _load_volume(path: Path, name: str):
    import nibabel as nib
    nii = nib.load(str(path))
    volume = np.asarray(nii.get_fdata(dtype=np.float32))
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f'{name}: expected a 3D volume, got shape {volume.shape}')
    return nii, volume


def load_nifti_case(files: dict[str, Path], require_seg: bool = True
                    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Load the 4 MRI modalities (+ label map) of one patient.

    Returns:
        image    (4, X, Y, Z) float32 in the original voxel grid
        labels   (X, Y, Z) uint8 with internal labels 0..3, or None when there is no label file
        affine   (4, 4) voxel -> world (mm) matrix
        spacing  (3,) voxel size in mm
    """
    channels, shape, affine, spacing = [], None, None, None
    for modality in MODALITIES:
        if modality not in files:
            raise FileNotFoundError(f'missing {modality} file')
        nii, volume = _load_volume(files[modality], modality)
        if shape is None:
            shape, affine = volume.shape, nii.affine.astype(np.float64)
            spacing = np.array(nii.header.get_zooms()[:3], dtype=np.float32)
        elif volume.shape != shape:
            raise ValueError(f'{modality}: shape {volume.shape} differs from {shape}')
        channels.append(np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0))
    image = np.stack(channels, 0)
    labels = None
    if 'seg' in files:
        _, raw_labels = _load_volume(files['seg'], 'seg')
        if raw_labels.shape != shape:
            raise ValueError(f'seg: shape {raw_labels.shape} differs from {shape}')
        labels = brats_to_internal_labels(raw_labels)
    elif require_seg:
        raise FileNotFoundError('missing seg file')
    return image, labels, affine, spacing


def brats_to_internal_labels(raw_labels: np.ndarray) -> np.ndarray:
    """BraTS labels {0, 1, 2, 4} (or already {0, 1, 2, 3}) -> internal {0, 1, 2, 3}."""
    rounded = np.rint(raw_labels).astype(np.int32)
    unexpected = set(np.unique(rounded).tolist()) - {0, 1, 2, 3, 4}
    if unexpected:
        raise ValueError(f'unexpected label values {sorted(unexpected)}')
    labels = rounded.astype(np.uint8)
    labels[rounded == 4] = LABEL_ENHANCING
    return labels


def internal_to_brats_labels(labels: np.ndarray) -> np.ndarray:
    """Internal labels {0, 1, 2, 3} -> BraTS labels {0, 1, 2, 4}."""
    return INTERNAL_TO_BRATS_LABEL[labels]


def save_label_nifti(labels_full: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    """Save an internal label map (original grid) as a BraTS-style NIfTI with values 0, 1, 2, 4."""
    import nibabel as nib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nii = nib.Nifti1Image(internal_to_brats_labels(labels_full).astype(np.uint8), np.asarray(affine, dtype=np.float64))
    nii.set_data_dtype(np.uint8)
    tmp_path = path.with_name(path.name.replace('.nii.gz', '') + '.tmp.nii.gz')
    with_retries(nib.save, nii, str(tmp_path), desc=f'save {path.name}')
    atomic_replace(tmp_path, path)
    return path


# -----------------------------------------------------------------------------
# Cropping and normalization
# -----------------------------------------------------------------------------
def brain_bbox(image: np.ndarray, margin: int = CROP_MARGIN) -> tuple[BBox, np.ndarray]:
    """Bounding box of all non-zero voxels (BraTS is skull-stripped, so non-zero = brain) and the brain mask."""
    brain = np.any(image != 0, axis=0)
    if not brain.any():
        raise ValueError('image is empty (all zeros)')
    bbox = []
    for axis in range(3):
        other_axes = tuple(a for a in range(3) if a != axis)
        indices = np.where(np.any(brain, axis=other_axes))[0]
        bbox += [max(0, int(indices[0]) - margin), min(brain.shape[axis], int(indices[-1]) + 1 + margin)]
    return np.array(bbox, dtype=np.int32), brain


def crop(array: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop the last three axes of array to bbox."""
    x0, x1, y0, y1, z0, z1 = (int(v) for v in bbox)
    return array[..., x0:x1, y0:y1, z0:z1]


def uncrop(cropped: np.ndarray, bbox: BBox, orig_shape, fill=0) -> np.ndarray:
    """Put a cropped array (..., x, y, z) back into the full original grid."""
    x0, x1, y0, y1, z0, z1 = (int(v) for v in bbox)
    full = np.full(cropped.shape[:-3] + tuple(int(s) for s in orig_shape), fill, dtype=cropped.dtype)
    full[..., x0:x1, y0:y1, z0:z1] = cropped
    return full


def lossless_storage_array(image: np.ndarray) -> np.ndarray:
    """Raw BraTS intensities are whole numbers - store them as int16 (lossless and compresses well)."""
    if np.all(np.mod(image, 1) == 0):
        low, high = float(image.min()), float(image.max())
        if np.iinfo(np.int16).min <= low and high <= np.iinfo(np.int16).max:
            return image.astype(np.int16)
        if np.iinfo(np.int32).min <= low and high <= np.iinfo(np.int32).max:
            return image.astype(np.int32)
    return image.astype(np.float32)


def normalization_params(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-modality mean and standard deviation over the non-zero (brain) voxels."""
    means, stds = [], []
    for channel in image:
        brain_values = channel[channel != 0].astype(np.float64)
        if brain_values.size < 10:
            means.append(0.0)
            stds.append(1.0)
            continue
        mean, std = float(brain_values.mean()), float(brain_values.std())
        means.append(mean)
        stds.append(std if std > 1e-6 else 1.0)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def normalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float = NORM_CLIP) -> np.ndarray:
    """z-score each modality, clip to [-clip, clip]; background (exactly 0) stays 0."""
    normalized = np.empty(image.shape, dtype=np.float32)
    for c in range(image.shape[0]):
        channel = image[c].astype(np.float32)
        z = (channel - float(mean[c])) / float(std[c])
        np.clip(z, -clip, clip, out=z)
        z[channel == 0] = 0.0
        normalized[c] = z
    return normalized


# -----------------------------------------------------------------------------
# Tumor regions
# -----------------------------------------------------------------------------
def region_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Internal label map -> boolean masks for the BraTS regions WT / TC / ET and the labels NCR / ED."""
    return {
        'WT': labels > 0,
        'TC': (labels == LABEL_NECROTIC) | (labels == LABEL_ENHANCING),
        'ET': labels == LABEL_ENHANCING,
        'NCR': labels == LABEL_NECROTIC,
        'ED': labels == LABEL_EDEMA,
    }


def region_channels(labels: np.ndarray) -> np.ndarray:
    """(3, ...) uint8 stack of the WT, TC, ET masks: the training target of the segmentation model."""
    masks = region_masks(labels)
    return np.stack([masks[region] for region in REGIONS], 0).astype(np.uint8)


# -----------------------------------------------------------------------------
# The prepared case
# -----------------------------------------------------------------------------
@dataclass
class PreparedCase:
    """One patient cropped to the brain, as stored on disk (one .npz per patient) and used everywhere.

    image       (4, x, y, z) raw intensities inside the brain crop (int16 when lossless, else float32)
    bbox        crop box in the original grid (x0, x1, y0, y1, z0, z1)
    affine      (4, 4) voxel -> world (mm) matrix of the ORIGINAL grid
    spacing     (3,) voxel size in mm
    orig_shape  (3,) shape of the original grid
    norm_mean   (4,) per-modality mean used for z-score normalization
    norm_std    (4,) per-modality standard deviation
    labels      (x, y, z) internal labels 0..3 inside the crop, or None for new scans without labels
    """
    image: np.ndarray
    bbox: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    orig_shape: np.ndarray
    norm_mean: np.ndarray
    norm_std: np.ndarray
    labels: np.ndarray | None = None

    # The .npz keys; 'seg' is the historical name of the label map and is kept so old files still load.
    _ARRAY_KEYS = ('image', 'bbox', 'affine', 'spacing', 'orig_shape', 'norm_mean', 'norm_std')

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    @property
    def crop_origin(self) -> tuple[int, int, int]:
        """(x0, y0, z0): where voxel (0, 0, 0) of the crop sits in the original grid."""
        return int(self.bbox[0]), int(self.bbox[2]), int(self.bbox[4])

    def normalized(self, region: BBox | None = None) -> np.ndarray:
        """Normalized float32 image (4, x, y, z); region = (x0, x1, y0, y1, z0, z1) inside the crop."""
        image = self.image if region is None else crop(self.image, region)
        return normalize_image(image, self.norm_mean, self.norm_std)

    def brain_mask(self) -> np.ndarray:
        return np.any(self.image != 0, axis=0)

    def save(self, path: Path) -> Path:
        arrays = {key: getattr(self, key) for key in self._ARRAY_KEYS}
        if self.labels is not None:
            arrays['seg'] = self.labels
        return save_npz(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> 'PreparedCase':
        with np.load(path, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in cls._ARRAY_KEYS}
            labels = stored['seg'] if 'seg' in stored.files else None
        return cls(**arrays, labels=labels)


def prepare_case(image: np.ndarray, labels: np.ndarray | None, affine: np.ndarray, spacing: np.ndarray,
                 margin: int = CROP_MARGIN) -> PreparedCase:
    """Crop to the brain and compute the normalization parameters (used for training AND the app)."""
    bbox, _ = brain_bbox(image, margin)
    image_crop = crop(image, bbox)
    mean, std = normalization_params(image_crop)
    return PreparedCase(
        image=lossless_storage_array(image_crop),
        bbox=bbox,
        affine=np.asarray(affine, dtype=np.float64),
        spacing=np.asarray(spacing, dtype=np.float32),
        orig_shape=np.array(image.shape[1:], dtype=np.int32),
        norm_mean=mean,
        norm_std=std,
        labels=crop(labels, bbox).astype(np.uint8) if labels is not None else None,
    )


def prepare_case_from_files(files: dict[str, Path], require_seg: bool = False) -> PreparedCase:
    """NIfTI files -> PreparedCase (what the offline app calls on uploaded scans)."""
    image, labels, affine, spacing = load_nifti_case(files, require_seg=require_seg)
    return prepare_case(image, labels, affine, spacing)
