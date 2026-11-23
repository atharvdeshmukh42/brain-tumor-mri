import gzip
import io
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from settings import MODALITIES, Settings


def validate_files(files: dict[str, Path], settings: Settings):
    shape, affine = None, None
    for name, path in files.items():
        if path.stat().st_size > settings.max_file_bytes:
            raise ValueError(f"{name}: file exceeds 128 MiB limit")
        if path.name.endswith(".gz"):
            total = 0
            with gzip.open(path, "rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.max_expanded_bytes:
                        raise ValueError(f"{name}: expanded gzip exceeds 256 MiB limit")
        image = nib.load(path)
        if (
            len(image.shape) != 3
            or min(image.shape) < 2
            or math.prod(image.shape) > settings.max_voxels
        ):
            raise ValueError(
                f"{name}: expected a 3D volume with at most {settings.max_voxels} voxels"
            )
        if (
            not np.isfinite(image.affine).all()
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8
        ):
            raise ValueError(f"{name}: invalid affine")
        if shape is None:
            shape, affine = image.shape, image.affine
        elif shape != image.shape or not np.allclose(affine, image.affine, atol=1e-4):
            raise ValueError(f"{name}: all volumes must have the same shape and affine")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError(f"{name}: contains non-finite values")
        if name == "seg":
            if not np.isin(data, [0, 1, 2, 4]).all():
                raise ValueError("seg: expected BraTS labels 0, 1, 2, 4")
        elif not np.any(data):
            raise ValueError(f"{name}: volume is empty")
    return {
        "shape": list(shape),
        "orientation": list(nib.aff2axcodes(affine)),
        "has_reference": "seg" in files,
    }


def input_files(folder: Path):
    return {
        name: next(folder.glob(f"{name}.nii*"))
        for name in (*MODALITIES, "seg")
        if list(folder.glob(f"{name}.nii*"))
    }


def plane_slice(volume, plane, index):
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}[plane]
    if not 0 <= index < volume.shape[axis]:
        raise ValueError("Slice index outside volume")
    return np.rot90(np.take(volume, index, axis=axis))


def render_slice(files, folder, modality, plane, index, overlay, opacity, cam=None):
    volume = nib.load(files[modality]).get_fdata(dtype=np.float32)
    positive = volume[volume != 0]
    low, high = np.percentile(positive, [1, 99]) if positive.size else (0, 1)
    grey = np.clip(
        (plane_slice(volume, plane, index) - low) / max(high - low, 1e-6), 0, 1
    )
    rgb = np.repeat(grey[..., None], 3, axis=-1)
    if overlay != "none":
        path = (
            folder / "tumor_mask.nii.gz" if overlay == "predicted" else files.get("seg")
        )
        if path is None or not path.exists():
            raise ValueError("Requested overlay is not available")
        labels = plane_slice(nib.load(path).get_fdata(), plane, index)
        for label, color in [
            (1, [0.95, 0.55, 0.25]),
            (2, [0.15, 0.8, 0.7]),
            (4, [0.8, 0.35, 0.85]),
        ]:
            selected = labels == label
            rgb[selected] = rgb[selected] * (1 - opacity) + np.array(color) * opacity
    if cam is not None:
        heat = np.rot90(cam)
        heat_rgb = np.stack([np.ones_like(heat), heat * 0.6, np.zeros_like(heat)], -1)
        alpha = heat[..., None] * opacity
        rgb = rgb * (1 - alpha) + heat_rgb * alpha
    output = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
        output, format="PNG"
    )
    return output.getvalue()
