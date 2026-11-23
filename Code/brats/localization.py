"""
Tumor localization: where is the tumor, how big is it, which side of the brain.

World coordinates follow the NIfTI convention (RAS+): +x = patient Right, +y = Anterior, +z = Superior.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy import ndimage


def localize_mask(mask: np.ndarray, affine: np.ndarray, spacing, brain_mask: np.ndarray | None = None,
                  offset=(0, 0, 0)) -> OrderedDict:
    """Position, size and volume of a boolean tumor mask.

    mask and brain_mask may be cropped; offset is then the crop origin (x0, y0, z0) in the original grid,
    so voxel indices and world coordinates always refer to the original scan.
    """
    spacing = np.asarray(spacing, dtype=float)
    affine = np.asarray(affine, dtype=float)
    offset = np.asarray(offset, dtype=float)
    out = OrderedDict(present=bool(mask.any()))
    if not out['present']:
        out.update(volume_cm3=0.0, n_components=0)
        return out
    voxels = np.argwhere(mask).astype(np.float64) + offset
    centroid = voxels.mean(0)
    centroid_world = affine @ np.append(centroid, 1.0)
    low, high = voxels.min(0), voxels.max(0)
    components, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    component_sizes = np.bincount(components.ravel())[1:]

    out['volume_cm3'] = float(mask.sum()) * float(np.prod(spacing)) / 1000.0
    out['n_components'] = int(n_components)
    out['largest_component_pct'] = float(100.0 * component_sizes.max() / component_sizes.sum())
    out['centroid_i'], out['centroid_j'], out['centroid_k'] = (float(v) for v in centroid)
    out['centroid_x_mm'], out['centroid_y_mm'], out['centroid_z_mm'] = (float(v) for v in centroid_world[:3])
    for name, axis in zip('ijk', range(3)):
        out[f'bbox_{name}_min'], out[f'bbox_{name}_max'] = int(low[axis]), int(high[axis])
    # Extent along the world axes (voxel axes may be flipped, so use the absolute affine columns).
    extent_voxels = high - low + 1
    extent_world = np.abs(affine[:3, :3]) @ extent_voxels
    out['extent_x_mm'], out['extent_y_mm'], out['extent_z_mm'] = (float(v) for v in extent_world)
    if brain_mask is not None and brain_mask.any():
        brain_centre = np.argwhere(brain_mask).astype(np.float64).mean(0) + offset
        brain_centre_world = affine @ np.append(brain_centre, 1.0)
        relative = centroid_world[:3] - brain_centre_world[:3]
        out['offset_x_mm'], out['offset_y_mm'], out['offset_z_mm'] = (float(v) for v in relative)
        out['side'] = 'Right' if relative[0] > 0 else 'Left'
        out['ap'] = 'Anterior' if relative[1] > 0 else 'Posterior'
        out['si'] = 'Superior' if relative[2] > 0 else 'Inferior'
    return out


def describe_location(loc: dict) -> str:
    """One readable sentence for the report, e.g. 'Left hemisphere, anterior and inferior ...; size ...'."""
    if not loc.get('present'):
        return 'No tumor region found.'
    parts = []
    if 'side' in loc:
        parts.append(f"{loc['side']} hemisphere, {loc['ap'].lower()} and {loc['si'].lower()} of the brain centre")
    parts.append(f"size {loc['extent_x_mm']:.0f} x {loc['extent_y_mm']:.0f} x {loc['extent_z_mm']:.0f} mm "
                 f"(left-right x front-back x up-down)")
    parts.append(f"volume {loc['volume_cm3']:.1f} cm3")
    if loc.get('n_components', 1) > 1:
        parts.append(f"{loc['n_components']} separate parts (largest {loc['largest_component_pct']:.0f}%)")
    return '; '.join(parts)


def localization_errors(loc_pred: dict, loc_true: dict) -> OrderedDict:
    """Centre distance (mm), bounding-box IoU, hemisphere agreement and volume error of a prediction."""
    out = OrderedDict()
    if loc_pred.get('present') and loc_true.get('present'):
        pred_centre = np.array([loc_pred[f'centroid_{axis}_mm'] for axis in 'xyz'])
        true_centre = np.array([loc_true[f'centroid_{axis}_mm'] for axis in 'xyz'])
        out['centroid_error_mm'] = float(np.linalg.norm(pred_centre - true_centre))
        intersection, pred_volume, true_volume = 1.0, 1.0, 1.0
        for axis in 'ijk':
            low = max(loc_pred[f'bbox_{axis}_min'], loc_true[f'bbox_{axis}_min'])
            high = min(loc_pred[f'bbox_{axis}_max'], loc_true[f'bbox_{axis}_max'])
            intersection *= max(0, high - low + 1)
            pred_volume *= loc_pred[f'bbox_{axis}_max'] - loc_pred[f'bbox_{axis}_min'] + 1
            true_volume *= loc_true[f'bbox_{axis}_max'] - loc_true[f'bbox_{axis}_min'] + 1
        out['bbox_iou'] = intersection / (pred_volume + true_volume - intersection)
        out['side_agree'] = loc_pred.get('side') == loc_true.get('side')
    else:
        both_absent = not loc_pred.get('present') and not loc_true.get('present')
        out['centroid_error_mm'] = float('nan')
        out['bbox_iou'] = 1.0 if both_absent else 0.0
        out['side_agree'] = float('nan')
    out['volume_error_cm3'] = loc_pred.get('volume_cm3', 0.0) - loc_true.get('volume_cm3', 0.0)
    return out
