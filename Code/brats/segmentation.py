"""
3D tumor segmentation. The model sees the 4 modalities and predicts, for every voxel, the probability of the
three BraTS regions WT (whole tumor), TC (tumor core) and ET (enhancing tumor), as 3 independent outputs.

Training uses random 128^3 patches (a third of them centred on tumor); prediction uses a sliding window over the
whole brain crop with Gaussian blending.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet, SegResNet, UNet
from tqdm.auto import tqdm

from .case import PreparedCase, normalize_image, region_channels, region_masks
from .config import LABEL_EDEMA, LABEL_ENHANCING, LABEL_NECROTIC, REGIONS
from .fileio import load_json
from .metrics import dice_score
from .training import WorkerRandom, autocast, load_model_weights

SEGMENTATION_MODELS = ('segresnet', 'unet3d', 'unet3d_large', 'dynunet')


def build_segmentation_model(name: str = 'segresnet', in_channels: int = 4, out_channels: int = 3) -> torch.nn.Module:
    """segresnet: residual encoder-decoder, winner of BraTS 2018 (the project's main model).
    unet3d / unet3d_large: classic 3D U-Net with residual units (comparison). dynunet: nnU-Net-style (large)."""
    if name == 'segresnet':
        return SegResNet(spatial_dims=3, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), init_filters=16,
                         in_channels=in_channels, out_channels=out_channels, dropout_prob=0.2)
    if name == 'unet3d':
        return UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2)
    if name == 'unet3d_large':
        return UNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                    channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2), num_res_units=2)
    if name == 'dynunet':
        return DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                       kernel_size=[3, 3, 3, 3, 3, 3], strides=[1, 2, 2, 2, 2, 2],
                       upsample_kernel_size=[2, 2, 2, 2, 2], norm_name='instance', res_block=True)
    raise ValueError(f'Unknown segmentation model "{name}". Choose one of {SEGMENTATION_MODELS}.')


# -----------------------------------------------------------------------------
# Training patches
# -----------------------------------------------------------------------------
def pad_to_min_shape(arrays: list[np.ndarray], roi) -> tuple[list[np.ndarray], tuple[int, int, int]]:
    """Zero-pad (..., X, Y, Z) arrays symmetrically so every spatial size is >= roi. Returns arrays and padding before."""
    shape = arrays[0].shape[-3:]
    pads = []
    for size, wanted in zip(shape, roi):
        missing = max(0, wanted - size)
        pads.append((missing // 2, missing - missing // 2))
    if not any(before or after for before, after in pads):
        return arrays, (0, 0, 0)
    padded = [np.pad(a, [(0, 0)] * (a.ndim - 3) + pads) for a in arrays]
    return padded, tuple(before for before, _ in pads)


def random_patch(case: PreparedCase, roi, foreground_prob: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Random patch (normalized image (4, *roi), region target (3, *roi)).

    With probability foreground_prob the patch is centred (with jitter) on a random voxel of a random tumor label,
    so the rare tumor classes are seen often enough.
    """
    (image, labels), _ = pad_to_min_shape([case.image, case.labels], roi)
    shape = labels.shape
    center = None
    if rng.random() < foreground_prob:
        present = [label for label in (LABEL_NECROTIC, LABEL_EDEMA, LABEL_ENHANCING) if (labels == label).any()]
        if present:
            label = present[int(rng.integers(len(present)))]
            voxels = np.flatnonzero(labels == label)
            center = np.array(np.unravel_index(voxels[int(rng.integers(len(voxels)))], shape))
    starts = []
    for axis in range(3):
        highest_start = shape[axis] - roi[axis]
        if center is None:
            starts.append(int(rng.integers(0, highest_start + 1)))
        else:
            jitter = int(rng.integers(-roi[axis] // 4, roi[axis] // 4 + 1))
            starts.append(int(np.clip(center[axis] - roi[axis] // 2 + jitter, 0, highest_start)))
    window = tuple(slice(start, start + size) for start, size in zip(starts, roi))
    normalized = normalize_image(image[(slice(None),) + window], case.norm_mean, case.norm_std)
    return normalized, region_channels(labels[window])


def augment_patch(image: np.ndarray, target: np.ndarray, rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Random mirroring along each axis and per-modality intensity scale / shift."""
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis)
            target = np.flip(target, axis=axis)
    if rng.random() < 0.8:
        brain = np.any(image != 0, axis=0)
        scale = rng.uniform(0.9, 1.1, size=(image.shape[0], 1, 1, 1)).astype(np.float32)
        shift = rng.uniform(-0.1, 0.1, size=(image.shape[0], 1, 1, 1)).astype(np.float32)
        image = np.where(brain[None], image * scale + shift, 0.0).astype(np.float32)
    return np.ascontiguousarray(image), np.ascontiguousarray(target)


class SegPatchDataset(torch.utils.data.Dataset):
    """Training: one random patch per patient per item; items are (image float16, target uint8)."""

    def __init__(self, case_ids, cases_dir: Path, roi=(128, 128, 128), foreground_prob: float = 0.33,
                 augment: bool = True, repeats: int = 1):
        self.case_ids = list(case_ids)
        self.cases_dir = Path(cases_dir)
        self.roi = tuple(int(r) for r in roi)
        self.foreground_prob = foreground_prob
        self.augment = augment
        self.repeats = int(repeats)
        self._random = WorkerRandom()

    def __len__(self) -> int:
        return len(self.case_ids) * self.repeats

    def __getitem__(self, index: int):
        rng = self._random.get()
        case = PreparedCase.load(self.cases_dir / f'{self.case_ids[index % len(self.case_ids)]}.npz')
        image, target = random_patch(case, self.roi, self.foreground_prob, rng)
        if self.augment:
            image, target = augment_patch(image, target, rng)
        return torch.from_numpy(image.astype(np.float16)), torch.from_numpy(target)


@dataclass
class EvaluationSample:
    """One whole patient for sliding-window evaluation (everything except the raw image)."""
    case_id: str
    image: torch.Tensor          # normalized (4, x, y, z) float16
    labels: np.ndarray | None    # internal labels inside the crop
    brain: np.ndarray            # brain mask inside the crop
    bbox: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    orig_shape: np.ndarray

    @property
    def crop_origin(self) -> tuple[int, int, int]:
        return int(self.bbox[0]), int(self.bbox[2]), int(self.bbox[4])


class SegCaseDataset(torch.utils.data.Dataset):
    """Evaluation: one whole (cropped) patient per item, as an EvaluationSample."""

    def __init__(self, case_ids, cases_dir: Path):
        self.case_ids = list(case_ids)
        self.cases_dir = Path(cases_dir)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> EvaluationSample:
        case_id = self.case_ids[index]
        case = PreparedCase.load(self.cases_dir / f'{case_id}.npz')
        return EvaluationSample(case_id=case_id, image=torch.from_numpy(case.normalized().astype(np.float16)),
                                labels=case.labels, brain=case.brain_mask(), bbox=case.bbox, affine=case.affine,
                                spacing=case.spacing, orig_shape=case.orig_shape)


def single_item_collate(batch):
    """Evaluation loaders use batch_size 1 and hand over the item itself."""
    return batch[0]


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class BCEDiceLoss:
    """Binary cross-entropy + soft Dice over the 3 region channels (WT, TC, ET)."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0, eps: float = 1e-5):
        self.dice_weight, self.bce_weight, self.eps = dice_weight, bce_weight, eps

    def __call__(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        sum_dims = (0,) + tuple(range(2, logits.ndim))          # over batch and space, separately per region
        intersection = (prob * target).sum(sum_dims)
        denominator = prob.sum(sum_dims) + target.sum(sum_dims)
        dice_loss = 1.0 - ((2 * intersection + self.eps) / (denominator + self.eps)).mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_volume(model: torch.nn.Module, image, device: torch.device, roi=(128, 128, 128), sw_batch: int = 4,
                   overlap: float = 0.5, amp: bool = True) -> np.ndarray:
    """Sliding-window prediction for one patient: image (4, x, y, z) -> region probabilities (3, x, y, z)."""
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image))
    batch = image[None].to(device).float()
    model.eval()
    with autocast(device, amp):
        logits = sliding_window_inference(batch, roi_size=tuple(roi), sw_batch_size=sw_batch,
                                          predictor=lambda patch: model(patch).float(), overlap=overlap,
                                          mode='gaussian')
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def predict_volume_tta(model: torch.nn.Module, image, device: torch.device, roi=(128, 128, 128), sw_batch: int = 4,
                       overlap: float = 0.5, amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Test-time augmentation: predict all 8 mirror combinations. Returns their mean and std (= uncertainty)."""
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.asarray(image))
    total, total_sq, n = None, None, 0
    flip_sets = itertools.chain.from_iterable(itertools.combinations((1, 2, 3), r) for r in range(4))
    for flips in flip_sets:
        flipped = torch.flip(image, dims=flips) if flips else image
        probs = predict_volume(model, flipped, device, roi, sw_batch, overlap, amp)
        if flips:
            probs = np.flip(probs, axis=flips).copy()
        total = probs if total is None else total + probs
        total_sq = probs ** 2 if total_sq is None else total_sq + probs ** 2
        n += 1
    mean = total / n
    std = np.sqrt(np.clip(total_sq / n - mean ** 2, 0, None))
    return mean.astype(np.float32), std.astype(np.float32)


def probs_to_labels(probs: np.ndarray, threshold: float = 0.5, et_min_voxels: int = 0) -> np.ndarray:
    """Region probabilities (WT, TC, ET) -> internal label map 0..3 (nested regions: ET inside TC inside WT).

    et_min_voxels > 0 relabels a tiny predicted enhancing tumor as necrotic core (a common BraTS post-processing).
    """
    whole, core, enhancing = probs[0] > threshold, probs[1] > threshold, probs[2] > threshold
    labels = np.zeros(whole.shape, dtype=np.uint8)
    labels[whole] = LABEL_EDEMA
    labels[core] = LABEL_NECROTIC
    labels[enhancing] = LABEL_ENHANCING
    if et_min_voxels and 0 < (labels == LABEL_ENHANCING).sum() < et_min_voxels:
        labels[labels == LABEL_ENHANCING] = LABEL_NECROTIC
    return labels


def validation_dice(model: torch.nn.Module, loader, device: torch.device, roi=(128, 128, 128), sw_batch: int = 4,
                    overlap: float = 0.5, amp: bool = True, threshold: float = 0.5) -> dict[str, float]:
    """Mean Dice per region over a SegCaseDataset loader (used during training)."""
    scores = {region: [] for region in REGIONS}
    for sample in tqdm(loader, desc='val', leave=False):
        probs = predict_volume(model, sample.image, device, roi, sw_batch, overlap, amp)
        pred_masks = region_masks(probs_to_labels(probs, threshold))
        true_masks = region_masks(sample.labels)
        for region in REGIONS:
            scores[region].append(dice_score(pred_masks[region], true_masks[region]))
    return {region: float(np.mean(values)) for region, values in scores.items()}


def load_segmenter(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Trained segmentation model (eval mode) and its model card from a run folder."""
    weights = load_model_weights(Path(run_dir) / 'checkpoints' / 'best.pt', device)
    model = build_segmentation_model(weights['model_name'])
    model.load_state_dict(weights['model'])
    return model.to(device).eval(), load_json(Path(run_dir) / 'model_card.json')
