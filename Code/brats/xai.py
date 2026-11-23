"""
Explainable AI for the detection model (Grad-CAM, Grad-CAM++, occlusion) and uncertainty helpers.

Test-time-augmentation uncertainty for the segmentation model lives in segmentation.predict_volume_tta.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model: torch.nn.Module, inputs: torch.Tensor, plus_plus: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM (or Grad-CAM++) heatmaps of a timm classifier for the 'tumor' logit.

    inputs: (B, C, H, W) float tensor on the model's device.
    Returns heatmaps (B, H, W) scaled to [0, 1] and the predicted tumor probabilities (B,).
    """
    model.eval()
    with torch.enable_grad():
        features = model.forward_features(inputs)
        features.retain_grad()
        logits = model.forward_head(features)
        logits = logits.reshape(logits.shape[0], -1)[:, 0]
        model.zero_grad(set_to_none=True)
        logits.sum().backward()
    grads = features.grad.detach().float()
    activations = features.detach().float()
    if plus_plus:
        grads_2, grads_3 = grads ** 2, grads ** 3
        denominator = 2 * grads_2 + activations.sum(dim=(2, 3), keepdim=True) * grads_3
        alpha = grads_2 / torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        channel_weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
    else:
        channel_weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((channel_weights * activations).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=inputs.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    flat = cam.reshape(cam.shape[0], -1)
    low, high = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    cam = (cam - low) / torch.clamp(high - low, min=1e-8)
    model.zero_grad(set_to_none=True)
    return cam.cpu().numpy(), torch.sigmoid(logits.detach().float()).cpu().numpy()


@torch.no_grad()
def occlusion_map(model: torch.nn.Module, image: np.ndarray, device: torch.device, patch: int = 24, stride: int = 12,
                  batch: int = 64) -> tuple[np.ndarray, float]:
    """How much does the tumor probability drop when each square is hidden?

    image: (C, H, W). Returns the average probability drop per pixel (H, W) and the original probability.
    """
    model.eval()
    _, height, width = image.shape
    base = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None].to(device)
    base_prob = torch.sigmoid(model(base).float()).item()
    positions = [(i, j) for i in range(0, height - patch + 1, stride) for j in range(0, width - patch + 1, stride)]
    drop_sum = np.zeros((height, width), np.float64)
    count = np.zeros((height, width), np.float64)
    for start in range(0, len(positions), batch):
        chunk = positions[start:start + batch]
        occluded = base.repeat(len(chunk), 1, 1, 1)
        for n, (i, j) in enumerate(chunk):
            occluded[n, :, i:i + patch, j:j + patch] = 0
        probs = torch.sigmoid(model(occluded).float()).squeeze(1).cpu().numpy()
        for n, (i, j) in enumerate(chunk):
            drop_sum[i:i + patch, j:j + patch] += base_prob - probs[n]
            count[i:i + patch, j:j + patch] += 1
    return (drop_sum / np.maximum(count, 1)).astype(np.float32), base_prob


def heatmap_share_inside(heatmap: np.ndarray, mask: np.ndarray) -> float:
    """Share of the heatmap's total energy that falls inside mask (1.0 = all of it)."""
    total = float(heatmap.sum())
    return float(heatmap[mask].sum()) / total if total > 0 else float('nan')


def binary_entropy(prob: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Uncertainty of a probability in bits: 0 when certain, 1 at p = 0.5."""
    prob = np.clip(prob, eps, 1 - eps)
    return -(prob * np.log2(prob) + (1 - prob) * np.log2(1 - prob))
