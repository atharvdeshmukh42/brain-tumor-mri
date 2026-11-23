"""Metrics and localization on masks whose answers are known."""
import numpy as np
import pytest
import torch

from brats import localization, metrics
from brats.config import HD95_EMPTY_PENALTY_MM


def cube(shape=(40, 40, 40), start=(10, 10, 10), size=10):
    mask = np.zeros(shape, bool)
    mask[tuple(slice(s, s + size) for s in start)] = True
    return mask


def test_dice_and_iou():
    a, b = cube(start=(10, 10, 10)), cube(start=(15, 10, 10))       # half overlap along x
    assert metrics.dice_score(a, b) == pytest.approx(0.5)
    assert metrics.iou_score(a, b) == pytest.approx(1 / 3)
    empty = np.zeros_like(a)
    assert metrics.dice_score(empty, empty) == 1.0
    assert metrics.dice_score(a, empty) == 0.0


def test_hd95_of_a_shifted_cube_is_the_shift():
    assert metrics.hd95(cube(start=(10, 10, 10)), cube(start=(13, 10, 10))) == pytest.approx(3.0)
    assert metrics.hd95(cube(), cube()) == 0.0
    assert metrics.hd95(cube(), np.zeros((40, 40, 40), bool)) == HD95_EMPTY_PENALTY_MM


def test_binary_metrics_counts():
    y = np.array([1, 1, 0, 0, 1])
    prob = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
    result = metrics.binary_metrics(y, prob, 0.5)
    assert (result['tp'], result['fn'], result['fp'], result['tn']) == (2, 1, 1, 1)
    assert result['sensitivity'] == pytest.approx(2 / 3)
    assert result['specificity'] == pytest.approx(0.5)


def test_threshold_is_kept_in_a_sensible_range():
    y = np.array([0, 0, 1, 1, 1, 1])
    prob = np.array([0.001, 0.0015, 0.002, 0.003, 0.004, 0.005])     # an almost untrained model
    assert metrics.choose_threshold(y, prob, 'f1') == pytest.approx(0.05)
    assert metrics.choose_threshold(np.ones(4), np.ones(4)) == 0.5


def test_weighted_bce_matches_the_training_loss():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    logits = np.array([2.0, -1.0, -0.5, 0.3])
    torch_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5]))(torch.tensor(logits), torch.tensor(y))
    ours = metrics.weighted_bce(y, 1 / (1 + np.exp(-logits)), pos_weight=2.5)
    assert ours == pytest.approx(float(torch_loss), rel=1e-5)


def test_label_confusion():
    pred = np.array([0, 1, 2, 3, 3])
    truth = np.array([0, 1, 2, 2, 3])
    confusion = metrics.label_confusion(pred, truth)
    assert confusion.sum() == 5 and confusion[2, 3] == 1 and np.trace(confusion) == 4


def test_localization_of_a_cube():
    mask = cube(start=(5, 10, 20), size=6)
    brain = np.ones_like(mask)
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])            # x axis flipped, as in many real scans
    loc = localization.localize_mask(mask, affine, (1, 1, 1), brain_mask=brain)
    assert loc['volume_cm3'] == pytest.approx(0.216)
    assert (loc['centroid_i'], loc['centroid_j'], loc['centroid_k']) == (7.5, 12.5, 22.5)
    assert loc['centroid_x_mm'] == pytest.approx(-7.5)
    assert (loc['extent_x_mm'], loc['extent_y_mm'], loc['extent_z_mm']) == (6, 6, 6)
    assert loc['n_components'] == 1
    assert loc['side'] == 'Right'           # voxel index below the centre + flipped x axis = patient right
    assert 'Right hemisphere' in localization.describe_location(loc)


def test_localization_respects_the_crop_offset():
    mask = cube(start=(2, 2, 2), size=4)
    full = localization.localize_mask(np.pad(mask, ((10, 0), (20, 0), (30, 0))), np.eye(4), (1, 1, 1))
    cropped = localization.localize_mask(mask, np.eye(4), (1, 1, 1), offset=(10, 20, 30))
    assert cropped['centroid_x_mm'] == full['centroid_x_mm']
    assert cropped['bbox_k_min'] == full['bbox_k_min']


def test_localization_errors():
    a = localization.localize_mask(cube(start=(10, 10, 10)), np.eye(4), (1, 1, 1))
    b = localization.localize_mask(cube(start=(13, 10, 10)), np.eye(4), (1, 1, 1))
    errors = localization.localization_errors(a, b)
    assert errors['centroid_error_mm'] == pytest.approx(3.0)
    assert errors['bbox_iou'] == pytest.approx(70 / 130)
