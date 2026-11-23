"""
The refactored code must compute exactly what version 1 computed (Code_v1_before_refactor/brats_utils.py).

Skipped automatically when the old folder is not next to this one (e.g. on Google Drive).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from brats import case as case_tools
from brats import dataset, detection, localization, metrics, segmentation, training, xai

from conftest import synthetic_patient

OLD_CODE = Path(__file__).resolve().parents[2] / 'Code_v1_before_refactor' / 'brats_utils.py'
pytestmark = pytest.mark.skipif(not OLD_CODE.is_file(), reason='version 1 code not found')


@pytest.fixture(scope='module')
def old():
    spec = importlib.util.spec_from_file_location('brats_utils_v1', OLD_CODE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_old_dict(prepared):
    d = {key: getattr(prepared, key) for key in ('image', 'bbox', 'affine', 'spacing', 'orig_shape',
                                                   'norm_mean', 'norm_std')}
    d['seg'] = prepared.labels
    return d


def test_preprocessing_and_statistics(old):
    image, labels, affine, spacing = synthetic_patient(seed=3)
    new_case = case_tools.prepare_case(image, labels, affine, spacing)
    old_case = old.prepare_case(image, labels, affine, spacing)
    for key, value in as_old_dict(new_case).items():
        np.testing.assert_array_equal(value, old_case[key], err_msg=key)
        assert value.dtype == old_case[key].dtype, key
    new_row, new_slices, new_raw, new_norm = dataset.case_statistics('P', image, labels, affine, spacing, new_case)
    old_row, old_slices, old_raw, old_norm = old.case_statistics('P', image, labels, affine, spacing, old_case)
    assert list(new_row) == list(old_row)
    for key in old_row:
        assert new_row[key] == old_row[key] or (pd.isna(new_row[key]) and pd.isna(old_row[key])), key
    pd.testing.assert_frame_equal(new_slices, old_slices)
    np.testing.assert_array_equal(new_raw, old_raw)
    np.testing.assert_array_equal(new_norm, old_norm)
    new_preview = dataset.make_preview(image, labels, new_case)
    old_preview = old.make_preview(image, labels, old_case)
    for key in old_preview:
        np.testing.assert_array_equal(new_preview[key], old_preview[key], err_msg=key)


def test_patient_split(old):
    rng = np.random.default_rng(0)
    cases = pd.DataFrame({'case_id': [f'P{i:03d}' for i in range(60)], 'status': 'ok',
                          'vol_wt_cm3': rng.uniform(5, 200, 60), 'et_present': rng.random(60) > 0.2})
    pd.testing.assert_frame_equal(dataset.make_patient_split(cases), old.make_split(cases))


def test_metrics_and_localization(old):
    _, truth, affine, spacing = synthetic_patient(seed=1)
    pred = np.roll(truth, 2, axis=0)
    assert metrics.segmentation_metrics(pred, truth, spacing) == old.segmentation_metrics(pred, truth, spacing)
    np.testing.assert_array_equal(metrics.label_confusion(pred, truth), old.label_confusion(pred, truth))
    brain = np.ones_like(truth, bool)
    for region in ('WT', 'TC', 'ET'):
        new_loc = localization.localize_mask(case_tools.region_masks(pred)[region], affine, spacing, brain, (3, 4, 5))
        old_loc = old.localize_mask(old.region_masks(pred)[region], affine, spacing, brain, (3, 4, 5))
        assert new_loc == old_loc
        assert localization.describe_location(new_loc) == old.localization_text(old_loc)
    true_loc = localization.localize_mask(truth > 0, affine, spacing, brain)
    assert localization.localization_errors(new_loc, true_loc) == old.localization_errors(new_loc, true_loc)

    rng = np.random.default_rng(0)
    y, prob = rng.integers(0, 2, 500), rng.random(500)
    assert metrics.binary_metrics(y, prob, 0.4) == old.binary_metrics(y, prob, 0.4)
    table, ece = metrics.calibration_table(y, prob)
    old_table, old_ece = old.calibration_table(y, prob)
    pd.testing.assert_frame_equal(table, old_table)
    assert ece == old_ece
    predictions = pd.DataFrame({'case_id': np.repeat(['A', 'B'], 250), 'z_orig': np.tile(np.arange(250), 2),
                                'label': y, 'prob': prob})
    pd.testing.assert_frame_equal(detection.patient_detection_summary(predictions, 0.5),
                                  old.patient_detection_summary(predictions, 0.5))


def test_threshold_choice_is_unchanged_for_a_trained_model(old):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    prob = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 2000), 0, 1)     # a reasonably good model
    for method in ('f1', 'youden'):
        assert metrics.choose_threshold(y, prob, method) == old.choose_threshold(y, prob, method)


def test_model_inputs_patches_and_augmentation(old):
    image, labels, affine, spacing = synthetic_patient(seed=2)
    new_case = case_tools.prepare_case(image, labels, affine, spacing)
    old_case = old.prepare_case(image, labels, affine, spacing)
    z = [3, 8, 20]
    np.testing.assert_array_equal(detection.detection_inputs(new_case, z), old.detection_slices_from_case(old_case, z))
    np.testing.assert_array_equal(detection.detection_masks(new_case, z), old.detection_masks_from_case(old_case, z))
    slices = detection.detection_inputs(new_case, z)
    np.testing.assert_array_equal(detection.augment_slices(slices.copy(), np.random.default_rng(7)),
                                  old.augment_slices_2d(slices.copy(), np.random.default_rng(7)))
    for fg_prob in (0.0, 1.0):
        new_patch = segmentation.random_patch(new_case, (32, 32, 32), fg_prob, np.random.default_rng(5))
        old_patch = old.random_patch(old_case, (32, 32, 32), fg_prob, np.random.default_rng(5))
        for a, b in zip(new_patch, old_patch):
            np.testing.assert_array_equal(a, b)
    new_aug = segmentation.augment_patch(*new_patch, np.random.default_rng(9))
    old_aug = old.augment_patch_3d(*old_patch, np.random.default_rng(9))
    for a, b in zip(new_aug, old_aug):
        np.testing.assert_array_equal(a, b)
    probs = np.random.default_rng(4).random((3, 20, 20, 20))
    np.testing.assert_array_equal(segmentation.probs_to_labels(probs, 0.6, 50), old.probs_to_labels(probs, 0.6, 50))


def test_models_loss_and_schedule(old):
    torch.manual_seed(0)
    new_model = segmentation.build_segmentation_model('segresnet')
    old_model = old.build_segmentation_model('segresnet')
    old_model.load_state_dict(new_model.state_dict())              # same architecture, same parameter names
    logits, target = torch.randn(2, 3, 8, 8, 8), (torch.rand(2, 3, 8, 8, 8) > 0.5).float()
    assert torch.equal(segmentation.BCEDiceLoss()(logits, target), old.BCEDiceLoss()(logits, target))
    volume = torch.randn(4, 40, 36, 32)
    new_probs = segmentation.predict_volume(new_model, volume, torch.device('cpu'), (32, 32, 32), 2, 0.5, False)
    old_probs = old.segmentation_predict(old_model, volume, torch.device('cpu'), (32, 32, 32), 2, 0.5, False)
    np.testing.assert_allclose(new_probs, old_probs, atol=1e-6)

    detector, _ = detection.build_detection_model('efficientnet_b0', pretrained=False)
    inputs = torch.randn(2, 4, 64, 64)
    for plus_plus in (False, True):
        new_cam, new_prob = xai.gradcam(detector, inputs, plus_plus)
        old_cam, old_prob = old.gradcam(detector, inputs, plus_plus)
        np.testing.assert_allclose(new_cam, old_cam, atol=1e-6)
        np.testing.assert_allclose(new_prob, old_prob, atol=1e-6)

    def lr_factors(make_scheduler):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = make_scheduler(optimizer, 50, 5)
        return [scheduler.lr_lambdas[0](step) for step in range(50)]
    np.testing.assert_allclose(lr_factors(training.make_scheduler), lr_factors(old.make_scheduler))


def test_checkpoints_load_across_versions(old, tmp_path):
    model = segmentation.build_segmentation_model('segresnet')
    old.save_model_weights(tmp_path / 'v1.pt', model, 'segresnet', {'roi': [128, 128, 128]}, 3, 'val_mean_dice', 0.8)
    loaded = training.load_model_weights(tmp_path / 'v1.pt')
    assert loaded['model_name'] == 'segresnet' and loaded['epoch'] == 3
    segmentation.build_segmentation_model(loaded['model_name']).load_state_dict(loaded['model'])
