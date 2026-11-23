"""One patient: labels, cropping, normalization, storage."""
import numpy as np
import pytest

from brats import case as case_tools
from brats.case import PreparedCase


def test_label_conversion_round_trip():
    brats = np.array([0, 1, 2, 4, 4, 0], np.float32)
    internal = case_tools.brats_to_internal_labels(brats)
    assert internal.tolist() == [0, 1, 2, 3, 3, 0]
    assert case_tools.internal_to_brats_labels(internal).tolist() == [0, 1, 2, 4, 4, 0]


def test_unexpected_label_values_are_rejected():
    with pytest.raises(ValueError):
        case_tools.brats_to_internal_labels(np.array([0, 5]))


def test_crop_and_uncrop_round_trip(patient):
    image, labels, _, _ = patient
    bbox, brain = case_tools.brain_bbox(image)
    cropped = case_tools.crop(labels, bbox)
    restored = case_tools.uncrop(cropped, bbox, labels.shape)
    np.testing.assert_array_equal(restored, labels)
    assert brain.sum() == np.any(case_tools.crop(image, bbox) != 0, axis=0).sum()   # no brain voxel lost


def test_storage_is_lossless_int16(patient):
    image = patient[0]
    stored = case_tools.lossless_storage_array(image)
    assert stored.dtype == np.int16
    np.testing.assert_array_equal(stored.astype(np.float32), image)
    assert case_tools.lossless_storage_array(image + 0.5).dtype == np.float32


def test_normalization_keeps_background_zero_and_centres_the_brain(patient):
    image = patient[0]
    mean, std = case_tools.normalization_params(image)
    normalized = case_tools.normalize_image(image, mean, std)
    brain = image[0] != 0
    assert np.all(normalized[:, ~brain] == 0)
    assert abs(normalized[0][brain].mean()) < 0.05
    assert normalized.max() <= 5.0 and normalized.min() >= -5.0


def test_region_masks_are_nested(patient):
    masks = case_tools.region_masks(patient[1])
    assert np.all(masks['ET'] <= masks['TC']) and np.all(masks['TC'] <= masks['WT'])
    assert case_tools.region_channels(patient[1]).shape == (3,) + patient[1].shape


def test_prepared_case_save_and_load(tmp_path, patient):
    prepared = case_tools.prepare_case(*patient)
    path = prepared.save(tmp_path / 'case.npz')
    with np.load(path) as stored:                     # same keys as the files written before the refactor
        assert set(stored.files) == {'image', 'bbox', 'affine', 'spacing', 'orig_shape', 'norm_mean', 'norm_std', 'seg'}
    loaded = PreparedCase.load(path)
    for field in ('image', 'bbox', 'affine', 'spacing', 'orig_shape', 'norm_mean', 'norm_std', 'labels'):
        np.testing.assert_array_equal(getattr(loaded, field), getattr(prepared, field))
    assert loaded.crop_origin == (int(prepared.bbox[0]), int(prepared.bbox[2]), int(prepared.bbox[4]))


def test_prepare_case_from_nifti_files(nifti_files, patient):
    prepared = case_tools.prepare_case_from_files(nifti_files, require_seg=True)
    image, labels, _, _ = patient
    np.testing.assert_array_equal(case_tools.uncrop(prepared.labels, prepared.bbox, prepared.orig_shape), labels)
    np.testing.assert_array_equal(case_tools.uncrop(prepared.image, prepared.bbox, prepared.orig_shape),
                                  image.astype(np.int16))


def test_label_nifti_is_written_with_brats_values(tmp_path, patient):
    import nibabel as nib
    labels, affine = patient[1], patient[2]
    path = case_tools.save_label_nifti(labels, affine, tmp_path / 'mask.nii.gz')
    saved = np.asarray(nib.load(str(path)).dataobj)
    assert set(np.unique(saved).tolist()) == {0, 1, 2, 4}
