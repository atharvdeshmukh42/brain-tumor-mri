"""Model inputs, datasets, prediction shapes, the training loop and resuming."""
import numpy as np
import pandas as pd
import pytest
import torch

from brats import case as case_tools
from brats import detection, segmentation, training, xai
from brats.config import DETECTION_CANVAS

from conftest import synthetic_patient


@pytest.fixture
def prepared(patient):
    return case_tools.prepare_case(*patient)


@pytest.fixture
def cases_dir(tmp_path):
    for i in range(3):
        case_tools.prepare_case(*synthetic_patient(seed=i)).save(tmp_path / f'case{i}.npz')
    return tmp_path


def test_detection_inputs_put_the_crop_back_in_place(prepared, patient):
    z = [5, 10]
    inputs = detection.detection_inputs(prepared, z)
    assert inputs.shape == (2, 4, DETECTION_CANVAS, DETECTION_CANVAS)
    # the synthetic volume is 64 x 72: it sits centred on the 240 x 240 canvas
    brain_on_canvas = np.any(inputs[0] != 0, axis=0)
    full_brain = np.any(patient[0][:, :, :, z[0] + int(prepared.bbox[4])] != 0, axis=0)
    assert brain_on_canvas.sum() == full_brain.sum()
    masks = detection.detection_masks(prepared, z)
    assert masks.shape == (2, DETECTION_CANVAS, DETECTION_CANVAS) and masks.dtype == bool


def test_detection_dataset_and_collate(cases_dir):
    rows = [{'case_id': f'case{i}', 'z': z, 'z_orig': z, 'label': int(z > 20), 'wt_px': 0, 'brain_px': 900}
            for i in range(3) for z in range(30)]
    table = pd.DataFrame(rows)
    train_set = detection.DetectionDataset(table, cases_dir, slices_per_case=4, repeats=2, augment=True)
    assert len(train_set) == 6
    batch = detection.collate_patients([train_set[0], train_set[1]])
    assert batch[0].shape == (8, 4, DETECTION_CANVAS, DETECTION_CANVAS) and batch[1].shape == (8,)
    eval_set = detection.DetectionDataset(table, cases_dir)
    assert eval_set[0][0].shape[0] == 30


def test_random_patch_and_prediction_shapes(prepared):
    rng = np.random.default_rng(0)
    image, target = segmentation.random_patch(prepared, (32, 32, 32), foreground_prob=1.0, rng=rng)
    assert image.shape == (4, 32, 32, 32) and target.shape == (3, 32, 32, 32)
    assert target[0].any()                                   # foreground patch contains tumor
    image, target = segmentation.augment_patch(image, target, rng)
    assert image.flags['C_CONTIGUOUS']
    model = segmentation.build_segmentation_model('segresnet')
    probs = segmentation.predict_volume(model, prepared.normalized(), torch.device('cpu'), roi=(32, 32, 32),
                                        sw_batch=2, amp=False)
    assert probs.shape == (3,) + prepared.image.shape[1:] and 0 <= probs.min() and probs.max() <= 1
    mean, std = segmentation.predict_volume_tta(model, prepared.normalized(), torch.device('cpu'), roi=(32, 32, 32),
                                                sw_batch=2, amp=False)
    assert mean.shape == probs.shape and std.min() >= 0


def test_all_segmentation_models_build():
    for name in segmentation.SEGMENTATION_MODELS:
        assert training.count_parameters(segmentation.build_segmentation_model(name)) > 1e6
    with pytest.raises(ValueError):
        segmentation.build_segmentation_model('nope')


def test_probs_to_labels_nests_the_regions():
    probs = np.zeros((3, 4, 1, 1))
    probs[0, 1:] = 0.9          # WT
    probs[1, 2:] = 0.9          # TC
    probs[2, 3:] = 0.9          # ET
    assert segmentation.probs_to_labels(probs)[:, 0, 0].tolist() == [0, 2, 1, 3]
    assert segmentation.probs_to_labels(probs, et_min_voxels=5)[:, 0, 0].tolist() == [0, 2, 1, 1]


def test_scheduler_warmup_then_cosine():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = training.make_scheduler(optimizer, total_steps=100, warmup_steps=10, min_lr_ratio=0.01)
    lrs = []
    for _ in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.01, abs=0.01) and all(a >= b for a, b in zip(lrs[10:], lrs[11:]))


def tiny_problem():
    torch.manual_seed(0)
    x = torch.randn(64, 3)
    y = (x[:, 0] > 0).float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=16)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = training.make_scheduler(optimizer, 40)
    scaler = training.make_grad_scaler(torch.device('cpu'))
    return model, loader, optimizer, scheduler, scaler


def test_train_one_epoch_learns():
    model, loader, optimizer, scheduler, scaler = tiny_problem()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = training.train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, torch.device('cpu'),
                                     track_accuracy=True)
    for _ in range(8):
        last = training.train_one_epoch(model, loader, optimizer, scheduler, scaler, loss_fn, torch.device('cpu'),
                                        track_accuracy=True)
    assert last['train_loss'] < first['train_loss'] and last['train_acc'] > 0.8
    assert last['train_samples'] == 64


def test_resumable_training_resumes_and_stops_early(tmp_path):
    def new_run(patience):
        model, _, optimizer, scheduler, scaler = tiny_problem()
        return training.ResumableTraining(tmp_path, model, optimizer, scheduler, scaler, torch.device('cpu'),
                                          model_name='linear', config={'a': 1}, metric_name='val_score', epochs=10,
                                          patience=patience, keep_every_epoch=False)

    run = new_run(patience=2)
    scores = iter([0.5, 0.7, 0.6])
    for epoch in run.remaining_epochs():
        run.end_epoch(epoch, {'epoch_seconds': 1.0}, metric=next(scores))
        if epoch == 3:
            break                                   # simulate a Colab disconnect after epoch 3
    resumed = new_run(patience=2)
    assert resumed.start_epoch == 4 and resumed.best_epoch == 2 and resumed.validations_since_best == 1
    for epoch in resumed.remaining_epochs():
        resumed.end_epoch(epoch, {'epoch_seconds': 1.0}, metric=0.1)
    assert resumed.early_stopped and resumed.history[-1]['epoch'] == 4
    best = training.load_model_weights(resumed.best_path)
    assert best['epoch'] == 2 and best['metric'] == pytest.approx(0.7)
    assert len(pd.read_csv(tmp_path / 'history.csv')) == 4


def test_gradcam_and_occlusion_shapes():
    model, _ = detection.build_detection_model('efficientnet_b0', pretrained=False)
    inputs = torch.randn(2, 4, 64, 64)
    cam, prob = xai.gradcam(model, inputs)
    assert cam.shape == (2, 64, 64) and 0 <= cam.min() and cam.max() <= 1 and prob.shape == (2,)
    cam_pp, _ = xai.gradcam(model, inputs, plus_plus=True)
    assert cam_pp.shape == (2, 64, 64)
    drop, base = xai.occlusion_map(model, inputs[0].numpy(), torch.device('cpu'), patch=16, stride=16)
    assert drop.shape == (64, 64) and 0 <= base <= 1
