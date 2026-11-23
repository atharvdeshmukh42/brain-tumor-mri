import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import create_app
from settings import Settings
from store import Store


class IdleEngine:
    def __init__(self, settings, store):
        self.status = {"status": "ready", "device": "cpu", "error": None}

    def start(self):
        pass

    def close(self):
        pass

    def submit(self, id):
        pass


def volume(path, *, affine=None, data=None):
    image = nib.Nifti1Image(
        np.ones((8, 9, 10), dtype=np.float32) if data is None else data,
        np.eye(4) if affine is None else affine,
    )
    image.header.set_xyzt_units("mm")
    nib.save(image, path)


@pytest.fixture
def setup(tmp_path):
    settings = Settings(
        models=tmp_path / "models",
        samples=tmp_path / "samples",
        storage=tmp_path / "storage",
    )
    folder = settings.samples / "test-case"
    folder.mkdir(parents=True)
    for name in ("flair", "t1", "t1ce", "t2", "seg"):
        volume(folder / f"test-case_{name}.nii.gz")
    app = create_app(settings, IdleEngine)
    with TestClient(app) as client:
        yield client, app, settings


def test_sample_history_and_restart(setup):
    client, _, settings = setup
    assert client.get("/api/samples").json() == [
        {"id": "test-case", "has_reference": True}
    ]
    response = client.post("/api/analyses/sample", json={"sample_id": "test-case"})
    assert response.status_code == 202
    item = response.json()
    assert item["metadata"]["shape"] == [8, 9, 10]
    assert item["status"] == "queued"
    assert client.get("/api/analyses").json()[0]["id"] == item["id"]
    assert client.get(f"/api/analyses/{item['id']}").json() == item
    reopened = Store(settings.storage)
    assert reopened.get(item["id"])["status"] == "queued"
    reopened.recover_interrupted()
    restarted = reopened.get(item["id"])
    assert restarted["status"] == "failed"
    assert "interrupted" in restarted["error"]


def test_ids_and_invalid_requests(setup):
    client, _, _ = setup
    assert (
        client.post("/api/analyses/sample", json={"sample_id": "../../etc"}).status_code
        == 404
    )
    assert client.get("/api/analyses/not-a-uuid").status_code == 422
    assert client.post("/api/analyses/upload").status_code == 422
    assert (
        client.get("/api/analyses/00000000-0000-0000-0000-000000000000").status_code
        == 404
    )


def test_render_and_download_contract(setup):
    client, app, settings = setup
    id = client.post("/api/analyses/sample", json={"sample_id": "test-case"}).json()[
        "id"
    ]
    base = f"/api/analyses/{id}"
    assert client.get(base + "/download/report").status_code == 409
    assert client.get(base + "/slice?overlay=predicted").status_code == 409
    assert client.get(base + "/slice?index=10").status_code == 422
    assert client.get(base + "/slice?plane=coronal&gradcam=true").status_code == 422
    for plane in ("axial", "coronal", "sagittal"):
        response = client.get(
            base + "/slice", params={"plane": plane, "index": 2, "overlay": "reference"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")
    folder = settings.storage / "jobs" / id
    (folder / "report.json").write_text('{"verified": true}')
    volume(folder / "tumor_mask.nii.gz")
    app.state.store.update(id, "completed", report={"verified": True})
    assert client.get(base + "/download/report").json() == {"verified": True}
    assert client.get(base + "/download/mask").content[:2] == b"\x1f\x8b"
    assert client.get(base + "/slice?overlay=predicted").status_code == 200


def test_upload_rejects_geometry_nonfinite_and_unsafe_names(setup, tmp_path):
    client, _, settings = setup
    files = {}
    for name in ("flair", "t1", "t1ce", "t2"):
        path = tmp_path / f"{name}.nii.gz"
        volume(path, affine=np.diag([2.0, 1.0, 1.0, 1.0]) if name == "t2" else None)
        files[name] = ("../../outside.nii.gz", path.read_bytes(), "application/gzip")
    response = client.post("/api/analyses/upload", files=files)
    assert response.status_code == 422
    assert "affine" in response.json()["detail"]
    assert not list((settings.storage / "jobs").iterdir())
    path = tmp_path / "t2.nii.gz"
    data = np.ones((8, 9, 10), dtype=np.float32)
    data[0, 0, 0] = np.nan
    volume(path, data=data)
    files["t2"] = ("t2.nii.gz", path.read_bytes())
    assert (
        "non-finite"
        in client.post("/api/analyses/upload", files=files).json()["detail"]
    )
    volume(path)
    files["t2"] = ("t2.nii.gz", path.read_bytes())
    result = client.post("/api/analyses/upload", files=files)
    assert result.status_code == 202
    folder = settings.storage / "jobs" / result.json()["id"]
    assert sorted(p.name for p in folder.iterdir()) == [
        "flair.nii.gz",
        "t1.nii.gz",
        "t1ce.nii.gz",
        "t2.nii.gz",
    ]


def test_compressed_size_and_cors(setup):
    client, _, settings = setup
    settings.max_expanded_bytes = 20
    assert (
        client.post("/api/analyses/sample", json={"sample_id": "test-case"}).status_code
        == 422
    )
    response = client.get("/api/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in response.headers
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cancelled_explanation_does_not_stop_inference_worker(tmp_path, monkeypatch):
    import brats.inference
    from engine import Engine

    settings = Settings(storage=tmp_path / "storage", device="cpu")
    engine = Engine(settings, Store(settings.storage))
    monkeypatch.setattr(brats.inference, "TumorAnalyzer", lambda *args: object())
    monkeypatch.setattr(engine, "layout", lambda name: tmp_path)
    monkeypatch.setattr(engine, "make_cam", lambda *args: None)
    cancelled = engine.cam("cancelled", 1)
    assert cancelled.cancel()
    next_request = engine.cam("following", 2)
    engine.start()
    try:
        assert (
            next_request.result(timeout=30)
            == settings.storage / "jobs/following/cam-2.npy"
        )
        assert engine.thread.is_alive()
    finally:
        engine.close()


def test_browser_upload_with_empty_optional_reference(setup, tmp_path):
    client, _, _ = setup
    path = tmp_path / "scan.nii.gz"
    volume(path)
    files = {
        name: (f"{name}.nii.gz", path.read_bytes(), "application/gzip")
        for name in ("flair", "t1", "t1ce", "t2")
    }
    files["seg"] = ("", b"", "application/octet-stream")
    response = client.post("/api/analyses/upload", files=files)
    assert response.status_code == 202, response.text
    assert response.json()["metadata"]["has_reference"] is False


def test_non_millimeter_units_rejected(setup):
    client, _, settings = setup
    path = settings.samples / "test-case/test-case_t2.nii.gz"
    image = nib.load(path)
    data = image.get_fdata()
    replacement = nib.Nifti1Image(data, image.affine)
    replacement.header.set_xyzt_units("meter")
    nib.save(replacement, path)
    response = client.post("/api/analyses/sample", json={"sample_id": "test-case"})
    assert response.status_code == 422
    assert "millimeters" in response.json()["detail"]


def test_reference_comparison_includes_voxels_outside_brain(tmp_path):
    from imaging import expert_comparison

    mask = tmp_path / "tumor_mask.nii.gz"
    reference = tmp_path / "reference.nii.gz"
    volume(mask, data=np.zeros((8, 9, 10), np.float32))
    labels = np.zeros((8, 9, 10), np.float32)
    labels[0, 0, 0] = 4
    volume(reference, data=labels)
    metrics = expert_comparison({"seg": reference}, tmp_path)
    assert metrics["dice_WT"] == 0.0
    assert metrics["gt_empty_WT"] is False


def test_full_queue_preserves_failed_upload_for_viewing(setup, tmp_path, monkeypatch):
    import queue

    client, app, settings = setup

    def full(_):
        raise queue.Full

    monkeypatch.setattr(app.state.engine, "submit", full)
    path = tmp_path / "scan.nii.gz"
    volume(path)
    files = {
        name: (f"{name}.nii.gz", path.read_bytes(), "application/gzip")
        for name in ("flair", "t1", "t1ce", "t2")
    }
    response = client.post("/api/analyses/upload", files=files)
    assert response.status_code == 503
    item = client.get("/api/analyses").json()[0]
    assert item["status"] == "failed"
    assert (settings.storage / "jobs" / item["id"] / "flair.nii.gz").exists()
    assert client.get(f"/api/analyses/{item['id']}/slice?index=2").status_code == 200
