from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from main import create_app
from settings import Settings
from store import Store


class IdleEngine:
    """Keep jobs queued to exercise the HTTP boundary without model inference."""

    def __init__(self, settings, store):
        self.status = {"status": "ready", "device": "cpu", "error": None}

    def start(self):
        pass

    def close(self):
        pass

    def submit(self, id):
        pass


def volume(path, *, affine=None, data=None):
    nib.save(
        nib.Nifti1Image(
            np.ones((8, 9, 10), dtype=np.float32) if data is None else data,
            np.eye(4) if affine is None else affine,
        ),
        path,
    )


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
    client, app, settings = setup
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
