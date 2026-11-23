import argparse
import json
import time

from fastapi.testclient import TestClient
from main import create_app
from settings import Settings

parser = argparse.ArgumentParser()
parser.add_argument("--sample", default="BraTS2021_00495")
parser.add_argument("--timeout", type=int, default=1800)
args = parser.parse_args()
started = time.monotonic()
with TestClient(create_app(Settings())) as client:
    while True:
        health = client.get("/api/health").json()["models"]
        if health["status"] == "error":
            raise RuntimeError(health["error"])
        if health["status"] == "ready":
            break
        if time.monotonic() - started > args.timeout:
            raise TimeoutError("Model loading timed out")
        time.sleep(1)
    response = client.post("/api/analyses/sample", json={"sample_id": args.sample})
    response.raise_for_status()
    id = response.json()["id"]
    while True:
        result = client.get(f"/api/analyses/{id}").json()
        if result["status"] in ("completed", "failed"):
            break
        if time.monotonic() - started > args.timeout:
            raise TimeoutError("Inference timed out")
        time.sleep(2)
    if result["status"] != "completed":
        raise RuntimeError(result["error"])
    for kind in ("report", "mask"):
        output = client.get(f"/api/analyses/{id}/download/{kind}")
        output.raise_for_status()
        assert output.content
    for plane in ("axial", "coronal", "sagittal"):
        output = client.get(
            f"/api/analyses/{id}/slice",
            params={"plane": plane, "index": 75, "overlay": "predicted"},
        )
        output.raise_for_status()
        assert output.content.startswith(b"\x89PNG")
    output = client.get(
        f"/api/analyses/{id}/slice", params={"index": 75, "gradcam": "true"}
    )
    output.raise_for_status()
    assert output.content.startswith(b"\x89PNG")
    print(
        json.dumps(
            {"id": id, "status": result["status"], "report": result["report"]}, indent=2
        )
    )
