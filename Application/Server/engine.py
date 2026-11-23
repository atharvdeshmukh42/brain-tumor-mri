import json
import logging
import queue
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Code"))

from imaging import expert_comparison, input_files

logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.tasks = queue.Queue(maxsize=32)
        self.status = {"status": "loading", "device": None, "error": None}
        self.thread = threading.Thread(
            target=self.run, daemon=True, name="mri-inference"
        )

    def start(self):
        self.thread.start()

    def close(self):
        self.tasks.put(None)
        self.thread.join()

    def submit(self, id):
        self.tasks.put_nowait(("analyze", id, None, None))

    def cam(self, id, index):
        future = Future()
        self.tasks.put_nowait(("cam", id, index, future))
        return future

    def layout(self, name):
        source = self.settings.models / name
        target = self.settings.storage / "model-cache" / name
        (target / "checkpoints").mkdir(parents=True, exist_ok=True)
        for src, dest in [
            (source / "best.pt", target / "checkpoints/best.pt"),
            (source / "model_card.json", target / "model_card.json"),
        ]:
            if not src.is_file():
                raise FileNotFoundError(f"Missing model asset: {src}")
            if dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src)
        return target

    def run(self):
        try:
            import torch
            from brats.inference import TumorAnalyzer

            requested = self.settings.device
            if requested == "auto":
                requested = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            if requested not in ("cuda", "mps", "cpu"):
                raise ValueError("MRI_DEVICE must be auto, cuda, mps, or cpu")
            if (
                requested == "cuda"
                and not torch.cuda.is_available()
                or requested == "mps"
                and not torch.backends.mps.is_available()
            ):
                raise ValueError(
                    f"Requested device {requested} is unavailable. Set MRI_DEVICE=cpu."
                )
            torch.set_num_threads(min(torch.get_num_threads(), 8))
            self.analyzer = TumorAnalyzer(
                self.layout("effb0"), self.layout("segresnet"), torch.device(requested)
            )
            self.status = {"status": "ready", "device": requested, "error": None}
        except Exception as exc:
            logger.exception("Model initialization failed")
            self.status = {
                "status": "error",
                "device": None,
                "error": f"{exc}. Check MRI_MODELS_DIR and MRI_DEVICE.",
            }
        while (task := self.tasks.get()) is not None:
            kind, id, index, future = task
            if future is not None and not future.set_running_or_notify_cancel():
                continue
            folder = self.settings.storage / "jobs" / id
            try:
                if self.status["status"] != "ready":
                    raise RuntimeError(self.status["error"])
                files = input_files(folder)
                if kind == "analyze":
                    self.store.update(id, "running")
                    report = self.analyzer.analyze(files, folder)
                    if "seg" in files:
                        report["comparison_with_expert"] = expert_comparison(
                            files, folder
                        )
                    report = json.loads(
                        json.dumps(report), parse_constant=lambda _: None
                    )
                    report["input"] = {name: path.name for name, path in files.items()}
                    (folder / "report.json").write_text(
                        json.dumps(report, indent=2, allow_nan=False)
                    )
                    self.store.update(id, "completed", report=report)
                else:
                    self.make_cam(files, folder, index)
                    future.set_result(folder / f"cam-{index}.npy")
            except Exception as exc:
                logger.exception("Job %s failed", id)
                if future is not None:
                    future.set_exception(exc)
                else:
                    self.store.update(
                        id,
                        "failed",
                        error=f"{exc}. If this is a device error, restart with MRI_DEVICE=cpu and retry.",
                    )

    def make_cam(self, files, folder, index):
        import numpy as np
        import torch
        from brats.case import prepare_case_from_files
        from brats.detection import detection_inputs
        from brats.xai import gradcam

        path = folder / f"cam-{index}.npy"
        if path.exists():
            return
        case = prepare_case_from_files(files)
        z = index - case.crop_origin[2]
        if not 0 <= z < case.image.shape[3]:
            np.save(path, np.zeros(tuple(case.orig_shape[:2]), np.float32))
            return
        inputs = detection_inputs(case, [z])
        heat, _ = gradcam(
            self.analyzer.detector, torch.from_numpy(inputs).to(self.analyzer.device)
        )
        if heat[0].shape != tuple(case.orig_shape[:2]):
            raise ValueError("Grad-CAM requires the 240 × 240 BraTS input grid")
        np.save(path, heat[0])
