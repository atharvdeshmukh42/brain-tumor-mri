import asyncio
import queue
import shutil
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID, uuid4

import numpy as np
from engine import Engine
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from imaging import input_files, render_slice, validate_files
from pydantic import BaseModel, ConfigDict
from settings import MODALITIES, Settings
from store import Store


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str


def discover(settings):
    if not settings.samples.is_dir():
        return {}
    return {
        folder.name: {
            key: folder / f"{folder.name}_{key}.nii.gz"
            for key in (*MODALITIES, "seg")
            if (folder / f"{folder.name}_{key}.nii.gz").is_file()
        }
        for folder in settings.samples.iterdir()
        if folder.is_dir()
        and all(
            (folder / f"{folder.name}_{key}.nii.gz").is_file() for key in MODALITIES
        )
    }


def create_app(settings=None, engine_factory=Engine):
    settings = settings or Settings()
    store = Store(settings.storage)
    engine = engine_factory(settings, store)

    @asynccontextmanager
    async def lifespan(app):
        store.recover_interrupted()
        engine.start()
        yield
        await asyncio.to_thread(engine.close)

    app = FastAPI(title="NeuroScope local research API", lifespan=lifespan)
    app.state.store, app.state.engine = store, engine
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def record(id):
        item = store.get(str(id))
        if item is None:
            raise HTTPException(404, "Analysis not found")
        return item

    def require_ready():
        if engine.status["status"] != "ready":
            raise HTTPException(
                503, engine.status["error"] or "Models are loading. Try again shortly."
            )

    def enqueue(id, source, folder):
        try:
            metadata = validate_files(input_files(folder), settings)
        except Exception as exc:
            shutil.rmtree(folder)
            raise HTTPException(422, f"Invalid NIfTI input: {exc}") from exc
        item = store.create(id, source, metadata)
        try:
            engine.submit(id)
        except queue.Full:
            store.update(id, "failed", error="Analysis queue is full. Retry later.")
            raise HTTPException(503, "Analysis queue is full")
        return item

    @app.get("/api/health")
    def health():
        return {"models": engine.status, "research_only": True}

    @app.get("/api/samples")
    def samples():
        return [
            {"id": key, "has_reference": "seg" in files}
            for key, files in sorted(discover(settings).items())
        ]

    @app.post("/api/analyses/sample", status_code=202)
    def sample(request: SampleRequest):
        require_ready()
        files = discover(settings).get(request.sample_id)
        if files is None:
            raise HTTPException(404, "Unknown sample ID")
        id = str(uuid4())
        folder = settings.storage / "jobs" / id
        folder.mkdir(parents=True)
        for key, path in files.items():
            shutil.copyfile(path, folder / f"{key}.nii.gz")
        return enqueue(id, request.sample_id, folder)

    @app.post("/api/analyses/upload", status_code=202)
    def upload(
        flair: Annotated[UploadFile, File()],
        t1: Annotated[UploadFile, File()],
        t1ce: Annotated[UploadFile, File()],
        t2: Annotated[UploadFile, File()],
        seg: Annotated[UploadFile | str | None, File()] = None,
    ):
        require_ready()
        if isinstance(seg, str):
            if seg:
                raise HTTPException(422, "seg: expected a NIfTI file")
            seg = None
        id = str(uuid4())
        folder = settings.storage / "jobs" / id
        folder.mkdir(parents=True)
        try:
            for key, file in {
                "flair": flair,
                "t1": t1,
                "t1ce": t1ce,
                "t2": t2,
                "seg": seg,
            }.items():
                if file is None or (key == "seg" and not file.filename):
                    continue
                filename = file.filename or ""
                suffix = (
                    ".nii.gz"
                    if filename.endswith(".nii.gz")
                    else ".nii"
                    if filename.endswith(".nii")
                    else None
                )
                if suffix is None:
                    raise HTTPException(422, f"{key}: use .nii or .nii.gz")
                total = 0
                with (folder / f"{key}{suffix}").open("wb") as output:
                    while chunk := file.file.read(1024 * 1024):
                        total += len(chunk)
                        if total > settings.max_file_bytes:
                            raise HTTPException(413, f"{key}: exceeds 128 MiB limit")
                        output.write(chunk)
            return enqueue(id, "Uploaded case", folder)
        except Exception:
            if store.get(id) is None:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            for file in (flair, t1, t1ce, t2, seg):
                if file is not None:
                    file.file.close()

    @app.get("/api/analyses")
    def history():
        return store.history()

    @app.get("/api/analyses/{id}")
    def get_analysis(id: UUID):
        return record(id)

    @app.get("/api/analyses/{id}/slice")
    async def slice_image(
        id: UUID,
        modality: Literal["flair", "t1", "t1ce", "t2"] = "flair",
        plane: Literal["axial", "coronal", "sagittal"] = "axial",
        index: int = Query(0, ge=0),
        overlay: Literal["none", "predicted", "reference"] = "none",
        opacity: float = Query(0.5, ge=0, le=1),
        gradcam: bool = False,
    ):
        item = record(id)
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[plane]
        if index >= item["metadata"]["shape"][axis]:
            raise HTTPException(422, "Slice index outside volume")
        if gradcam and plane != "axial":
            raise HTTPException(422, "Grad-CAM is available for axial slices only")
        if (overlay == "predicted" or gradcam) and item["status"] != "completed":
            raise HTTPException(409, "Analysis must complete first")
        folder = settings.storage / "jobs" / str(id)
        try:
            cam = None
            if gradcam:
                future = engine.cam(str(id), index)
                cam = np.load(await asyncio.wrap_future(future), allow_pickle=False)
            content = await asyncio.to_thread(
                render_slice,
                input_files(folder),
                folder,
                modality,
                plane,
                index,
                overlay,
                opacity,
                cam,
            )
            return Response(
                content,
                media_type="image/png",
                headers={"Cache-Control": "private, max-age=300"},
            )
        except queue.Full as exc:
            raise HTTPException(503, "Inference worker queue is full") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"Could not render slice: {exc}") from exc

    @app.get("/api/analyses/{id}/download/{kind}")
    def download(id: UUID, kind: Literal["report", "mask"]):
        item = record(id)
        if item["status"] != "completed":
            raise HTTPException(409, "Analysis must complete first")
        filename = "report.json" if kind == "report" else "tumor_mask.nii.gz"
        return FileResponse(
            settings.storage / "jobs" / str(id) / filename,
            filename=f"{id}-{filename}",
            media_type="application/json" if kind == "report" else "application/gzip",
        )

    return app


app = create_app()
