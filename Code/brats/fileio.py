"""
Safe file writing.

Google Drive on Colab occasionally fails with transient I/O errors, and a disconnect can interrupt a write.
Every writer here writes to a temporary file first and then renames it into place, so a file on Drive is
either the complete old version or the complete new version, never half a file.
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays, paths, sets and dates into JSON-friendly values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return str(value)


def to_jsonable(obj: Any) -> Any:
    """Deep copy of obj that contains only plain Python types (safe for json and torch weights_only)."""
    return json.loads(json.dumps(obj, default=_json_default))


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    """Move tmp_path to final_path in one step (falls back to copy + delete across file systems)."""
    try:
        os.replace(tmp_path, final_path)
    except OSError:
        shutil.copyfile(tmp_path, final_path)
        os.remove(tmp_path)


def with_retries(fn: Callable, *args, tries: int = 5, wait_s: float = 3.0, desc: str = 'operation', **kwargs) -> Any:
    """Call fn, retrying OSErrors with a growing pause (Google Drive sometimes fails for a moment)."""
    last_error: OSError | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except OSError as error:
            last_error = error
            print(f'  [retry {attempt}/{tries}] {desc} failed: {error}')
            time.sleep(wait_s * attempt)
    raise last_error


def _write_atomically(path: Path, write: Callable[[Path], None], desc: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def attempt():
        tmp_path = path.with_name(path.name + '.tmp')
        write(tmp_path)
        atomic_replace(tmp_path, path)
    with_retries(attempt, desc=desc)
    return path


def save_json(obj: Any, path: Path) -> Path:
    return _write_atomically(path, lambda tmp: tmp.write_text(json.dumps(obj, indent=2, default=_json_default)),
                             f'write {Path(path).name}')


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    return _write_atomically(path, lambda tmp: df.to_csv(tmp, index=index), f'write {Path(path).name}')


def save_npz(path: Path, **arrays: np.ndarray) -> Path:
    """Compressed .npz, written atomically."""
    def write(tmp: Path):
        with open(tmp, 'wb') as f:
            np.savez_compressed(f, **arrays)
    return _write_atomically(path, write, f'write {Path(path).name}')


def save_torch(obj: Any, path: Path) -> Path:
    import torch
    return _write_atomically(path, lambda tmp: torch.save(obj, tmp), f'save {Path(path).name}')


def copy_verified(src: Path, dst: Path, tries: int = 5, progress: bool = False, chunk_mb: int = 16) -> Path:
    """Copy a file through a '.partial' name, check the size, then rename it into place."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size

    def attempt():
        tmp_path = dst.with_name(dst.name + '.partial')
        copied, next_report, start = 0, 0.0, time.time()
        with open(src, 'rb') as fin, open(tmp_path, 'wb') as fout:
            while chunk := fin.read(chunk_mb * 1024 * 1024):
                fout.write(chunk)
                copied += len(chunk)
                if progress and size > 0 and copied / size >= next_report:
                    rate_mb_s = copied / 1e6 / max(time.time() - start, 1e-6)
                    print(f'    {src.name}: {copied / 1e9:6.2f} / {size / 1e9:.2f} GB ({rate_mb_s:.0f} MB/s)')
                    next_report += 0.1
            fout.flush()
            try:
                os.fsync(fout.fileno())
            except OSError:      # some network file systems do not support fsync
                pass
        copied_size = tmp_path.stat().st_size
        if copied_size != size:
            raise OSError(f'size mismatch after copy ({copied_size} != {size} bytes)')
        atomic_replace(tmp_path, dst)
    with_retries(attempt, tries=tries, desc=f'copy {src.name}')
    return dst


def disk_free_gb(path: Path) -> float:
    """Free space on the disk that holds path (or its nearest existing parent)."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def require_disk(path: Path, need_gb: float, what: str) -> float:
    """Stop with a clear message if there is not enough free disk for the next step."""
    free_gb = disk_free_gb(path)
    print(f'Disk check for {what}: need ~{need_gb:.1f} GB, free {free_gb:.1f} GB at {path}')
    if free_gb < need_gb:
        raise RuntimeError(
            f'Not enough local disk for {what}: need ~{need_gb:.1f} GB but only {free_gb:.1f} GB is free.\n'
            'Fix: Runtime -> Change runtime type -> pick a runtime with more disk (a CPU runtime has the most), '
            'or delete old files in /content.')
    return free_gb


def dir_size_gb(path: Path) -> float:
    total_bytes = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total_bytes / 1e9
