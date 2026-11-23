"""Colab / Google Drive environment, project folders, devices, seeds and reproducibility logs."""
from __future__ import annotations

import datetime
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .fileio import save_json


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


class ProjectPaths:
    """Every folder the project uses. Quick-run data and runs never mix with the real ones."""

    def __init__(self, project_dir: str | Path, work_dir: str | Path = '/content/work', quick: bool = False):
        self.quick = bool(quick)
        self.project = Path(project_dir)
        self.code = self.project / 'code'
        self.raw = self.project / 'data' / 'raw'                    # shared by quick and full runs
        self.data = self.project / ('data_quick' if quick else 'data')
        self.preprocessed = self.data / 'preprocessed'
        self.meta = self.data / 'meta'
        self.data_figures = self.data / 'figures'
        self.data_logs = self.data / 'logs'
        self.runs = self.project / 'runs'
        # Fast local disk of the Colab machine (temporary).
        self.work = Path(work_dir)
        self.work_download = self.work / 'download'
        self.work_raw_cases = self.work / ('raw_cases_quick' if quick else 'raw_cases')
        self.work_shards = self.work / 'shards_tmp'
        self.work_cases = self.work / ('npz_quick' if quick else 'npz')

    def make(self) -> 'ProjectPaths':
        for folder in (self.raw, self.preprocessed, self.meta, self.data_figures, self.data_logs, self.runs, self.work):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    def run_path(self, task: str, run_name: str) -> Path:
        """Folder of a run, without creating it (quick runs get a '_quick' suffix)."""
        return self.runs / task / (run_name + ('_quick' if self.quick else ''))

    def run_dir(self, task: str, run_name: str) -> Path:
        """Create and return runs/<task>/<run_name>[_quick]/ with its standard sub-folders."""
        run_dir = self.run_path(task, run_name)
        for sub in ('checkpoints', 'figures', 'predictions', 'export', 'logs'):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return run_dir

    def __repr__(self) -> str:
        rows = [f'  {key:16s} {value}' for key, value in self.__dict__.items()]
        return 'ProjectPaths(\n' + '\n'.join(rows) + '\n)'


def get_device(allow_mps: bool = False):
    """CUDA if available; Apple GPU (MPS) only when allowed (used by the offline app); else CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if allow_mps and getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def require_gpu(device, quick_run: bool) -> None:
    """Full runs must use a GPU; a quick run may fall back to the CPU (slow but fine for a smoke test)."""
    import torch
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print('GPU:', torch.cuda.get_device_name(0))
    elif quick_run:
        print('WARNING: no GPU found - running the quick test on the CPU (slow but fine for a smoke test).')
    else:
        raise RuntimeError('No GPU. Click Runtime -> Change runtime type -> T4 GPU, then Run all again.')


def set_seed(seed: int = 42) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_num_workers(max_workers: int = 4, min_shm_gb: float = 1.0) -> int:
    """DataLoader workers pass batches through /dev/shm; if it is tiny they crash, so use 0 workers then."""
    if sys.platform.startswith('linux') and os.path.exists('/dev/shm'):
        shm_gb = shutil.disk_usage('/dev/shm').total / 1e9
        if shm_gb < min_shm_gb:
            print(f'WARNING: /dev/shm is only {shm_gb:.2f} GB - loading data in the main process (slower but safe).')
            return 0
    return max(0, min(max_workers, os.cpu_count() or 2))


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def environment_info() -> dict:
    info = OrderedDict()
    info['time'] = datetime.datetime.now().isoformat(timespec='seconds')
    info['python'] = sys.version.split()[0]
    info['platform'] = platform.platform()
    info['in_colab'] = in_colab()
    info['cpu_count'] = os.cpu_count()
    try:
        import psutil
        info['ram_gb'] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        info['ram_gb'] = None
    info['disk_free_gb_root'] = round(shutil.disk_usage('/').free / 1e9, 1)
    try:
        import torch
        info['torch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        info['cuda'] = torch.version.cuda
        info['cudnn'] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info['gpu'] = props.name
            info['gpu_mem_gb'] = round(props.total_memory / 1e9, 1)
    except ImportError as error:
        info['torch_error'] = str(error)
    for pkg in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'nibabel', 'matplotlib', 'monai', 'timm', 'tqdm'):
        info[pkg] = package_version(pkg)
    return info


def save_environment(out_dir: Path) -> dict:
    """Store Python / GPU / package details next to the results (for reproducibility)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    info = environment_info()
    save_json(info, out_dir / f'environment_{stamp}.json')
    try:
        freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, timeout=180)
        (out_dir / f'pip_freeze_{stamp}.txt').write_text(freeze.stdout)
    except (OSError, subprocess.SubprocessError) as error:
        print('pip freeze failed:', error)
    if shutil.which('nvidia-smi'):
        try:
            smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=60)
            (out_dir / f'nvidia_smi_{stamp}.txt').write_text(smi.stdout)
        except (OSError, subprocess.SubprocessError) as error:
            print('nvidia-smi failed:', error)
    return info


def print_environment(info: dict) -> None:
    for key in ('python', 'cpu_count', 'ram_gb', 'disk_free_gb_root', 'torch', 'gpu'):
        print(f'{key:18s} {info.get(key)}')


def snapshot_code(run_dir: Path, code_dir: Path, notebook_name: str | None = None) -> Path:
    """Copy the project code (and, on Colab, the running notebook) into the run folder."""
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = Path(run_dir) / 'code_snapshot' / stamp
    dst.mkdir(parents=True, exist_ok=True)
    code_dir = Path(code_dir)
    for script in code_dir.glob('*.py'):
        shutil.copy2(script, dst / script.name)
    package_dir = code_dir / 'brats'
    if package_dir.is_dir():
        shutil.copytree(package_dir, dst / 'brats', ignore=shutil.ignore_patterns('__pycache__'))
    if notebook_name and in_colab():
        try:
            from google.colab import _message
            notebook = _message.blocking_request('get_ipynb', request='', timeout_sec=60)
            if notebook and 'ipynb' in notebook:
                (dst / notebook_name).write_text(json.dumps(notebook['ipynb']))
        except Exception as error:     # Colab-internal API; never let it stop a run
            print('(Could not snapshot the notebook itself:', error, ')')
    return dst


def flush_drive() -> None:
    """Make sure every file written to Google Drive is uploaded (Colab only)."""
    if in_colab() and os.path.isdir('/content/drive'):
        from google.colab import drive
        print('Flushing Google Drive (this waits until all files are uploaded)...')
        drive.flush_and_unmount()
        print('Done. Drive is unmounted; re-run the setup cell if you want to continue working.')
    else:
        print('Not on Colab - nothing to flush.')


def print_header(text: str) -> None:
    print('\n' + '=' * 78 + f'\n{text}\n' + '=' * 78)
