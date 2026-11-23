"""Raw BraTS data: Kaggle credentials and download, and extraction of the NIfTI files from the archives."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import BinaryIO, Callable

from .config import ARCHIVE_SUFFIXES, CASE_FILE_KEYS, CASE_FILE_PATTERN, MODALITIES
from .fileio import atomic_replace

KAGGLE_HELP = """
Kaggle credentials were not found. Do ONE of these, then re-run the cell:

  A) Colab secrets (recommended): click the key icon in the left sidebar of Colab,
     add KAGGLE_USERNAME and KAGGLE_KEY (from kaggle.com -> Settings -> API -> Create New Token)
     and switch on "Notebook access" for both.
  B) Put the kaggle.json file you downloaded from Kaggle into your project folder on Drive:
     MyDrive/BrainTumor/kaggle.json
  C) Skip Kaggle: download the dataset zip yourself and upload it to MyDrive/BrainTumor/data/raw/
"""

CaseFiles = dict[str, dict[str, Path]]    # {case_id: {modality or 'seg': file path}}


# -----------------------------------------------------------------------------
# Kaggle
# -----------------------------------------------------------------------------
def list_archives(folder: Path) -> list[Path]:
    """Complete dataset archives (.zip / .tar / ...) in folder, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return [path for path in sorted(folder.iterdir())
            if path.is_file() and path.name.lower().endswith(ARCHIVE_SUFFIXES)]


def setup_kaggle_credentials(project_dir: Path) -> str:
    """Find Kaggle credentials (environment, Colab secrets, or kaggle.json). Returns where they came from.

    Never prints the key.
    """
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        return 'environment variables'
    colab_source = _credentials_from_colab_secrets()
    if colab_source:
        return colab_source
    for candidate in (Path(project_dir) / 'kaggle.json', Path.home() / '.kaggle' / 'kaggle.json'):
        if not candidate.is_file():
            continue
        try:
            token = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        if token.get('username') and token.get('key'):
            _write_kaggle_json(token['username'], token['key'])
            return str(candidate)
    raise RuntimeError(KAGGLE_HELP)


def _credentials_from_colab_secrets() -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None

    def secret(name):
        try:
            return userdata.get(name)
        except Exception:        # missing secret, or notebook access switched off
            return None

    username, key = secret('KAGGLE_USERNAME'), secret('KAGGLE_KEY')
    if username and key:
        _write_kaggle_json(username, key)
        return 'Colab secrets KAGGLE_USERNAME / KAGGLE_KEY'
    api_token = secret('KAGGLE_API_TOKEN')
    if api_token:
        os.environ['KAGGLE_API_TOKEN'] = api_token
        return 'Colab secret KAGGLE_API_TOKEN'
    return None


def _write_kaggle_json(username: str, key: str) -> None:
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_file = kaggle_dir / 'kaggle.json'
    token_file.write_text(json.dumps({'username': username, 'key': key}))
    os.chmod(token_file, 0o600)


def _run_streaming(cmd: list[str], progress_every_s: float = 15.0) -> tuple[int, str]:
    """Run a command and show its output in the notebook; progress-bar lines are shown at most every few seconds."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, buffer, last_progress = [], '', 0.0
    while True:
        char = proc.stdout.read(1)
        if char == '':
            if proc.poll() is not None:
                break
            continue
        if char not in '\r\n':
            buffer += char
            continue
        line, buffer = buffer.strip(), ''
        if not line:
            continue
        lines.append(line)
        is_progress_bar = '%|' in line
        if not is_progress_bar or time.time() - last_progress > progress_every_s:
            print('   ', line)
            if is_progress_bar:
                last_progress = time.time()
    if buffer.strip():
        lines.append(buffer.strip())
        print('   ', buffer.strip())
    return proc.wait(), '\n'.join(lines)


def kaggle_download(dataset: str, out_dir: Path) -> list[Path]:
    """Download a Kaggle dataset into out_dir with the kaggle command-line tool; returns the archives."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which('kaggle')
    if exe is None:
        print('Installing the kaggle command line tool...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'])
        exe = shutil.which('kaggle')
    if exe is None:
        raise RuntimeError('The kaggle command was not found after installing it.')
    print(f'Downloading {dataset} from Kaggle into {out_dir} (about 12-13 GB, usually 5-15 minutes)...')
    code, output = _run_streaming([exe, 'datasets', 'download', '-d', dataset, '-p', str(out_dir)])
    if code != 0 and 'unrecognized arguments' in output:     # newer kaggle versions dropped "-d"
        code, output = _run_streaming([exe, 'datasets', 'download', dataset, '-p', str(out_dir)])
    if code != 0:
        hint = ''
        if '401' in output or 'Unauthorized' in output:
            hint = 'Your Kaggle username/key are wrong or expired - create a new token on kaggle.com.'
        elif '403' in output:
            hint = 'Open the dataset page on kaggle.com while logged in and accept its terms, then retry.'
        elif '404' in output:
            hint = f'Dataset "{dataset}" was not found - check KAGGLE_DATASET.'
        raise RuntimeError(f'Kaggle download failed (exit code {code}). {hint}\nLast output:\n{output[-2000:]}')
    archives = list_archives(out_dir)
    if not archives:
        raise RuntimeError(f'Kaggle reported success but no archive was found in {out_dir}.')
    for archive in archives:
        print(f'  downloaded {archive.name}: {archive.stat().st_size / 1e9:.2f} GB')
    return archives


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
class _Extraction:
    """Bookkeeping while walking through (possibly nested) archives."""

    MAX_NESTING = 3

    def __init__(self, out_dir: Path, wanted_ids: list[str] | None = None, max_cases: int | None = None):
        self.out_dir = Path(out_dir)
        self.wanted = set(wanted_ids) if wanted_ids is not None else None
        self.max_cases = max_cases
        self.cases: CaseFiles = {}
        self.duplicates = 0
        self.ignored = 0
        self.written = 0
        self.start = time.time()

    def wants(self, case_id: str) -> bool:
        if self.wanted is not None:
            return case_id in self.wanted
        if self.max_cases is None or case_id in self.cases:
            return True
        return len(self.cases) < self.max_cases

    def finished(self) -> bool:
        """True once every requested case has all 5 files (lets quick runs stop reading early)."""
        if self.wanted is not None:
            target = self.wanted
        elif self.max_cases is not None and len(self.cases) >= self.max_cases:
            target = self.cases.keys()
        else:
            return False
        return all(len(self.cases.get(case_id, {})) >= len(CASE_FILE_KEYS) for case_id in target)

    def write_case_file(self, fileobj: BinaryIO, case_id: str, key: str, gzipped: bool) -> None:
        if key in self.cases.get(case_id, {}):
            self.duplicates += 1
            return
        case_dir = self.out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        dst = case_dir / (f'{case_id}_{key}.nii' + ('.gz' if gzipped else ''))
        tmp_path = dst.with_name(dst.name + '.partial')
        with open(tmp_path, 'wb') as f:
            shutil.copyfileobj(fileobj, f, 8 * 1024 * 1024)
        atomic_replace(tmp_path, dst)
        self.cases.setdefault(case_id, {})[key] = dst
        self.written += 1
        if self.written % 500 == 0:
            print(f'    extracted {self.written} files from {len(self.cases)} cases ({time.time() - self.start:.0f} s)')

    def handle_member(self, name: str, open_member: Callable[[], BinaryIO], depth: int) -> None:
        base = os.path.basename(name.rstrip('/'))
        if not base or base.startswith('.'):          # skips macOS "._" files and hidden files
            return
        lower = base.lower()
        if lower.endswith(('.tar', '.tar.gz', '.tgz')):
            if depth <= self.MAX_NESTING:
                with open_member() as f:
                    self.read_tar_stream(f, depth + 1)
            return
        if lower.endswith('.zip'):
            if depth <= self.MAX_NESTING:
                # Nested zips need random access - spool to a temporary file first.
                tmp_path = self.out_dir / f'.nested_{depth}_{base}'
                with open_member() as f, open(tmp_path, 'wb') as out:
                    shutil.copyfileobj(f, out, 8 * 1024 * 1024)
                try:
                    self.read_zip(tmp_path, depth + 1)
                finally:
                    tmp_path.unlink(missing_ok=True)
            return
        match = CASE_FILE_PATTERN.match(base)
        if not match:
            self.ignored += 1
            return
        case_id, key = match.group('case'), match.group('mod').lower()
        if self.wants(case_id):
            with open_member() as f:
                self.write_case_file(f, case_id, key, lower.endswith('.gz'))

    def read_tar_stream(self, fileobj: BinaryIO, depth: int) -> None:
        with tarfile.open(fileobj=fileobj, mode='r|*') as tar:
            for member in tar:
                if not member.isfile():
                    continue
                self.handle_member(member.name, lambda m=member: tar.extractfile(m), depth)
                if self.finished():
                    break

    def read_zip(self, path: Path, depth: int) -> None:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                self.handle_member(info.filename, lambda i=info: archive.open(i), depth)
                if self.finished():
                    break


def extract_brats_archives(archives: list[Path], out_dir: Path, wanted_ids: list[str] | None = None,
                           max_cases: int | None = None) -> CaseFiles:
    """Extract BraTS NIfTI files from .zip / .tar archives (nested archives are fine).

    Files are written flat as out_dir/<case>/<case>_<modality>.nii.gz, so the result does not depend
    on how the archive was organised. Files extracted earlier in the same Colab session are re-used.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = _Extraction(out_dir, wanted_ids, max_cases)
    for case_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        for file in case_dir.iterdir():
            match = CASE_FILE_PATTERN.match(file.name)
            if match:
                job.cases.setdefault(match.group('case'), {})[match.group('mod').lower()] = file
    if job.cases:
        print(f'  {len(job.cases)} cases already present in {out_dir}')
    if job.finished() and (wanted_ids is not None or max_cases is not None):
        return {case_id: files for case_id, files in job.cases.items() if job.wants(case_id)}
    for archive in map(Path, archives):
        print(f'  reading {archive.name} ...')
        if archive.name.lower().endswith('.zip'):
            job.read_zip(archive, 0)
        else:
            with open(archive, 'rb') as f:
                job.read_tar_stream(f, 0)
        if job.finished():
            break
    print(f'  extraction done: {len(job.cases)} cases, {job.written} new files, '
          f'{job.duplicates} duplicate files skipped, {job.ignored} other files ignored, '
          f'{time.time() - job.start:.0f} s')
    cases = job.cases
    if wanted_ids is not None:
        cases = {case_id: files for case_id, files in cases.items() if case_id in job.wanted}
    elif max_cases is not None:
        cases = dict(sorted(cases.items())[:max_cases])
    return cases


def check_case_files(case_files: CaseFiles, require_seg: bool = True) -> tuple[CaseFiles, dict[str, str]]:
    """Split cases into complete ones and incomplete ones ({case_id: 'missing ...'})."""
    needed = CASE_FILE_KEYS if require_seg else MODALITIES
    complete, incomplete = {}, {}
    for case_id, files in sorted(case_files.items()):
        missing = [key for key in needed if key not in files]
        if missing:
            incomplete[case_id] = 'missing ' + ', '.join(missing)
        else:
            complete[case_id] = files
    return complete, incomplete
