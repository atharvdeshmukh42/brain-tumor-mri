"""
The whole dataset: resumable preprocessing in shards, dataset statistics, the patient-level split,
and copying the preprocessed data from Drive to the fast local Colab disk.

Storage layout on Drive (data/ or data_quick/):
    preprocessed/shard_000.tar          100 patients, one <case_id>.npz each (see case.PreparedCase)
    preprocessed/shard_000.done.json    written LAST: a shard without it is redone after a disconnect
    preprocessed/shard_000_cases.csv    one row of statistics per patient
    preprocessed/shard_000_slices.csv   one row per axial slice (brain / tumor pixel counts)
    preprocessed/shard_000_stats.npz    histograms, tumor-frequency counts, preview slices
    meta/                               the merged tables, split.csv and dataset_summary.json
"""
from __future__ import annotations

import datetime
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .case import PreparedCase, crop, load_nifti_case, prepare_case, region_masks
from .config import (EXPECTED_SHAPE, FREQUENCY_MAP_DOWNSAMPLE, MODALITIES, NORM_CLIP, NORM_HIST_BINS,
                     RAW_HIST_BINS)
from .env import ProjectPaths
from .fileio import atomic_replace, copy_verified, load_json, require_disk, save_csv, save_json, save_npz
from .localization import localize_mask
from .raw_data import CaseFiles

LOCALIZATION_COLUMNS = ('n_components', 'largest_component_pct', 'centroid_i', 'centroid_j', 'centroid_k',
                        'centroid_x_mm', 'centroid_y_mm', 'centroid_z_mm', 'extent_x_mm', 'extent_y_mm',
                        'extent_z_mm', 'offset_x_mm', 'offset_y_mm', 'offset_z_mm', 'side', 'ap', 'si')


# -----------------------------------------------------------------------------
# Statistics of one patient
# -----------------------------------------------------------------------------
def _intensity_histograms(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-modality histograms of the brain voxels, raw (log bins) and after normalization."""
    raw_hist = np.zeros((len(MODALITIES), len(RAW_HIST_BINS) - 1), dtype=np.int64)
    norm_hist = np.zeros((len(MODALITIES), len(NORM_HIST_BINS) - 1), dtype=np.int64)
    for c in range(len(MODALITIES)):
        values = image[c][image[c] != 0].astype(np.float32)
        if values.size == 0:
            continue
        raw_hist[c] = np.histogram(np.clip(values, RAW_HIST_BINS[0], RAW_HIST_BINS[-1]), RAW_HIST_BINS)[0]
        z = np.clip((values - mean[c]) / std[c], -NORM_CLIP, NORM_CLIP)
        norm_hist[c] = np.histogram(z, NORM_HIST_BINS)[0]
    return raw_hist, norm_hist


def case_statistics(case_id: str, image: np.ndarray, labels: np.ndarray, affine: np.ndarray, spacing: np.ndarray,
                    case: PreparedCase) -> tuple[OrderedDict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Numbers about one patient for the dataset-analysis graphs.

    image / labels are the full original volumes; case is the prepared (cropped) version of the same patient.
    Returns the patient row, the per-slice table, and the raw / normalized intensity histograms.
    """
    voxel_cm3 = float(np.prod(spacing)) / 1000.0
    bbox = case.bbox
    brain = np.any(image != 0, axis=0)
    masks = region_masks(labels)
    row = OrderedDict(case_id=case_id, status='ok', error='')
    row['stored_dtype'] = str(case.image.dtype)
    row['shape_x'], row['shape_y'], row['shape_z'] = (int(s) for s in image.shape[1:])
    row['sp_x'], row['sp_y'], row['sp_z'] = (float(s) for s in spacing)
    for name, value in zip(('x0', 'x1', 'y0', 'y1', 'z0', 'z1'), bbox):
        row[f'bbox_{name}'] = int(value)
    row['crop_x'], row['crop_y'], row['crop_z'] = (int(s) for s in case.image.shape[1:])
    row['brain_vox'] = int(brain.sum())
    row['brain_cm3'] = row['brain_vox'] * voxel_cm3
    for region in ('WT', 'TC', 'ET', 'NCR', 'ED'):
        row[f'vol_{region.lower()}_cm3'] = float(masks[region].sum()) * voxel_cm3
    row['wt_brain_pct'] = 100.0 * masks['WT'].sum() / max(1, row['brain_vox'])
    row['et_present'] = bool(masks['ET'].any())
    row['seg_outside_crop'] = int(masks['WT'].sum() - crop(masks['WT'], bbox).sum())   # must be 0
    row['labels_found'] = ' '.join(str(int(v)) for v in np.unique(labels))
    location = localize_mask(masks['WT'], affine, spacing, brain_mask=brain)
    for key in LOCALIZATION_COLUMNS:
        row[f'wt_{key}'] = location.get(key)
    z0, z1 = int(bbox[4]), int(bbox[5])
    tumor_px_per_slice = masks['WT'].sum(axis=(0, 1))
    tumor_slices = np.where(tumor_px_per_slice > 0)[0]
    row['n_slices_crop'] = z1 - z0
    row['n_tumor_slices'] = int(tumor_slices.size)
    row['tumor_z_min'] = int(tumor_slices[0]) if tumor_slices.size else -1
    row['tumor_z_max'] = int(tumor_slices[-1]) if tumor_slices.size else -1
    for c, modality in enumerate(MODALITIES):
        values = case.image[c][case.image[c] != 0].astype(np.float32)
        if values.size:
            row[f'raw_{modality}_mean'] = float(values.mean())
            row[f'raw_{modality}_std'] = float(values.std())
            p01, p50, p99 = (float(x) for x in np.percentile(values, [1, 50, 99]))
            row[f'raw_{modality}_p01'], row[f'raw_{modality}_p50'], row[f'raw_{modality}_p99'] = p01, p50, p99
            row[f'raw_{modality}_max'] = float(values.max())
        row[f'norm_mean_{modality}'] = float(case.norm_mean[c])
        row[f'norm_std_{modality}'] = float(case.norm_std[c])
    row['affine'] = json.dumps(np.round(affine, 4).tolist())
    # One row per axial slice inside the crop; 'z' indexes the crop, 'z_orig' the original volume.
    slices = pd.DataFrame({
        'case_id': case_id,
        'z': np.arange(z1 - z0, dtype=np.int32),
        'z_orig': np.arange(z0, z1, dtype=np.int32),
        'brain_px': brain[:, :, z0:z1].sum(axis=(0, 1)).astype(np.int32),
        'wt_px': masks['WT'][:, :, z0:z1].sum(axis=(0, 1)).astype(np.int32),
        'tc_px': masks['TC'][:, :, z0:z1].sum(axis=(0, 1)).astype(np.int32),
        'et_px': masks['ET'][:, :, z0:z1].sum(axis=(0, 1)).astype(np.int32),
    })
    raw_hist, norm_hist = _intensity_histograms(case.image, case.norm_mean, case.norm_std)
    return row, slices, raw_hist, norm_hist


def make_preview(image: np.ndarray, labels: np.ndarray, case: PreparedCase) -> dict[str, np.ndarray]:
    """A few 2D slices through the tumor centre, kept for the example figures of notebook 01."""
    tumor = labels > 0
    if tumor.any():
        k = int(np.argmax(tumor.sum(axis=(0, 1))))                      # axial slice with the most tumor
        ci, cj, _ = (int(round(v)) for v in np.argwhere(tumor).mean(0))
    else:
        brain = np.any(image != 0, axis=0)
        k = int(np.argmax(brain.sum(axis=(0, 1))))
        ci, cj = (s // 2 for s in brain.shape[:2])
    norm_axial = np.zeros((len(MODALITIES),) + image.shape[1:3], dtype=np.float32)
    for c in range(len(MODALITIES)):
        channel = image[c, :, :, k]
        z = np.clip((channel - case.norm_mean[c]) / case.norm_std[c], -NORM_CLIP, NORM_CLIP)
        z[channel == 0] = 0
        norm_axial[c] = z
    return {
        'raw_axial': image[:, :, :, k].astype(np.float16),
        'norm_axial': norm_axial.astype(np.float16),
        'seg_axial': labels[:, :, k].astype(np.uint8),
        'flair_coronal': image[0, :, cj, :].astype(np.float16),
        'seg_coronal': labels[:, cj, :].astype(np.uint8),
        'flair_sagittal': image[0, ci, :, :].astype(np.float16),
        'seg_sagittal': labels[ci, :, :].astype(np.uint8),
        'index': np.array([ci, cj, k], dtype=np.int32),
    }


def _frequency_shape() -> tuple[int, int, int]:
    return tuple(math.ceil(s / FREQUENCY_MAP_DOWNSAMPLE) for s in EXPECTED_SHAPE)


def preprocess_case(task: tuple) -> dict:
    """Worker process: NIfTI files -> .npz + statistics. Never raises; errors are returned in the result."""
    case_id, files, out_path, want_preview = task
    try:
        image, labels, affine, spacing = load_nifti_case(files, require_seg=True)
        case = prepare_case(image, labels, affine, spacing)
        case.save(out_path)
        row, slices, raw_hist, norm_hist = case_statistics(case_id, image, labels, affine, spacing, case)
        row['npz_mb'] = os.path.getsize(out_path) / 1e6
        freq_tumor = freq_brain = None
        if tuple(image.shape[1:]) == EXPECTED_SHAPE:
            step = FREQUENCY_MAP_DOWNSAMPLE
            freq_tumor = np.packbits((labels > 0)[::step, ::step, ::step])
            freq_brain = np.packbits(np.any(image != 0, axis=0)[::step, ::step, ::step])
        preview = make_preview(image, labels, case) if want_preview else None
        return dict(case_id=case_id, ok=True, row=row, slices=slices, raw_hist=raw_hist, norm_hist=norm_hist,
                    freq_wt=freq_tumor, freq_brain=freq_brain, preview=preview)
    except Exception as error:      # one broken patient must not stop the whole shard
        return dict(case_id=case_id, ok=False, error=f'{type(error).__name__}: {error}',
                    trace=traceback.format_exc())


# -----------------------------------------------------------------------------
# Sharded, resumable preprocessing
# -----------------------------------------------------------------------------
def make_case_list(case_ids, shard_size: int = 100) -> pd.DataFrame:
    """Sorted case ids with their shard number."""
    ids = sorted(case_ids)
    return pd.DataFrame({'case_id': ids, 'shard': [i // shard_size for i in range(len(ids))]})


def shard_name(shard: int) -> str:
    return f'shard_{int(shard):03d}'


def is_shard_done(preprocessed_dir: Path, shard: int) -> bool:
    """A shard is complete when its marker exists and the tar on Drive has the recorded size."""
    marker = Path(preprocessed_dir) / f'{shard_name(shard)}.done.json'
    if not marker.is_file():
        return False
    try:
        info = load_json(marker)
        tar_path = Path(preprocessed_dir) / info['tar']
        return tar_path.is_file() and tar_path.stat().st_size == info['tar_bytes']
    except (OSError, ValueError, KeyError):
        return False


def all_shards_done(paths: ProjectPaths, case_list: pd.DataFrame | None) -> bool:
    return case_list is not None and all(is_shard_done(paths.preprocessed, s) for s in case_list['shard'].unique())


def _shard_statistics(ok_results: list[dict]) -> dict[str, np.ndarray]:
    frequency_shape = _frequency_shape()
    n_voxels = int(np.prod(frequency_shape))
    freq_tumor = np.zeros(n_voxels, dtype=np.int32)
    freq_brain = np.zeros(n_voxels, dtype=np.int32)
    n_freq = 0
    for result in ok_results:
        if result['freq_wt'] is not None:
            freq_tumor += np.unpackbits(result['freq_wt'])[:n_voxels]
            freq_brain += np.unpackbits(result['freq_brain'])[:n_voxels]
            n_freq += 1
    empty_hist = np.zeros((0, len(MODALITIES), 100), np.int64)
    stats = {
        'case_ids': np.array([r['case_id'] for r in ok_results]),
        'raw_hist': np.stack([r['raw_hist'] for r in ok_results]) if ok_results else empty_hist,
        'norm_hist': np.stack([r['norm_hist'] for r in ok_results]) if ok_results else empty_hist,
        'freq_wt': freq_tumor.reshape(frequency_shape), 'freq_brain': freq_brain.reshape(frequency_shape),
        'n_freq': np.array(n_freq),
    }
    for result in ok_results:
        if result['preview'] is not None:
            for key, value in result['preview'].items():
                stats[f'preview__{result["case_id"]}__{key}'] = value
    return stats


def preprocess_dataset(case_files: CaseFiles, case_list: pd.DataFrame, paths: ProjectPaths,
                       n_workers: int | None = None, preview_every: int = 20) -> None:
    """Preprocess every shard that is not done yet. Safe to re-run after a disconnect."""
    preprocessed_dir, tmp_root = Path(paths.preprocessed), Path(paths.work_shards)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    n_workers = n_workers or max(1, os.cpu_count() or 1)
    context = mp.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
    position = {case_id: i for i, case_id in enumerate(case_list['case_id'])}
    for shard in sorted(case_list['shard'].unique()):
        name = shard_name(shard)
        if is_shard_done(preprocessed_dir, shard):
            print(f'{name}: already done - skipping')
            continue
        shard_ids = case_list.loc[case_list['shard'] == shard, 'case_id'].tolist()
        local_dir = tmp_root / name
        if local_dir.exists():
            shutil.rmtree(local_dir)
        local_dir.mkdir(parents=True)
        tasks = [(case_id, case_files[case_id], str(local_dir / f'{case_id}.npz'),
                  position[case_id] % preview_every == 0)
                 for case_id in shard_ids if case_id in case_files]
        missing = [case_id for case_id in shard_ids if case_id not in case_files]
        print(f'\n{name}: {len(tasks)} cases with {n_workers} worker(s)')
        start = time.time()
        if n_workers == 1:
            results = [preprocess_case(task) for task in tqdm(tasks, desc=name)]
        else:
            with context.Pool(n_workers, maxtasksperchild=25) as pool:
                results = list(tqdm(pool.imap_unordered(preprocess_case, tasks), total=len(tasks), desc=name))
        results.sort(key=lambda r: r['case_id'])
        ok_results = [r for r in results if r['ok']]
        rows = [r['row'] for r in ok_results]
        for result in results:
            if not result['ok']:
                print(f'  ERROR {result["case_id"]}: {result["error"]}')
                rows.append(OrderedDict(case_id=result['case_id'], status='error', error=result['error']))
        for case_id in missing:
            rows.append(OrderedDict(case_id=case_id, status='error', error='files not found in archive'))
        for row in rows:
            row['shard'] = int(shard)
        cases_df = pd.DataFrame(rows)
        slices_df = pd.concat([r['slices'] for r in ok_results], ignore_index=True) if ok_results else pd.DataFrame()

        # Order matters for resuming: 1) tar of the case files, 2) statistics, 3) copy to Drive, 4) marker LAST.
        tar_local = tmp_root / f'{name}.tar'
        members = []
        with tarfile.open(tar_local, 'w') as tar:
            for result in ok_results:
                filename = f'{result["case_id"]}.npz'
                tar.add(local_dir / filename, arcname=filename)
                members.append(filename)
        save_csv(cases_df, local_dir / f'{name}_cases.csv')
        save_csv(slices_df, local_dir / f'{name}_slices.csv')
        save_npz(local_dir / f'{name}_stats.npz', **_shard_statistics(ok_results))
        print(f'  copying {name} to Drive ({tar_local.stat().st_size / 1e9:.2f} GB)...')
        copy_verified(tar_local, preprocessed_dir / f'{name}.tar')
        for extra in (f'{name}_cases.csv', f'{name}_slices.csv', f'{name}_stats.npz'):
            copy_verified(local_dir / extra, preprocessed_dir / extra)
        n_errors = len(rows) - len(ok_results)
        save_json({'tar': f'{name}.tar', 'tar_bytes': tar_local.stat().st_size, 'members': members,
                   'n_ok': len(ok_results), 'n_error': n_errors, 'seconds': round(time.time() - start, 1),
                   'finished': datetime.datetime.now().isoformat(timespec='seconds')},
                  preprocessed_dir / f'{name}.done.json')
        shutil.rmtree(local_dir, ignore_errors=True)
        tar_local.unlink(missing_ok=True)
        print(f'  {name} done: {len(ok_results)} ok, {n_errors} errors, {time.time() - start:.0f} s')


def merge_shard_statistics(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine the per-shard statistics into meta/ (cases.csv, slices.csv, dataset_stats.npz, previews.npz)."""
    preprocessed_dir, meta_dir = Path(paths.preprocessed), Path(paths.meta)
    meta_dir.mkdir(parents=True, exist_ok=True)
    markers = sorted(preprocessed_dir.glob('shard_*.done.json'))
    if not markers:
        raise FileNotFoundError(f'No finished shards in {preprocessed_dir}. Run the preprocessing step first.')
    cases, slices, raw_hists, norm_hists, ids, previews = [], [], [], [], [], {}
    frequency_shape = _frequency_shape()
    freq_tumor = np.zeros(frequency_shape, np.int64)
    freq_brain = np.zeros(frequency_shape, np.int64)
    n_freq = 0
    for marker in markers:
        name = marker.name.replace('.done.json', '')
        cases.append(pd.read_csv(preprocessed_dir / f'{name}_cases.csv'))
        slices_path = preprocessed_dir / f'{name}_slices.csv'
        if slices_path.stat().st_size > 5:            # an empty table is just a newline
            slices.append(pd.read_csv(slices_path))
        with np.load(preprocessed_dir / f'{name}_stats.npz', allow_pickle=False) as stats:
            ids.append(stats['case_ids'])
            raw_hists.append(stats['raw_hist'])
            norm_hists.append(stats['norm_hist'])
            freq_tumor += stats['freq_wt']
            freq_brain += stats['freq_brain']
            n_freq += int(stats['n_freq'])
            for key in stats.files:
                if key.startswith('preview__'):
                    previews[key] = stats[key]
    cases_df = pd.concat(cases, ignore_index=True).sort_values('case_id').reset_index(drop=True)
    slices_df = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()
    save_csv(cases_df, meta_dir / 'cases.csv')
    save_csv(slices_df, meta_dir / 'slices.csv')
    save_npz(meta_dir / 'dataset_stats.npz', case_ids=np.concatenate(ids), raw_hist=np.concatenate(raw_hists),
             norm_hist=np.concatenate(norm_hists), freq_wt=freq_tumor, freq_brain=freq_brain,
             n_freq=np.array(n_freq), raw_bins=RAW_HIST_BINS, norm_bins=NORM_HIST_BINS)
    save_npz(meta_dir / 'previews.npz', **previews)
    n_ok = int((cases_df['status'] == 'ok').sum())
    print(f'Merged {len(markers)} shards: {len(cases_df)} cases ({n_ok} ok), {len(slices_df)} slices.')
    return cases_df, slices_df


def load_dataset_stats(paths: ProjectPaths) -> dict[str, np.ndarray]:
    with np.load(Path(paths.meta) / 'dataset_stats.npz', allow_pickle=False) as stats:
        return {key: stats[key] for key in stats.files}


def load_previews(paths: ProjectPaths) -> dict[str, dict[str, np.ndarray]]:
    """{case_id: {'raw_axial': ..., 'seg_axial': ..., ...}} for the patients that have preview slices."""
    previews = {}
    with np.load(Path(paths.meta) / 'previews.npz', allow_pickle=False) as stored:
        for key in stored.files:
            _, case_id, name = key.split('__', 2)
            previews.setdefault(case_id, {})[name] = stored[key]
    return previews


# -----------------------------------------------------------------------------
# Patient-level split
# -----------------------------------------------------------------------------
def make_patient_split(cases_df: pd.DataFrame, fractions=(0.70, 0.15, 0.15), seed: int = 42) -> pd.DataFrame:
    """Split PATIENTS (never slices) into train / val / test, balanced by tumor size and ET presence."""
    from sklearn.model_selection import train_test_split
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError('fractions must add up to 1')
    ok = cases_df[cases_df['status'] == 'ok'].copy()
    ids = ok['case_id'].values
    try:
        volume_quartile = pd.qcut(ok['vol_wt_cm3'], 4, labels=False, duplicates='drop').astype(int)
        strata = (volume_quartile * 2 + ok['et_present'].astype(int)).values
    except (ValueError, TypeError):
        strata = None
    val_test_fraction = fractions[1] + fractions[2]

    def split(indices, stratify, size, seed_):
        try:
            return train_test_split(indices, test_size=size, random_state=seed_, stratify=stratify)
        except ValueError:          # a stratum is too small to be split - fall back to a plain random split
            return train_test_split(indices, test_size=size, random_state=seed_, stratify=None)

    indices = np.arange(len(ids))
    train_idx, rest_idx = split(indices, strata, val_test_fraction, seed)
    rest_strata = strata[rest_idx] if strata is not None else None
    val_idx, test_idx = split(rest_idx, rest_strata, fractions[2] / val_test_fraction, seed + 1)
    split_names = pd.Series('train', index=indices)
    split_names[val_idx] = 'val'
    split_names[test_idx] = 'test'
    return (pd.DataFrame({'case_id': ids, 'split': split_names.sort_index().values})
            .sort_values('case_id').reset_index(drop=True))


def load_metadata(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """cases (with their split), slices and split tables written by notebook 01."""
    meta_dir = Path(paths.meta)
    for name in ('cases.csv', 'slices.csv', 'split.csv'):
        if not (meta_dir / name).is_file():
            raise FileNotFoundError(f'{meta_dir / name} is missing - run notebook 01 '
                                    '(with the same QUICK_RUN setting) first.')
    cases = pd.read_csv(meta_dir / 'cases.csv')
    slices = pd.read_csv(meta_dir / 'slices.csv')
    split = pd.read_csv(meta_dir / 'split.csv')
    return cases.merge(split, on='case_id', how='left'), slices, split


def split_case_ids(split: pd.DataFrame) -> dict[str, list[str]]:
    """{'train': [...], 'val': [...], 'test': [...]} sorted case ids."""
    return {name: sorted(split.loc[split['split'] == name, 'case_id']) for name in ('train', 'val', 'test')}


# -----------------------------------------------------------------------------
# Staging to the local Colab disk
# -----------------------------------------------------------------------------
def stage_to_local_disk(paths: ProjectPaths, only_cases: set[str] | None = None) -> Path:
    """Copy the preprocessed shards from Drive to the fast local disk (skips what is already there).

    Returns the local folder that holds one <case_id>.npz per patient.
    """
    preprocessed_dir, local_dir = Path(paths.preprocessed), Path(paths.work_cases)
    markers = sorted(preprocessed_dir.glob('shard_*.done.json'))
    if not markers:
        raise FileNotFoundError(f'No preprocessed data in {preprocessed_dir}. Run notebook 01 first '
                                '(with the same QUICK_RUN setting as this notebook).')
    local_dir.mkdir(parents=True, exist_ok=True)
    staged_markers = local_dir / '.staged'
    staged_markers.mkdir(exist_ok=True)
    todo = []
    for marker in markers:
        info = load_json(marker)
        already_here = (staged_markers / marker.name).is_file() and all(
            (local_dir / member).is_file() for member in info['members'])
        if already_here:
            continue
        if only_cases is not None and not any(member[:-4] in only_cases for member in info['members']):
            continue
        todo.append((marker, info))
    if not todo:
        print(f'All shards already on local disk ({len(list(local_dir.glob("*.npz")))} case files in {local_dir}).')
        return local_dir
    need_gb = sum(info['tar_bytes'] for _, info in todo) / 1e9
    largest_gb = max(info['tar_bytes'] for _, info in todo) / 1e9
    require_disk(local_dir, need_gb + largest_gb + 2.0, 'staging the preprocessed data')
    start = time.time()
    for marker, info in tqdm(todo, desc='staging shards'):
        tar_local = local_dir / ('.' + info['tar'])
        copy_verified(preprocessed_dir / info['tar'], tar_local)
        with tarfile.open(tar_local, 'r') as tar:
            for member in tar.getmembers():
                if not member.isfile() or os.path.basename(member.name) != member.name:
                    continue        # only plain files at the top level (never paths that escape the folder)
                partial = local_dir / (member.name + '.partial')
                with tar.extractfile(member) as src, open(partial, 'wb') as dst:
                    shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                atomic_replace(partial, local_dir / member.name)
        tar_local.unlink(missing_ok=True)
        missing = [member for member in info['members'] if not (local_dir / member).is_file()]
        if missing:
            raise OSError(f'{info["tar"]}: {len(missing)} files missing after extraction')
        save_json(info, staged_markers / marker.name)
    print(f'Staged {len(todo)} shard(s), {need_gb:.2f} GB, in {time.time() - start:.0f} s -> {local_dir}')
    return local_dir


def verify_staged(cases_dir: Path, case_ids) -> None:
    missing = [case_id for case_id in case_ids if not (Path(cases_dir) / f'{case_id}.npz').is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} case files are missing on the local disk, e.g. {missing[:5]}. '
                                'Re-run the staging cell.')
    print(f'OK: all {len(case_ids)} case files are on the local disk.')


def load_case(cases_dir: Path, case_id: str) -> PreparedCase:
    return PreparedCase.load(Path(cases_dir) / f'{case_id}.npz')
