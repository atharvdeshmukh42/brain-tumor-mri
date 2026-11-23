"""
Training helpers shared by both models: mixed precision, the learning-rate schedule, one training epoch,
checkpoints, and `ResumableTraining`, which runs the epoch loop so that a Colab disconnect loses at most one epoch.

Files written in <run_dir>/checkpoints/:
    last.pt          full training state (model, optimizer, scheduler, scaler, RNGs, history), saved every epoch
    best.pt          weights of the best epoch so far + settings: what the offline app loads
    epoch_XXX.pt     weights of every epoch (optional)
"""
from __future__ import annotations

import datetime
import math
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .fileio import save_csv, save_torch, to_jsonable


# -----------------------------------------------------------------------------
# Mixed precision and learning-rate schedule
# -----------------------------------------------------------------------------
def autocast(device: torch.device, enabled: bool = True):
    """float16 autocast on CUDA; a no-op elsewhere (CPU / Apple GPU run in float32)."""
    use = bool(enabled) and device.type == 'cuda'
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use)


def make_grad_scaler(device: torch.device, enabled: bool = True):
    use = bool(enabled) and device.type == 'cuda'
    try:
        return torch.amp.GradScaler('cuda', enabled=use)
    except (AttributeError, TypeError):      # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=use)


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int = 0,
                   min_lr_ratio: float = 0.01) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up, then cosine decay to min_lr_ratio x the initial rate. Stepped once per iteration."""
    total_steps = max(1, int(total_steps))
    warmup_steps = int(min(warmup_steps, total_steps - 1))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device,
                collate_fn=None, drop_last: bool = False) -> torch.utils.data.DataLoader:
    """DataLoader with pinned memory on CUDA and workers that stay alive between epochs."""
    options = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                   pin_memory=(device.type == 'cuda'), collate_fn=collate_fn, drop_last=drop_last)
    if num_workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    return torch.utils.data.DataLoader(dataset, **options)


class WorkerRandom:
    """Gives every DataLoader worker process its own random stream that keeps changing across epochs."""

    def __init__(self):
        self._rng, self._pid = None, None

    def get(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._pid != pid:
            self._rng = np.random.default_rng([torch.initial_seed() % (2 ** 32), pid])
            self._pid = pid
        return self._rng


# -----------------------------------------------------------------------------
# One epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model: torch.nn.Module, loader, optimizer, scheduler, scaler, loss_fn: Callable,
                    device: torch.device, amp: bool = True, max_grad_norm: float = 5.0,
                    max_iters: int | None = None, track_accuracy: bool = False) -> dict:
    """Train for one pass over loader. Batches are (inputs, targets, ...); extra items are ignored.

    A classifier with one output (B, 1) is compared with targets of shape (B,).
    Batches with a non-finite loss are skipped and counted. The scheduler only advances when the optimizer
    really stepped (mixed precision skips the step when gradients overflow).
    """
    model.train()
    n_samples, loss_sum, correct, skipped, start = 0, 0.0, 0, 0, time.time()
    progress = tqdm(loader, desc='train', leave=False)
    for iteration, batch in enumerate(progress):
        if max_iters is not None and iteration >= max_iters:
            break
        inputs = batch[0].to(device, non_blocking=True).float()
        targets = batch[1].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
            logits = model(inputs)
        if logits.ndim == targets.ndim + 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        loss = loss_fn(logits.float(), targets)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:     # the scale only drops when the step was skipped
            scheduler.step()
        batch_size = targets.shape[0]
        n_samples += batch_size
        loss_sum += loss.item() * batch_size
        if track_accuracy:
            correct += ((logits.detach() > 0).float() == targets).sum().item()
        if iteration % 10 == 0:
            progress.set_postfix(loss=f'{loss_sum / max(n_samples, 1):.4f}')
    stats = {'train_loss': loss_sum / max(n_samples, 1), 'train_samples': n_samples,
             'train_skipped_batches': skipped, 'train_seconds': time.time() - start}
    if track_accuracy:
        stats['train_acc'] = correct / max(n_samples, 1)
    return stats


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------
def save_model_weights(path: Path, model: torch.nn.Module, model_name: str, config: dict, epoch: int,
                       metric_name: str, metric: float) -> None:
    """Small file with only the weights + settings: this is what the offline app loads."""
    save_torch({'model': model.state_dict(), 'model_name': model_name, 'config': to_jsonable(config),
                'epoch': int(epoch), 'metric_name': metric_name, 'metric': float(metric),
                'saved_at': datetime.datetime.now().isoformat(timespec='seconds')}, path)


def load_model_weights(path: Path, device='cpu') -> dict:
    """Load a best.pt / epoch_XXX.pt file (safe: weights_only, no arbitrary code)."""
    return torch.load(path, map_location=device, weights_only=True)


def save_training_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_metric: float,
                        best_epoch: int, validations_since_best: int, history: list[dict], config: dict,
                        finished: bool = False, early_stopped: bool = False) -> None:
    state = {
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'epoch': int(epoch), 'best_metric': float(best_metric), 'best_epoch': int(best_epoch),
        'validations_since_best': int(validations_since_best),
        'history': to_jsonable(history), 'config': to_jsonable(config),
        'finished': bool(finished), 'early_stopped': bool(early_stopped),
        'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
        'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    if torch.cuda.is_available():
        state['cuda_rng'] = torch.cuda.get_rng_state_all()
    save_torch(state, path)


def load_training_state(path: Path, model, optimizer, scheduler, scaler, device) -> dict:
    """Restore everything saved by save_training_state; returns the state dict."""
    state = torch.load(path, map_location=device, weights_only=False)   # our own file: contains RNG objects
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    try:
        scaler.load_state_dict(state['scaler'])
    except (KeyError, RuntimeError, ValueError):     # e.g. resumed on a CPU after training on a GPU
        pass
    try:
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if 'cuda_rng' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        print('(RNG state not restored:', error, ')')
    return state


def gpu_peak_memory_gb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return 0.0


def reset_gpu_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# -----------------------------------------------------------------------------
# The resumable epoch loop
# -----------------------------------------------------------------------------
class ResumableTraining:
    """Bookkeeping of a training run that survives Colab disconnects.

    Usage in a notebook:

        run = ResumableTraining(run_dir, model, optimizer, scheduler, scaler, device, model_name=..., config=...,
                                metric_name='val_auc', epochs=EPOCHS, patience=6)
        for epoch in run.remaining_epochs():
            row = {...train and validation numbers...}
            run.end_epoch(epoch, row, metric=validation_score)   # metric=None when not validated this epoch

    end_epoch saves best.pt when the metric improves, epoch_XXX.pt (optional), last.pt and history.csv.
    `patience` counts validations without improvement (None = never stop early).
    """

    def __init__(self, run_dir: Path, model, optimizer, scheduler, scaler, device: torch.device, *,
                 model_name: str, config: dict, metric_name: str, epochs: int, patience: int | None = None,
                 keep_every_epoch: bool = True):
        self.run_dir = Path(run_dir)
        self.model, self.optimizer, self.scheduler, self.scaler = model, optimizer, scheduler, scaler
        self.device = device
        self.model_name, self.config, self.metric_name = model_name, config, metric_name
        self.epochs, self.patience, self.keep_every_epoch = int(epochs), patience, keep_every_epoch
        self.checkpoints = self.run_dir / 'checkpoints'
        self.last_path = self.checkpoints / 'last.pt'
        self.best_path = self.checkpoints / 'best.pt'
        self.history: list[dict] = []
        self.start_epoch, self.best_epoch, self.best_metric = 1, 0, -float('inf')
        self.validations_since_best = 0
        self.finished = self.early_stopped = False
        self._resume()

    def _resume(self) -> None:
        if not self.last_path.is_file():
            print('Starting a new training run.')
            return
        state = load_training_state(self.last_path, self.model, self.optimizer, self.scheduler, self.scaler,
                                    self.device)
        self.start_epoch = state['epoch'] + 1
        self.best_metric, self.best_epoch = state['best_metric'], state['best_epoch']
        self.history = state['history']
        self.validations_since_best = state.get('validations_since_best', sum(
            1 for row in self.history if row.get(self.metric_name) is not None and row['epoch'] > self.best_epoch))
        self.early_stopped = state.get('early_stopped', False)
        self.finished = self.early_stopped or (state['finished'] and state['epoch'] >= self.epochs)
        print(f'RESUMING: epoch {state["epoch"]} was finished (saved {state.get("saved_at")}); best '
              f'{self.metric_name} {self.best_metric:.4f} at epoch {self.best_epoch}.'
              + (' Training is already complete.' if self.finished else ''))

    def remaining_epochs(self) -> Iterator[int]:
        epoch = self.start_epoch
        while not self.finished and epoch <= self.epochs:
            reset_gpu_peak_memory()
            yield epoch
            epoch += 1
        self.finished = True

    def end_epoch(self, epoch: int, row: dict, metric: float | None = None) -> bool:
        """Record one finished epoch and save everything. Returns True when this epoch is the new best."""
        improved = False
        if metric is not None:
            if metric > self.best_metric:
                self.best_metric, self.best_epoch, self.validations_since_best = metric, epoch, 0
                save_model_weights(self.best_path, self.model, self.model_name, self.config, epoch,
                                   self.metric_name, metric)
                improved = True
            else:
                self.validations_since_best += 1
        self.early_stopped = self.patience is not None and self.validations_since_best >= self.patience
        row = {'epoch': epoch, **row, 'gpu_mem_gb': gpu_peak_memory_gb()}
        self.history.append(row)
        if self.keep_every_epoch:
            save_model_weights(self.checkpoints / f'epoch_{epoch:03d}.pt', self.model, self.model_name, self.config,
                               epoch, self.metric_name, metric if metric is not None else float('nan'))
        self.finished = self.early_stopped or epoch >= self.epochs
        save_training_state(self.last_path, self.model, self.optimizer, self.scheduler, self.scaler, epoch,
                            self.best_metric, self.best_epoch, self.validations_since_best, self.history,
                            self.config, finished=self.finished, early_stopped=self.early_stopped)
        save_csv(pd.DataFrame(self.history), self.run_dir / 'history.csv')
        if self.early_stopped:
            print(f'Early stopping: no improvement in {self.patience} validations.')
        return improved

    def hours_left(self, epoch: int) -> float:
        """Rough remaining time from the last 5 epochs."""
        recent = [row['epoch_seconds'] for row in self.history[-5:] if 'epoch_seconds' in row]
        return (self.epochs - epoch) * (np.mean(recent) if recent else 0.0) / 3600

    @property
    def training_hours(self) -> float:
        return float(sum(row.get('epoch_seconds', 0.0) for row in self.history) / 3600)
