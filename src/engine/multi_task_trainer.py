"""
Multi-Task Trainer for Foundation Model Training
================================================
Extends the original 3-task trainer to support all 5 tasks:
  1. Classification  (cross-entropy + label smoothing)
  2. Regression      (MSE / Huber)
  3. Reconstruction  (ConservationLoss — MPA pre-training)
  4. Generative      (CVAE ELBO = recon MSE + β·KL)
  5. SuperResolution (MSE on upsampled 4-momenta)

Two distinct training modes:
  MultiTaskTrainer : simultaneous classification + regression + reconstruction
                     (the original multi-task loop, now with better logging)
  FineTuneTrainer  : single-task fine-tuning from a pre-trained encoder
                     (frozen / partial / full encoder modes)

Author: Ranjeet Gupta
"""

import os
import csv
import json
from typing import Dict, Optional, Callable, Union, Tuple, List, Literal
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

try:
    from torch.distributed import is_available, is_initialized, get_rank, get_world_size
    from torch.nn.parallel import DistributedDataParallel as DDP
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Original multi-task trainer (classification + regression + reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskTrainer:
    """
    Simultaneous training on classification + regression + reconstruction.
    Unchanged interface from original — just extended logging and task-weight
    scheduling hook.

    Parameters
    ----------
    model             : FoundationLorentzParT
    train_loader      : DataLoader (returns dict batches from JetClassDataset)
    val_loader        : DataLoader
    test_loader       : DataLoader, optional
    criterion         : HybridLoss
    optimizer         : Optimizer (AdamW recommended)
    scheduler         : LR scheduler, optional
    device            : cuda / cpu
    num_epochs        : training epochs
    task_modes        : which tasks are active
    mixed_precision   : use AMP FP16
    gradient_clip_val : max gradient norm
    save_dir          : output root directory
    experiment_name   : subdirectory name
    log_interval      : steps between batch-level logs
    save_checkpoint_interval : epochs between checkpoints
    eval_metrics      : {task: metric_fn} callables for additional metrics
    early_stopping_patience : None = disabled
    verbose           : print epoch summaries
    """

    def __init__(
        self,
        model:             nn.Module,
        train_loader:      DataLoader,
        val_loader:        DataLoader,
        test_loader:       Optional[DataLoader],
        criterion:         nn.Module,
        optimizer:         Optimizer,
        scheduler:         Optional[_LRScheduler]  = None,
        device:            Union[str, torch.device] = "cuda",
        num_epochs:        int                     = 100,
        task_modes:        Dict[str, bool]         = {
            "classification": True, "regression": True, "reconstruction": True,
        },
        mixed_precision:   bool                    = True,
        gradient_clip_val: Optional[float]         = 1.0,
        save_dir:          str                     = "./outputs",
        experiment_name:   str                     = "foundation_model",
        log_interval:      int                     = 50,
        save_checkpoint_interval: int              = 10,
        eval_metrics:      Optional[Dict[str, Callable]] = None,
        early_stopping_patience: Optional[int]     = None,
        verbose:           bool                    = True,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.device       = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_epochs   = num_epochs
        self.task_modes   = task_modes
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.gradient_clip_val     = gradient_clip_val
        self.log_interval          = log_interval
        self.save_checkpoint_interval = save_checkpoint_interval
        self.eval_metrics          = eval_metrics or {}
        self.early_stopping_patience = early_stopping_patience
        self.verbose = verbose

        self.is_distributed = DISTRIBUTED_AVAILABLE and is_initialized()
        self.rank       = get_rank()       if self.is_distributed else 0
        self.world_size = get_world_size() if self.is_distributed else 1
        self.is_main    = (self.rank == 0)

        self.save_dir = Path(save_dir) / experiment_name
        if self.is_main:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = self.save_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)

        self.model = self.model.to(self.device)
        if self.is_distributed:
            self.model = DDP(self.model, device_ids=[self.rank])

        self.scaler = GradScaler() if self.mixed_precision else None

        self.current_epoch           = 0
        self.global_step             = 0
        self.best_val_loss           = float('inf')
        self.epochs_without_improvement = 0

        self.history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
        for task in self.task_modes:
            self.history[f'train_{task}_loss'] = []
            self.history[f'val_{task}_loss']   = []

        if self.is_main and self.verbose:
            print("=" * 80)
            print("  Multi-Task Foundation Model Trainer")
            print("=" * 80)
            print(f"  Device:          {self.device}")
            print(f"  Distributed:     {self.is_distributed} (world={self.world_size})")
            print(f"  Mixed Precision: {self.mixed_precision}")
            print(f"  Active Tasks:    {[k for k,v in self.task_modes.items() if v]}")
            print(f"  Save Dir:        {self.save_dir}")
            print("=" * 80)

    # ── Epoch logic ────────────────────────────────────────────────────────

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {k: 0.0 for k in
                        ['total','classification','regression','reconstruction',
                         'pT','eta','phi','energy']}
        num_batches = 0

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Train]",
                    leave=False, disable=not self.is_main)

        for batch_idx, batch in enumerate(pbar):
            x, padding_mask, U, targets = self._prepare_batch(batch)

            with autocast(enabled=self.mixed_precision):
                task_mode    = self._get_task_mode()
                mask_indices = targets.get('mask_indices', None)
                predictions  = self.model(x, padding_mask, U,
                                          task=task_mode,
                                          mask_indices=mask_indices)
                loss, loss_components = self.criterion(
                    predictions, targets, return_components=True)

            self.optimizer.zero_grad()
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_val:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()

            epoch_losses['total'] += loss.item()
            for task, tl in loss_components.items():
                if task in epoch_losses:
                    epoch_losses[task] += tl.item() if hasattr(tl, 'item') else tl

            num_batches      += 1
            self.global_step += 1

            if self.is_main and (batch_idx + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

        return {k: v / num_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = {k: 0.0 for k in
                        ['total','classification','regression','reconstruction',
                         'pT','eta','phi','energy']}
        num_batches = 0
        all_preds   = {t: [] for t in self.task_modes}
        all_targets = {t: [] for t in self.task_modes}

        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {self.current_epoch+1}/{self.num_epochs} [Val]",
                    leave=False, disable=not self.is_main)

        for batch in pbar:
            x, padding_mask, U, targets = self._prepare_batch(batch)
            mask_indices = targets.get('mask_indices', None)
            predictions  = self.model(x, padding_mask, U,
                                       task=self._get_task_mode(),
                                       mask_indices=mask_indices)
            loss, loss_components = self.criterion(
                predictions, targets, return_components=True)

            epoch_losses['total'] += loss.item()
            for task, tl in loss_components.items():
                if task in epoch_losses:
                    epoch_losses[task] += tl.item() if hasattr(tl, 'item') else tl

            for task in self.task_modes:
                if task in predictions:
                    all_preds[task].append(predictions[task].cpu())
                    if task in targets:
                        all_targets[task].append(targets[task].cpu())
            num_batches += 1

        epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        epoch_losses.update(self._compute_metrics(all_preds, all_targets))
        return epoch_losses

    # ── Main loop ─────────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        if self.is_main and self.verbose:
            print("\n" + "=" * 80 + "\n  Starting Training\n" + "=" * 80 + "\n")

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()
            val_losses   = self.validate()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['learning_rate'].append(lr)
            for task in self.task_modes:
                self.history[f'train_{task}_loss'].append(train_losses.get(task, 0.0))
                self.history[f'val_{task}_loss'].append(val_losses.get(task, 0.0))

            if self.is_main and self.verbose:
                self._print_summary(epoch, train_losses, val_losses, lr)

            if self.is_main and (epoch + 1) % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_losses['total'], is_best=False)

            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.epochs_without_improvement = 0
                if self.is_main:
                    self._save_checkpoint(epoch, val_losses['total'], is_best=True)
            else:
                self.epochs_without_improvement += 1

            if (self.early_stopping_patience is not None and
                    self.epochs_without_improvement >= self.early_stopping_patience):
                if self.is_main and self.verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if self.is_main:
            self._save_history()
        if self.is_main and self.verbose:
            print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")
        return self.history

    # ── Helpers ───────────────────────────────────────────────────────────

    def _prepare_batch(self, batch) -> Tuple:
        x            = batch['x'].to(self.device)
        padding_mask = batch['padding_mask'].to(self.device)
        U            = batch['U'].to(self.device)
        targets = {}
        for key in ('classification_target', 'regression_target',
                    'reconstruction_target', 'mask_indices'):
            if key in batch:
                # Strip '_target' suffix to match loss dict keys
                tkey = key.replace('_target', '')
                targets[tkey] = batch[key].to(self.device)
        return x, padding_mask, U, targets

    def _get_task_mode(self) -> str:
        active = [t for t, v in self.task_modes.items() if v]
        return "all" if len(active) == len(self.task_modes) else (
            active[0] if len(active) == 1 else "all")

    def _compute_metrics(self, preds, targets) -> Dict[str, float]:
        metrics = {}
        for task, fn in self.eval_metrics.items():
            if preds[task] and targets[task]:
                p = torch.cat(preds[task], 0)
                t = torch.cat(targets[task], 0)
                metrics[f'{task}_metric'] = fn(p, t).item()
        return metrics

    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        state = self.model.module.state_dict() if self.is_distributed \
                else self.model.state_dict()
        ckpt  = {
            'epoch': epoch, 'global_step': self.global_step,
            'model_state_dict': state, 'val_loss': val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(ckpt, self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
        if is_best:
            torch.save(ckpt, self.checkpoint_dir / "best_model.pt")
            if self.verbose:
                print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

    def _save_history(self):
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        keys    = list(self.history.keys())
        min_len = min(len(self.history[k]) for k in keys)
        with open(self.save_dir / "training_history.csv", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch'] + keys)
            for i in range(min_len):
                w.writerow([i + 1] + [self.history[k][i] for k in keys])

    def _print_summary(self, epoch, train, val, lr):
        print(f"\nEpoch {epoch+1}/{self.num_epochs}  |  lr={lr:.2e}")
        print(f"  Total:           train={train['total']:.4f}  val={val['total']:.4f}")
        for task in ['classification', 'regression', 'reconstruction']:
            if task in train and train[task] > 0:
                print(f"  {task.capitalize():<16} train={train[task]:.4f}  "
                      f"val={val.get(task, 0):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning trainer — single-task with encoder freeze control
# ─────────────────────────────────────────────────────────────────────────────

class FineTuneTrainer:
    """
    Fine-tunes FoundationLorentzParT on a single downstream task.

    Fine-tuning modes
    -----------------
    'frozen'  : encoder frozen, only task head trained
    'partial' : last K encoder layers + head trained (default K=4)
    'full'    : all weights fine-tuned

    Dataset format expected (same JetClassDataset dict):
      classification  : uses 'classification_target'
      regression      : uses 'regression_target'
      generative      : uses 'x' + padding_mask as raw_particles input
      superresolution : uses 'x' low-res, targets from 'superres_target' key
    """

    SUPPORTED_TASKS = ('classification', 'regression', 'generative', 'superresolution')

    def __init__(
        self,
        model:             nn.Module,
        train_loader:      DataLoader,
        val_loader:        DataLoader,
        task:              str,
        optimizer:         Optimizer,
        scheduler:         Optional[_LRScheduler]         = None,
        device:            Union[str, torch.device]       = "cuda",
        num_epochs:        int                            = 30,
        finetune_mode:     Literal['frozen','partial','full'] = 'partial',
        unfreeze_last_k:   int                            = 4,
        mixed_precision:   bool                           = True,
        gradient_clip_val: float                          = 1.0,
        save_dir:          str                            = "./outputs/finetune",
        log_interval:      int                            = 50,
        save_every:        int                            = 5,
        verbose:           bool                           = True,
    ):
        assert task in self.SUPPORTED_TASKS, \
            f"task must be one of {self.SUPPORTED_TASKS}"

        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.task          = task
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.device        = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_epochs    = num_epochs
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        self.gradient_clip_val = gradient_clip_val
        self.log_interval  = log_interval
        self.save_every    = save_every
        self.verbose       = verbose

        self.save_dir = Path(save_dir) / task
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = self.model.to(self.device)
        self.scaler = GradScaler() if self.mixed_precision else None

        # Apply encoder freeze strategy
        self._apply_finetune_mode(finetune_mode, unfreeze_last_k)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Fine-Tuning  —  Task: {task.upper()}")
            print(f"  Mode: {finetune_mode}   |   Trainable: {trainable:,}")
            print(f"{'='*60}")

        self.history = {'train_loss': [], 'val_loss': [],
                        'train_metric': [], 'val_metric': []}
        self.best_val_loss = float('inf')

    def _apply_finetune_mode(self, mode: str, k: int):
        if mode == 'frozen':
            self.model.freeze_encoder(True)
        elif mode == 'partial':
            self.model.unfreeze_top_k_layers(k)
        else:  # 'full'
            self.model.freeze_encoder(False)

    def load_pretrained_encoder(self, path: str):
        """Load encoder weights from MPA pre-training checkpoint."""
        if not os.path.exists(path):
            print(f"[WARNING] Encoder checkpoint not found: {path}. Training from scratch.")
            return
        state = torch.load(path, map_location=self.device)
        # Support both raw state dict and wrapped checkpoint
        if 'encoder_state_dict' in state:
            state = state['encoder_state_dict']
        elif 'model_state_dict' in state:
            # Extract encoder sub-keys
            state = {k[len('encoder.'):]: v
                     for k, v in state['model_state_dict'].items()
                     if k.startswith('encoder.')}
        self.model.encoder.load_state_dict(state, strict=False)
        print(f"  ✓ Loaded pre-trained encoder from {path}")

    def _step(self, batch, train: bool) -> Tuple[float, float]:
        """Single forward pass. Returns (loss, metric)."""
        x = batch['x'].to(self.device)
        pm = batch['padding_mask'].to(self.device)
        U  = batch['U'].to(self.device)

        if train:
            self.optimizer.zero_grad()

        with autocast(enabled=self.mixed_precision):
            loss, metric = self._task_forward(x, pm, U, batch, train)

        if train:
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_val:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()

        return loss.item(), metric

    def _task_forward(self, x, pm, U, batch, train):
        if self.task == 'classification':
            out    = self.model(x, pm, U, task='classification')
            labels = batch['classification_target'].to(self.device)
            loss   = F.cross_entropy(out['classification'], labels)
            metric = (out['classification'].argmax(-1) == labels).float().mean().item()

        elif self.task == 'regression':
            out     = self.model(x, pm, U, task='regression')
            targets = batch['regression_target'].to(self.device).float()
            loss    = F.huber_loss(out['regression'], targets)
            metric  = ((out['regression'] - targets).abs() /
                       targets.abs().clamp(min=1e-6)).mean().item()

        elif self.task == 'generative':
            # Use raw (normalized) particles from the batch itself
            raw = batch.get('raw_particles', batch['x'][:, :, :4]).to(self.device)
            out    = self.model(x, pm, U, task='generative', raw_particles=raw)
            loss   = out['loss']
            metric = out['recon_loss'].item()

        elif self.task == 'superresolution':
            out    = self.model(x, pm, U, task='superresolution')
            target = batch['superres_target'].to(self.device).float()
            loss   = F.mse_loss(out['high_res'], target)
            metric = (out['high_res'][:, :, 0] - target[:, :, 0]).abs().mean().item()

        return loss, metric

    def finetune(self) -> Dict[str, List[float]]:
        """Run fine-tuning loop. Returns history dict."""
        for epoch in range(1, self.num_epochs + 1):
            # Train
            self.model.train()
            tl, tm = [], []
            for i, batch in enumerate(tqdm(self.train_loader, leave=False,
                                            desc=f"Epoch {epoch} [train]",
                                            disable=not self.verbose)):
                l, m = self._step(batch, train=True)
                tl.append(l); tm.append(m)
                if self.verbose and (i + 1) % self.log_interval == 0:
                    print(f"  step {i+1:>5} | loss={l:.4f} metric={m:.4f}")

            # Val
            self.model.eval()
            vl, vm = [], []
            with torch.no_grad():
                for batch in tqdm(self.val_loader, leave=False,
                                   desc=f"Epoch {epoch} [val]",
                                   disable=not self.verbose):
                    l, m = self._step(batch, train=False)
                    vl.append(l); vm.append(m)

            mean_tl = sum(tl) / len(tl)
            mean_vl = sum(vl) / len(vl)
            self.history['train_loss'].append(mean_tl)
            self.history['val_loss'].append(mean_vl)
            self.history['train_metric'].append(sum(tm) / len(tm))
            self.history['val_metric'].append(sum(vm) / len(vm))

            if self.scheduler:
                self.scheduler.step(mean_vl) if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ) else self.scheduler.step()

            if self.verbose:
                print(f"Epoch {epoch}/{self.num_epochs} | "
                      f"train={mean_tl:.4f} val={mean_vl:.4f}")

            if mean_vl < self.best_val_loss:
                self.best_val_loss = mean_vl
                self._save(epoch, tag='best')

            if epoch % self.save_every == 0:
                self._save(epoch)

        print(f"\nFine-tuning done. Best val={self.best_val_loss:.4f}")
        return self.history

    def _save(self, epoch: int, tag: str = ""):
        label = f"_{tag}" if tag else f"_epoch{epoch:04d}"
        path  = self.save_dir / f"{self.task}{label}.pt"
        torch.save({
            'epoch': epoch, 'task': self.task,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
        if self.verbose and tag == 'best':
            print(f"  ✓ Best fine-tune checkpoint saved → {path}")


if __name__ == "__main__":
    print("Multi-Task + Fine-Tune trainers ready.")
    print("  MultiTaskTrainer  — simultaneous classification + regression + reconstruction")
    print("  FineTuneTrainer   — single-task fine-tuning with encoder freeze control")