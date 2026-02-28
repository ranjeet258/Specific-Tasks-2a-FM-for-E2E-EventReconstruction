"""
pretrain.py — Masked Particle Autoencoder (MPA) Pre-Training
=============================================================
Self-supervised pre-training for FoundationLorentzParT.

Strategy: Masked Particle Autoencoder
  1. Randomly mask K% of particles per jet (no labels needed)
  2. Replace masked positions with learnable [MASK] token
  3. Encode the masked jet through the shared encoder
  4. Reconstruction head predicts 4-momenta of masked particles
  5. Loss = ConservationLoss on masked positions only
  6. Save encoder weights → fine-tune on downstream tasks

Three masking strategies:
  'random' : uniform random
  'biased' : high-pT particles more likely masked (model learns from hard cores)
  'block'  : angular patches masked (simulates calorimeter dead zones)
"""

import os
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.models.foundation_lorentz_part import FoundationLorentzParT
from src.utils.data_factory import create_dataloaders
from src.loss.hybrid_loss import ConservationLoss


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("MPA Pre-Training for FoundationLorentzParT")
    p.add_argument('--config',       default='configs/foundation_config.yaml')
    p.add_argument('--data-path',    default='./data/jetclass_100k')
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch-size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--mask-ratio',   type=float, default=0.30)
    p.add_argument('--mask-strategy',default='biased',
                   choices=['random', 'biased', 'block'])
    p.add_argument('--save-dir',     default='./outputs/pretrain')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--num-workers',  type=int,   default=4)
    p.add_argument('--warmup-epochs',type=int,   default=5)
    p.add_argument('--grad-clip',    type=float, default=1.0)
    p.add_argument('--mixed-precision', action='store_true', default=True)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Masking utilities
# ─────────────────────────────────────────────────────────────────────────────

class ParticleMasker:
    """
    Applies masking strategies to a batch and returns masked inputs.

    The learnable mask_token replaces masked particle features so the
    encoder knows which positions were masked. This is the standard
    BERT-style approach adapted to HEP particle sets.
    """
    def __init__(self, mask_ratio: float, strategy: str, mv_dim: int = 16):
        self.mask_ratio = mask_ratio
        self.strategy   = strategy
        self.mask_token = nn.Parameter(torch.zeros(mv_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def to(self, device):
        self.mask_token = nn.Parameter(self.mask_token.data.to(device))
        return self

    def parameters(self):
        """Expose mask_token for optimizer."""
        yield self.mask_token

    def __call__(self, x: Tensor, padding_mask: Tensor,
                 raw_particles: Optional[Tensor] = None):
        """
        Returns (masked_x, padding_mask, mask_positions).
        mask_positions: (B, N) bool — True where real particle was masked.
        """
        if self.strategy == 'random':
            return self._random(x, padding_mask)
        elif self.strategy == 'biased':
            return self._biased(x, padding_mask, raw_particles)
        elif self.strategy == 'block':
            return self._block(x, padding_mask, raw_particles)
        else:
            return self._random(x, padding_mask)

    def _random(self, x, pm):
        B, N, _ = x.shape
        noise = torch.rand(B, N, device=x.device).masked_fill(~pm, 1.0)
        threshold = torch.quantile(
            noise.masked_fill(~pm, float('inf')),
            self.mask_ratio, dim=-1, keepdim=True)
        mask_pos = (noise <= threshold) & pm
        return self._apply(x, mask_pos), pm, mask_pos

    def _biased(self, x, pm, raw):
        """Mask particles proportional to pT — models learn hard-core → soft structure."""
        B, N, _ = x.shape
        if raw is not None:
            pt = raw[:, :, 0].abs().clamp(min=1e-6)
            weight = pt / (pt.masked_fill(~pm, 0.0).sum(-1, keepdim=True) + 1e-6)
        else:
            weight = torch.ones(B, N, device=x.device) / N
        weight = weight.masked_fill(~pm, 0.0)
        # Sample k particles per jet according to weight
        n_to_mask = (pm.float().sum(-1) * self.mask_ratio).long().clamp(min=1)
        noise     = torch.rand_like(weight) * weight
        _, idx    = noise.sort(dim=-1, descending=True)
        mask_pos  = torch.zeros_like(pm)
        for b in range(B):
            k = n_to_mask[b].item()
            mask_pos[b].scatter_(0, idx[b, :k], True)
        mask_pos = mask_pos & pm
        return self._apply(x, mask_pos), pm, mask_pos

    def _block(self, x, pm, raw):
        """Mask angular patches to simulate calorimeter dead zones."""
        import math
        B, N, _ = x.shape
        if raw is None:
            return self._random(x, pm)
        eta, phi = raw[:, :, 1], raw[:, :, 2]
        mask_pos = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for b in range(B):
            n_valid = pm[b].sum().item()
            if n_valid == 0:
                continue
            ref = torch.randint(0, n_valid, (1,)).item()
            de  = (eta[b] - eta[b, ref]).abs()
            dp  = (phi[b] - phi[b, ref]).abs()
            dp  = torch.min(dp, 2 * math.pi - dp)
            block_size = 0.4
            mask_pos[b] = (de < block_size) & (dp < block_size) & pm[b]
        return self._apply(x, mask_pos), pm, mask_pos

    def _apply(self, x: Tensor, mask_pos: Tensor) -> Tensor:
        out = x.clone()
        out[mask_pos] = self.mask_token
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Pre-trainer
# ─────────────────────────────────────────────────────────────────────────────

class MaskedParticlePreTrainer:
    """
    Self-supervised pre-training loop.

    After training, saves encoder weights to:
      <save_dir>/encoder_best.pt   (lowest val loss)
      <save_dir>/encoder_final.pt  (last epoch)

    These can be loaded by FineTuneTrainer.load_pretrained_encoder().
    """

    def __init__(
        self,
        model:         FoundationLorentzParT,
        train_loader,
        val_loader,
        device:        torch.device,
        args,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.args         = args

        self.masker = ParticleMasker(args.mask_ratio, args.mask_strategy).to(device)
        self.recon_loss = ConservationLoss()

        # Optimizer includes learnable mask token
        all_params = list(model.parameters()) + list(self.masker.parameters())
        self.opt    = AdamW(all_params, lr=args.lr, weight_decay=1e-2)
        self.sched  = CosineAnnealingLR(
            self.opt,
            T_max=max(args.epochs - args.warmup_epochs, 1),
            eta_min=1e-6,
        )
        self.scaler = GradScaler() if args.mixed_precision and torch.cuda.is_available() else None

        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history     = {"train_loss": [], "val_loss": []}
        self.best_val    = float('inf')

    def _warmup(self, epoch: int):
        if epoch < self.args.warmup_epochs:
            scale = (epoch + 1) / max(self.args.warmup_epochs, 1)
            for pg in self.opt.param_groups:
                pg['lr'] = self.args.lr * scale

    def _forward(self, batch, train: bool) -> float:
        x   = batch['x'].to(self.device)
        pm  = batch['padding_mask'].to(self.device)
        U   = batch['U'].to(self.device)
        # Use reconstruction_target as ground truth; fall back to raw particles
        if 'reconstruction_target' in batch:
            raw = batch['reconstruction_target'].unsqueeze(1).expand_as(
                batch['x'][:, :, :4]).to(self.device)
        else:
            raw = None

        if train:
            self.opt.zero_grad()

        with autocast(enabled=self.scaler is not None):
            masked_x, pm, mask_pos = self.masker(x, pm, raw)

            # Encode masked jet
            embeddings, equi = self.model.encoder(masked_x, pm, U)

            # Gather all masked positions
            b_idx, n_idx = mask_pos.nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                return 0.0

            # Predict 4-momenta for masked positions
            pred = self.model.reconstruction_head.reconstructor(
                self.model.reconstruction_head.feature_combine(
                    torch.cat([embeddings[b_idx, n_idx],
                               equi[b_idx, n_idx]], dim=-1)
                )
            )   # (M, 4)

            # Ground truth: use raw particles at masked positions
            if raw is not None:
                target = raw[b_idx, n_idx]
            else:
                # Use x[:,:,:4] as proxy target (normalized features)
                target = x[b_idx, n_idx, :4]

            loss = self.recon_loss(pred, target)

        if train:
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.opt.step()

        return loss.item()

    def pretrain(self) -> Dict[str, List[float]]:
        print(f"\n{'='*60}")
        print(f"  Masked Particle Autoencoder Pre-Training")
        print(f"  Strategy: {self.args.mask_strategy}  |  Ratio: {self.args.mask_ratio}")
        print(f"  Epochs: {self.args.epochs}  |  LR: {self.args.lr}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.args.epochs + 1):
            self._warmup(epoch - 1)

            # Train
            self.model.train()
            tl, t0 = [], time.time()
            for i, batch in enumerate(tqdm(self.train_loader, desc=f"Pretrain {epoch}", leave=False)):
                l = self._forward(batch, train=True)
                tl.append(l)

            # Val
            self.model.eval()
            vl = []
            with torch.no_grad():
                for batch in self.val_loader:
                    vl.append(self._forward(batch, train=False))

            mean_tl = sum(tl) / max(len(tl), 1)
            mean_vl = sum(vl) / max(len(vl), 1)
            self.history["train_loss"].append(mean_tl)
            self.history["val_loss"].append(mean_vl)

            if epoch >= self.args.warmup_epochs:
                self.sched.step()

            print(f"Epoch {epoch:>3}/{self.args.epochs} | "
                  f"train={mean_tl:.4f}  val={mean_vl:.4f} | "
                  f"{time.time()-t0:.1f}s")

            if mean_vl < self.best_val:
                self.best_val = mean_vl
                self._save_encoder("encoder_best.pt")

            if epoch % 10 == 0:
                self._save_full(epoch)

        self._save_encoder("encoder_final.pt")
        with open(self.save_dir / "pretrain_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nPre-training complete. Best val loss: {self.best_val:.4f}")
        return self.history

    def _save_encoder(self, filename: str):
        path = self.save_dir / filename
        torch.save(self.model.encoder.state_dict(), path)
        print(f"  ✓ Encoder saved → {path}")

    def _save_full(self, epoch: int):
        path = self.save_dir / f"pretrain_epoch{epoch:04d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'history': self.history,
        }, path)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config if provided
    model_cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get('model', {})

    # Build model
    model = FoundationLorentzParT(
        embed_dim=model_cfg.get('embed_dim', 128),
        num_heads=model_cfg.get('num_heads', 8),
        num_layers=model_cfg.get('num_layers', 8),
        dropout=model_cfg.get('dropout', 0.1),
        expansion_factor=model_cfg.get('expansion_factor', 4),
        pair_embed_dims=model_cfg.get('pair_embed_dims', [64, 64, 64]),
        num_classes=model_cfg.get('num_classes', 10),
    )
    model.print_parameter_summary()

    # Data (masking enabled — labels not used during pre-training)
    train_loader, val_loader, _ = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_particle=True,
        mask_strategy=args.mask_strategy,
    )

    trainer = MaskedParticlePreTrainer(model, train_loader, val_loader, device, args)
    trainer.pretrain()


if __name__ == "__main__":
    main()