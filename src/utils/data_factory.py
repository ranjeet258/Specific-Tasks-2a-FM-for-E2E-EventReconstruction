"""
Data Factory for Foundation Model Training

This module handles data loading, preprocessing, and batching for the JetClass dataset.
Specifically configured for 100,000 events with 80-10-10 train-val-test split as per
the GSoC task requirements.

Physics Data Format:
-------------------
Each jet contains up to 128 particles, where each particle is characterized by:
- pT:   Transverse momentum       (right-skewed, 0-922 GeV)
- deta: Relative pseudorapidity   (already small, ~[-3, 3])
- dphi: Relative azimuthal angle  (periodic, ~[-pi, pi])
- E:    Energy                    (right-skewed, 0-2570 GeV)

Normalization Strategy (Optimal):
----------------------------------
Different features need different treatment:

  pT   -> log1p(pT) then z-score   (log1p compresses the right-skewed tail)
  deta -> z-score only              (already roughly symmetric, near-zero mean)
  dphi -> left unchanged            (periodic: cosine loss in HybridLoss handles it)
  E    -> log1p(E)  then z-score   (same reason as pT)

  Jet mass (regression) -> z-score
  Reconstruction target -> same per-component normalization as input features

All stats are computed from the TRAINING set only, then passed to val/test
to prevent any data leakage.

Dataset Tasks:
-------------
1. Classification: Jet tagging (10 classes in JetClass)
2. Regression:     Jet mass prediction      (z-score normalized)
3. Reconstruction: Masked particle 4-momentum (log1p + z-score for pT/E)

Author: GSoC 2026 Candidate
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

def _safe_log1p(x: np.ndarray) -> np.ndarray:
    """log1p that also handles negative values gracefully (sign-preserving)."""
    return np.sign(x) * np.log1p(np.abs(x))


class NormStats:
    """
    Holds mean/std for z-score normalization.
    Computed once from the training set, reused for val/test (no leakage).
    """
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std  = float(std) + 1e-8   # safety guard against zero std

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean

    def __repr__(self):
        return f"NormStats(mean={self.mean:.4f}, std={self.std:.4f})"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JetClassDataset(Dataset):
    """
    Dataset for loading JetClass particle jets with optimal normalization.

    Parameters
    ----------
    data_path      : path to preprocessed .npz directory
    split          : 'train', 'val', or 'test'
    max_particles  : maximum particles per jet (zero-padded)
    mask_particle  : whether to mask one particle for reconstruction task
    mask_strategy  : 'random', 'high_pt', or 'biased'
    compute_mass   : whether to include jet mass as regression target
    transform      : optional additional transform callable
    particle_stats : list of 4 NormStats [pT, deta, dphi, E]
                     If None, computed from this split (use only for train).
                     Pass train's stats to val/test.
    mass_stats     : NormStats for jet mass normalization.
                     If None, computed from this split (use only for train).
    """

    def __init__(
        self,
        data_path:      Union[str, Path],
        split:          str = 'train',
        max_particles:  int = 128,
        mask_particle:  bool = False,
        mask_strategy:  str = 'biased',
        compute_mass:   bool = True,
        transform:      Optional[callable] = None,
        particle_stats: Optional[List[NormStats]] = None,
        mass_stats:     Optional[NormStats] = None,
    ):
        super().__init__()

        self.data_path     = Path(data_path)
        self.split         = split
        self.max_particles = max_particles
        self.mask_particle = mask_particle
        self.mask_strategy = mask_strategy
        self.compute_mass  = compute_mass
        self.transform     = transform

        self._load_data()

        # ── Particle feature normalization ────────────────────────────────
        if particle_stats is not None:
            self.particle_stats = particle_stats
        else:
            self.particle_stats = self._compute_particle_stats()
            if split == 'train':
                self._print_particle_stats()

        # ── Jet mass normalization ─────────────────────────────────────────
        if mass_stats is not None:
            self.mass_stats = mass_stats
        elif compute_mass and self.jet_masses is not None:
            self.mass_stats = NormStats(
                self.jet_masses.mean(),
                self.jet_masses.std()
            )
            if split == 'train':
                print(f"  [mass]  mean={self.mass_stats.mean:.2f} GeV  "
                      f"std={self.mass_stats.std:.2f} GeV")
        else:
            self.mass_stats = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        data_file = self.data_path / f"{self.split}.npz"
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                "Run preprocessing first to generate train/val/test splits."
            )
        data = np.load(data_file)
        self.particles     = data['particles'].astype(np.float32)   # (N, P, 4)
        self.labels        = data['labels']                          # (N,)
        self.num_particles = data['num_particles']                   # (N,)
        self.jet_masses    = (data['jet_masses'].astype(np.float32)
                              if 'jet_masses' in data else None)     # (N,)
        self.num_samples   = len(self.labels)
        print(f"Loaded {self.split} split: {self.num_samples} jets")

    # ------------------------------------------------------------------
    # Normalization stat computation (train only)
    # ------------------------------------------------------------------

    def _compute_particle_stats(self) -> List[NormStats]:
        """
        Compute per-feature normalization stats over all VALID (non-padded)
        particles in this split.

        pT and E: apply log1p first to compress the right-skewed distribution,
                  then compute mean/std of the log-transformed values.
        deta:     compute mean/std directly (already symmetric, small range).
        dphi:     identity normalization (mean=0, std=1) — no change applied.
                  The cosine-based dphi loss in HybridLoss handles periodicity.
        """
        valid_mask  = (np.arange(self.max_particles)[None, :]
                       < self.num_particles[:, None])       # (N, P) bool
        valid_parts = self.particles[valid_mask]             # (M, 4)

        log_pT = _safe_log1p(valid_parts[:, 0])
        deta   = valid_parts[:, 1]
        # dphi: no normalization
        log_E  = _safe_log1p(valid_parts[:, 3])

        stats = [
            NormStats(log_pT.mean(), log_pT.std()),  # pT  (log-space z-score)
            NormStats(deta.mean(),   deta.std()),     # deta (z-score)
            NormStats(0.0,           1.0),            # dphi (identity — no change)
            NormStats(log_E.mean(),  log_E.std()),    # E   (log-space z-score)
        ]
        return stats

    def _print_particle_stats(self):
        labels = ['pT (log)', 'deta', 'dphi (raw)', 'E (log)']
        print("\n  [train] Per-feature normalization stats:")
        for name, s in zip(labels, self.particle_stats):
            print(f"    {name:<14}  mean={s.mean:+.4f}  std={s.std:.4f}")

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_particle_array(self, particles: np.ndarray) -> np.ndarray:
        """
        Normalize a (max_particles, 4) or (4,) array in-place copy.
        Applies: log1p -> z-score for pT and E; z-score for deta; raw for dphi.
        """
        out = particles.copy()
        if out.ndim == 1:
            # Single particle (4,)
            out[0] = self.particle_stats[0].normalize(_safe_log1p(out[0]))
            out[1] = self.particle_stats[1].normalize(out[1])
            # out[2] dphi: unchanged
            out[3] = self.particle_stats[3].normalize(_safe_log1p(out[3]))
        else:
            # Full array (max_particles, 4)
            out[:, 0] = self.particle_stats[0].normalize(_safe_log1p(particles[:, 0]))
            out[:, 1] = self.particle_stats[1].normalize(particles[:, 1])
            # col 2 dphi: unchanged
            out[:, 3] = self.particle_stats[3].normalize(_safe_log1p(particles[:, 3]))
        return out

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Returns a dict with:
          x                    : (max_particles, 16)  normalized multivector
          padding_mask         : (max_particles,)      bool  True=valid particle
          U                    : (max_particles, max_particles, 4)
          classification_target: scalar long
          regression_target    : (1,)   z-score normalized jet mass
          reconstruction_target: (4,)   normalized [pT, deta, dphi, E]
          mask_indices         : scalar long
        """
        particles = self.particles[idx].copy()    # (max_particles, 4)  raw
        num_valid = int(self.num_particles[idx])
        label     = int(self.labels[idx])

        # Padding mask
        padding_mask = torch.zeros(self.max_particles, dtype=torch.bool)
        padding_mask[:num_valid] = True

        # ── Reconstruction masking (before normalization) ─────────────────
        mask_idx        = None
        masked_particle = None
        if self.mask_particle:
            mask_idx        = self._choose_mask_index(particles, num_valid)
            masked_particle = particles[mask_idx].copy()  # save raw target
            particles[mask_idx] = 0.0                      # zero out in input

        # ── Normalize input particles ─────────────────────────────────────
        particles_norm = self._normalize_particle_array(particles)

        # ── Build model inputs from normalized particles ──────────────────
        x_mv = self._to_multivector(particles_norm)
        U    = self._compute_pairwise_features(particles_norm, num_valid)

        sample: Dict[str, Tensor] = {
            'x':                     torch.from_numpy(x_mv).float(),
            'padding_mask':          padding_mask,
            'U':                     torch.from_numpy(U).float(),
            'classification_target': torch.tensor(label, dtype=torch.long),
        }

        # ── Regression target: z-score normalized jet mass ────────────────
        if self.compute_mass:
            if self.jet_masses is not None:
                mass = float(self.jet_masses[idx])
            else:
                mass = self._compute_jet_mass(particles, num_valid)  # raw
            if self.mass_stats is not None:
                mass = self.mass_stats.normalize(mass)
            sample['regression_target'] = torch.tensor([mass], dtype=torch.float32)

        # ── Reconstruction target: normalized masked particle ─────────────
        if self.mask_particle and masked_particle is not None:
            norm_target = self._normalize_particle_array(masked_particle)
            sample['reconstruction_target'] = torch.from_numpy(norm_target).float()
            sample['mask_indices']          = torch.tensor(mask_idx, dtype=torch.long)


        # ── Super-resolution target: full high-res particle array ─────────
        n_high     = self.max_particles
        high_raw   = np.zeros((n_high, 4), dtype=np.float32)
        n_copy     = min(num_valid, n_high)
        high_raw[:n_copy] = self.particles[idx][:n_copy]
        high_norm  = self._normalize_particle_array(high_raw)
        sample['superres_target'] = torch.from_numpy(high_norm).float()

        if self.transform:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _to_multivector(self, particles: np.ndarray) -> np.ndarray:
        """
        Convert normalized [pT, deta, dphi, E] -> 16D multivector.

        pT and E here are already log1p + z-score normalized, so they are
        on a unit scale. We use them to build Cartesian 3-momentum so that
        the EquiLinear layers receive well-scaled inputs.
        """
        N  = particles.shape[0]
        mv = np.zeros((N, 16), dtype=np.float32)

        pT  = particles[:, 0]   # normalized
        eta = particles[:, 1]   # normalized
        phi = particles[:, 2]   # raw dphi
        E   = particles[:, 3]   # normalized

        # Clip eta to avoid sinh overflow at large values
        eta_clipped = np.clip(eta, -5.0, 5.0)

        px = pT * np.cos(phi)
        py = pT * np.sin(phi)
        pz = pT * np.sinh(eta_clipped)

        mv[:, 0] = E    # grade-0: energy
        mv[:, 1] = px   # grade-1: spatial momentum
        mv[:, 2] = py
        mv[:, 3] = pz
        # Grades 4-15: left zero, filled by EquiLinear layers

        return mv

    def _compute_pairwise_features(
        self,
        particles: np.ndarray,
        num_valid: int
    ) -> np.ndarray:
        """
        Pairwise features [delta_eta, delta_phi, delta_R, log(pT_i * pT_j)].
        Built from NORMALIZED particles so all values stay in a sensible range.
        """
        N = self.max_particles
        U = np.zeros((N, N, 4), dtype=np.float32)

        if num_valid == 0:
            return U

        pT  = particles[:num_valid, 0]   # normalized log-pT
        eta = particles[:num_valid, 1]   # normalized deta
        phi = particles[:num_valid, 2]   # raw dphi

        delta_eta = eta[:, None] - eta[None, :]
        delta_phi = phi[:, None] - phi[None, :]
        delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))  # wrap to (-pi, pi)
        delta_R   = np.sqrt(delta_eta**2 + delta_phi**2)

        pT_product     = pT[:, None] * pT[None, :]
        log_pT_product = np.log(np.abs(pT_product) + 1e-8)

        U[:num_valid, :num_valid, 0] = delta_eta
        U[:num_valid, :num_valid, 1] = delta_phi
        U[:num_valid, :num_valid, 2] = delta_R
        U[:num_valid, :num_valid, 3] = log_pT_product

        return U

    def _compute_jet_mass(self, particles: np.ndarray, num_valid: int) -> float:
        """
        Compute invariant jet mass from RAW (un-normalized) particles.
        Used as a fallback when jet_masses not precomputed.

        Physics:  m^2 = (sum E)^2 - (sum px)^2 - (sum py)^2 - (sum pz)^2
        """
        if num_valid == 0:
            return 0.0
        pT  = particles[:num_valid, 0]
        eta = particles[:num_valid, 1]
        phi = particles[:num_valid, 2]
        E   = particles[:num_valid, 3]
        eta = np.clip(eta, -5.0, 5.0)
        px  = pT * np.cos(phi)
        py  = pT * np.sin(phi)
        pz  = pT * np.sinh(eta)
        m2  = E.sum()**2 - px.sum()**2 - py.sum()**2 - pz.sum()**2
        return float(np.sqrt(max(0.0, m2)))

    def _choose_mask_index(self, particles: np.ndarray, num_valid: int) -> int:
        """
        Choose which particle to mask for reconstruction task.
        Uses RAW pT values for 'biased' strategy.
        """
        if num_valid == 0:
            return 0
        pT = particles[:num_valid, 0]
        if self.mask_strategy == 'random':
            return int(np.random.randint(0, num_valid))
        elif self.mask_strategy == 'high_pt':
            return int(np.argmax(pT))
        elif self.mask_strategy == 'biased':
            probs = pT / (pT.sum() + 1e-8)
            return int(np.random.choice(num_valid, p=probs))
        else:
            raise ValueError(
                f"Unknown mask_strategy: {self.mask_strategy!r}. "
                "Choose from 'random', 'high_pt', 'biased'."
            )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_path:     Union[str, Path],
    batch_size:    int = 32,
    num_workers:   int = 4,
    max_particles: int = 128,
    mask_particle: bool = False,
    mask_strategy: str = 'biased',
    distributed:   bool = False,
    seed:          int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train / val / test DataLoaders with shared normalization stats.

    Normalization flow:
    -------------------
    1. Train dataset computes particle_stats and mass_stats from training data.
    2. Val and test datasets receive those exact same stats.
    3. No data leakage: val/test stats never influence normalization parameters.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Step 1: Train — computes normalization stats
    train_dataset = JetClassDataset(
        data_path=data_path,
        split='train',
        max_particles=max_particles,
        mask_particle=mask_particle,
        mask_strategy=mask_strategy,
    )

    # Step 2: Val / Test — reuse train stats (no leakage)
    val_dataset = JetClassDataset(
        data_path=data_path,
        split='val',
        max_particles=max_particles,
        mask_particle=mask_particle,
        mask_strategy=mask_strategy,
        particle_stats=train_dataset.particle_stats,
        mass_stats=train_dataset.mass_stats,
    )

    test_dataset = JetClassDataset(
        data_path=data_path,
        split='test',
        max_particles=max_particles,
        mask_particle=False,                         # no masking at test time
        mask_strategy=mask_strategy,
        particle_stats=train_dataset.particle_stats,
        mass_stats=train_dataset.mass_stats,
    )

    # Distributed samplers
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True,  seed=seed)
        val_sampler   = DistributedSampler(val_dataset,   shuffle=False)
        test_sampler  = DistributedSampler(test_dataset,  shuffle=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True

    pw = num_workers > 0   # persistent_workers requires at least 1 worker

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=pw,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=pw,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=pw,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("JetClass Data Factory — Optimal Normalization")
    print("=" * 70)
    print("  pT   : log1p -> z-score  (tames right-skewed GeV distribution)")
    print("  deta : z-score           (already small range)")
    print("  dphi : unchanged         (cosine loss handles periodicity)")
    print("  E    : log1p -> z-score  (same as pT)")
    print("  mass : z-score           (regression target)")
    print("  All stats: computed from TRAIN only, shared with val/test")
    print("=" * 70)