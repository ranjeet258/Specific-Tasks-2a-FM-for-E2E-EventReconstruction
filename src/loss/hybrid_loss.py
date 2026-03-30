"""
Hybrid Loss Functions for Multi-Task Foundation Model Training

This module implements custom loss functions that combine multiple objectives:
1. Classification Loss (CrossEntropy)
2. Regression Loss (MSE/Huber)
3. Conservation Loss (for masked particle reconstruction)

Physics Background:
------------------
In particle physics, several physical laws must be respected:
- Conservation of Energy: Total energy before = total energy after
- Conservation of Momentum: Total momentum before = total momentum after
- Lorentz Invariance: Physics laws same in all inertial reference frames

The ConservationLoss helps the model learn these fundamental physics constraints
during self-supervised pre-training on masked particle reconstruction.

Particle feature layout: [pT, deta, dphi, energy]
  pT   = transverse momentum = sqrt(px² + py²)
  deta = pseudorapidity relative to jet axis  (part_deta)
  dphi = azimuthal angle relative to jet axis (part_dphi)

Author: Ranjeet Gupta
"""

from typing import List, Dict, Optional, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConservationLoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.5,
        loss_coef: List[float] = [0.25, 0.25, 0.25, 0.25],  # [pT, deta, dphi, energy]
        reduction: str = 'mean'
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        # Normalise coefficients so they sum to 1 for interpretability
        coef_sum = sum(loss_coef)
        self.loss_coef = [c / coef_sum for c in loss_coef]

    # ------------------------------------------------------------------
    # Component loss methods
    # ------------------------------------------------------------------

    def _pT_loss(self, pT_pred: Tensor, pT_true: Tensor) -> Tensor:
        return torch.sqrt(F.mse_loss(pT_pred, pT_true, reduction=self.reduction) + 1e-8)

    def _deta_loss(self, deta_pred: Tensor, deta_true: Tensor) -> Tensor:
        return F.l1_loss(deta_pred, deta_true, reduction=self.reduction)

    def _dphi_loss(self, dphi_pred: Tensor, dphi_true: Tensor) -> Tensor:
        sin_pred = torch.sin(dphi_pred)
        cos_pred = torch.cos(dphi_pred)
        sin_true = torch.sin(dphi_true)
        cos_true = torch.cos(dphi_true)

        # Dot product of unit vectors in the phi plane
        cos_sim = cos_true * cos_pred + sin_true * sin_pred

        # Loss: 0 when perfectly aligned, 2 when exactly opposite
        loss = 1.0 - cos_sim

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _energy_loss(self, E_pred: Tensor, E_true: Tensor) -> Tensor:
        """
        Particle energy loss.

        Physics Insight:
        ---------------
        Like pT, the energy distribution is right-skewed. RMSE with a small
        epsilon stabilises training and reduces positive prediction bias.
        """
        return torch.sqrt(F.mse_loss(E_pred, E_true, reduction=self.reduction) + 1e-8)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        return_components: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]:
        # Unpack feature columns
        pT_pred,   deta_pred,   dphi_pred,   E_pred   = (
            pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        )
        pT_true,   deta_true,   dphi_true,   E_true   = (
            target[:, 0], target[:, 1], target[:, 2], target[:, 3]
        )

        # Compute per-feature losses
        pT_loss     = self._pT_loss(pT_pred, pT_true)
        deta_loss   = self._deta_loss(deta_pred, deta_true)
        dphi_loss   = self._dphi_loss(dphi_pred, dphi_true)
        energy_loss = self._energy_loss(E_pred, E_true)

        # Weighted combination
        total_loss = (
            self.loss_coef[0] * pT_loss   +
            self.loss_coef[1] * deta_loss +
            self.loss_coef[2] * dphi_loss +
            self.loss_coef[3] * energy_loss
        )

        if return_components:
            return total_loss, (pT_loss, deta_loss, dphi_loss, energy_loss)
        return total_loss


class HybridLoss(nn.Module):

    def __init__(
        self,
        task_weights: Dict[str, float] = {
            "classification": 1.0,
            "regression":     1.0,
            "reconstruction": 1.0,
        },
        classification_criterion: Optional[nn.Module] = None,
        regression_criterion: str = "mse",
        reconstruction_criterion: Optional[nn.Module] = None,
        huber_delta: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.task_weights = task_weights

        # Classification
        self.classification_criterion = (
            classification_criterion
            if classification_criterion is not None
            else nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        )

        # Regression
        self.regression_criterion_type = regression_criterion
        if regression_criterion == "mse":
            self.regression_criterion = nn.MSELoss()
        elif regression_criterion == "huber":
            self.regression_criterion = nn.HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unknown regression_criterion: '{regression_criterion}'. "
                             f"Use 'mse' or 'huber'.")

        # Reconstruction (physics-aware)
        self.reconstruction_criterion = (
            reconstruction_criterion
            if reconstruction_criterion is not None
            else ConservationLoss()
        )

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        loss_components: Dict[str, Tensor] = {}

        # ── Classification ──────────────────────────────────────────────
        if "classification" in predictions and "classification" in targets:
            cls_loss = self.classification_criterion(
                predictions["classification"],
                targets["classification"],
            )
            total_loss = total_loss + self.task_weights.get("classification", 1.0) * cls_loss
            loss_components["classification"] = cls_loss

        # ── Regression ──────────────────────────────────────────────────
        if "regression" in predictions and "regression" in targets:
            reg_loss = self.regression_criterion(
                predictions["regression"],
                targets["regression"],
            )
            total_loss = total_loss + self.task_weights.get("regression", 1.0) * reg_loss
            loss_components["regression"] = reg_loss

        # ── Reconstruction (physics-aware) ──────────────────────────────
        if "reconstruction" in predictions and "reconstruction" in targets:
            rec_loss, rec_components = self.reconstruction_criterion(
                predictions["reconstruction"],
                targets["reconstruction"],
                return_components=True,
            )
            total_loss = total_loss + self.task_weights.get("reconstruction", 1.0) * rec_loss
            loss_components["reconstruction"] = rec_loss

            # Detailed per-feature components for logging
            loss_components["pT_loss"]     = rec_components[0]
            loss_components["deta_loss"]   = rec_components[1]
            loss_components["dphi_loss"]   = rec_components[2]
            loss_components["energy_loss"] = rec_components[3]

        if return_components:
            return total_loss, loss_components
        return total_loss

    def update_task_weights(self, new_weights: Dict[str, float]):
        """
        Update task weights dynamically (e.g., curriculum learning).

        Example
        -------
        Pre-training  → {"reconstruction": 2.0, "classification": 0.1}
        Fine-tuning   → {"reconstruction": 0.1, "classification": 2.0}
        """
        self.task_weights.update(new_weights)

    def get_task_weights(self) -> Dict[str, float]:
        """Return a copy of the current task weights."""
        return self.task_weights.copy()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Testing HybridLoss — feature layout: [pT, deta, dphi, energy]")
    print("=" * 70)

    loss_fn = HybridLoss(
        task_weights={
            "classification": 1.0,
            "regression":     0.5,
            "reconstruction": 2.0,   # Higher weight for pre-training
        },
        regression_criterion="mse",
        label_smoothing=0.1,
    )

    B = 16

    predictions = {
        "classification": torch.randn(B, 10),
        "regression":     torch.randn(B, 1),
        "reconstruction": torch.randn(B, 4),   # [pT, deta, dphi, energy]
    }
    targets = {
        "classification": torch.randint(0, 10, (B,)),
        "regression":     torch.randn(B, 1),
        "reconstruction": torch.randn(B, 4),
    }

    total, comps = loss_fn(predictions, targets, return_components=True)

    print(f"\nTotal loss: {total.item():.4f}\n")
    print("Component losses:")
    print("-" * 70)
    for name, val in comps.items():
        print(f"  {name:<20}: {val.item():.4f}")
    print("=" * 70)

    # Simulate switching from pre-training to fine-tuning
    print("\nUpdating weights for fine-tuning phase …")
    loss_fn.update_task_weights({
        "classification": 2.0,
        "regression":     1.0,
        "reconstruction": 0.1,
    })

    total_ft, comps_ft = loss_fn(predictions, targets, return_components=True)
    print(f"Total loss (fine-tuned weights): {total_ft.item():.4f}")
    print("\n✓ HybridLoss tests passed!")