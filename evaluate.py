"""
evaluate.py — Evaluation & Benchmarking for FoundationLorentzParT

Metrics
-------
  Classification  : Accuracy, macro-F1, AUC-ROC (sklearn)
  Regression      : MSE, MAE, R², mass resolution (σ/μ)
  Generative      : MMD feature-space proxy, marginal KL divergence
  SuperResolution : Chamfer distance (η-φ), pT EMD (scipy)

  Run in notebook 03
  !python evaluate.py \
    --ckpt-cls  outputs/finetune/classification/classification_best.pt \
    --ckpt-reg  outputs/finetune/regression/regression_best.pt \
    --ckpt-gen  outputs/finetune/generative/generative_best.pt \
    --ckpt-sr   outputs/finetune/superresolution/superresolution_best.pt \
    --tasks classification regression generative superresolution \
    --data-path data/jetclass_100k \
    --batch-size 256 \
    --num-workers 2 \
    --save-results outputs/eval_all_results.json
  """

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.stats import wasserstein_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.models.foundation_lorentz_part import FoundationLorentzParT
from src.utils.data_factory import create_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    accuracy:   float
    macro_f1:   float
    auc_roc:    Optional[float]
    top3_acc:   float

    def __str__(self):
        auc = f"{self.auc_roc:.4f}" if self.auc_roc else "  N/A "
        return (f"Acc={self.accuracy:.4f}  Top3={self.top3_acc:.4f}  "
                f"F1={self.macro_f1:.4f}  AUC={auc}")


@dataclass
class RegressionResult:
    mse:        float
    mae:        float
    r2:         float
    resolution: float   # σ(pred-true)/μ(|true|)

    def __str__(self):
        return (f"MSE={self.mse:.4f}  MAE={self.mae:.4f}  "
                f"R²={self.r2:.4f}  Res={self.resolution:.4f}")


@dataclass
class GenerativeResult:
    mmd:           float  # MMD in embedding space
    marginal_kl:   float  # KL(true||gen) on pT marginal
    marginal_mse:  float  # per-feature histogram MSE

    def __str__(self):
        return (f"MMD={self.mmd:.4f}  "
                f"pT-KL={self.marginal_kl:.4f}  "
                f"Marg-MSE={self.marginal_mse:.4f}")


@dataclass
class SuperResolutionResult:
    chamfer:  float
    pt_emd:   float
    mult_err: float

    def __str__(self):
        return (f"Chamfer={self.chamfer:.4f}  "
                f"pT-EMD={self.pt_emd:.4f}  "
                f"Mult-Err={self.mult_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluators
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_classification(model, loader, device) -> ClassificationResult:
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        x  = batch['x'].to(device)
        pm = batch['padding_mask'].to(device)
        U  = batch['U'].to(device)
        out = model(x, pm, U, task='classification')
        probs  = F.softmax(out['classification'], -1).cpu().numpy()
        labels = batch['classification_target'].numpy()
        all_probs.append(probs);  all_labels.append(labels)

    probs  = np.concatenate(all_probs)    # (N, C)
    labels = np.concatenate(all_labels)   # (N,)
    preds  = probs.argmax(-1)

    accuracy = float((preds == labels).mean())

    # Top-3 accuracy
    top3     = np.argsort(probs, axis=-1)[:, -3:]
    top3_acc = float(np.any(top3 == labels[:, None], axis=-1).mean())

    if HAS_SKLEARN:
        macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0))
        try:
            auc = float(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))
        except Exception:
            auc = None
    else:
        macro_f1, auc = 0.0, None

    return ClassificationResult(accuracy=accuracy, macro_f1=macro_f1,
                                 auc_roc=auc, top3_acc=top3_acc)


@torch.no_grad()
def evaluate_regression(model, loader, device) -> RegressionResult:
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        x  = batch['x'].to(device)
        pm = batch['padding_mask'].to(device)
        U  = batch['U'].to(device)
        out  = model(x, pm, U, task='regression')
        all_pred.append(out['regression'].cpu().numpy())
        all_true.append(batch['regression_target'].numpy())

    pred = np.concatenate(all_pred).squeeze()
    true = np.concatenate(all_true).squeeze()

    mse  = float(np.mean((pred - true) ** 2))
    mae  = float(np.mean(np.abs(pred - true)))
    ss_r = np.sum((pred - true) ** 2)
    ss_t = np.sum((true - true.mean()) ** 2) + 1e-10
    r2   = float(1.0 - ss_r / ss_t)
    res  = float((pred - true).std() / (np.abs(true).mean() + 1e-10))

    return RegressionResult(mse=mse, mae=mae, r2=r2, resolution=res)


@torch.no_grad()
def evaluate_generative(model, loader, device, n_gen=30) -> GenerativeResult:
    """MMD + marginal histogram comparison between true and generated particles."""
    model.eval()
    real_parts, gen_parts = [], []

    for batch in loader:
        x  = batch['x'].to(device)
        pm = batch['padding_mask'].to(device)
        U  = batch['U'].to(device)
        # True particles (first n_gen valid per jet)
        real_parts.append(x[:, :n_gen, :4].cpu().numpy())
        # Generated particles
        gen = model.generate_particles(x, pm, U, num_particles=n_gen)
        gen_parts.append(gen.cpu().numpy())

    real = np.concatenate(real_parts).reshape(-1, 4)
    fake = np.concatenate(gen_parts).reshape(-1, 4)

    # Sub-sample for tractability
    idx_r = np.random.choice(len(real), min(5000, len(real)), replace=False)
    idx_f = np.random.choice(len(fake), min(5000, len(fake)), replace=False)
    r, f  = real[idx_r], fake[idx_f]

    # MMD with RBF kernel
    def rbf(x, y, s=1.0):
        d = ((x[:, None] - y[None]) ** 2).sum(-1)
        return np.exp(-d / (2 * s ** 2)).mean()

    mmd = float(max(rbf(r, r) + rbf(f, f) - 2 * rbf(r, f), 0.0))

    # Per-feature marginal KL + MSE
    n_bins = 50
    kls, mses = [], []
    for i in range(4):
        lo = min(real[:, i].min(), fake[:, i].min())
        hi = max(real[:, i].max(), fake[:, i].max()) + 1e-6
        bins = np.linspace(lo, hi, n_bins + 1)
        rh, _ = np.histogram(real[:, i], bins=bins, density=True)
        fh, _ = np.histogram(fake[:, i], bins=bins, density=True)
        rh += 1e-10; fh += 1e-10
        rh /= rh.sum(); fh /= fh.sum()
        if i == 0:  # pT KL
            kls.append(float(np.sum(rh * np.log(rh / fh))))
        mses.append(float(np.mean((rh - fh) ** 2)))

    return GenerativeResult(mmd=mmd, marginal_kl=float(np.mean(kls)),
                             marginal_mse=float(np.mean(mses)))


@torch.no_grad()
def evaluate_superresolution(model, loader, device, n_low=30) -> SuperResolutionResult:
    model.eval()
    chamfers, emds, mult_errs = [], [], []

    for batch in loader:
        x  = batch['x'][:, :n_low, :].to(device)
        pm = batch['padding_mask'][:, :n_low].to(device)
        U  = batch['U'][:, :n_low, :n_low, :].to(device)

        out    = model(x, pm, U, task='superresolution')
        pred   = out['high_res'].cpu().numpy()                  # (B, n_high, 4)
        if 'superres_target' in batch:
            true = batch['superres_target'].numpy()
        else:
            continue

        for b in range(pred.shape[0]):
            # Chamfer in (η, φ) space
            p, t = pred[b, :, 1:3], true[b, :, 1:3]
            d_pt = ((p[:, None] - t[None]) ** 2).sum(-1)
            chamfers.append(float(d_pt.min(1).mean() + d_pt.min(0).mean()))
            # pT EMD
            if HAS_SCIPY:
                emds.append(float(wasserstein_distance(pred[b,:,0], true[b,:,0])))
            # Multiplicity error
            n_pred = (pred[b, :, 0] > 0.1).sum()
            n_true = (true[b, :, 0] > 0.1).sum()
            mult_errs.append(abs(n_pred - n_true) / max(n_true, 1))

    return SuperResolutionResult(
        chamfer=float(np.mean(chamfers)) if chamfers else float('nan'),
        pt_emd=float(np.mean(emds))      if emds    else float('nan'),
        mult_err=float(np.mean(mult_errs)) if mult_errs else float('nan'),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarking harness
# ─────────────────────────────────────────────────────────────────────────────

class EvaluationHarness:
    """
    Registers multiple models, runs all tasks, prints comparison table.

    Usage:
        harness = EvaluationHarness(test_loader, device)
        harness.register("Foundation (pretrained)",  model_a)
        harness.register("Baseline (no pretrain)",   model_b)
        results = harness.run(tasks=['classification', 'regression'])
    """

    TASK_FNS = {
        'classification':  evaluate_classification,
        'regression':      evaluate_regression,
        'generative':      evaluate_generative,
        'superresolution': evaluate_superresolution,
    }

    def __init__(self, loader, device: torch.device):
        self.loader  = loader
        self.device  = device
        self._models: Dict[str, torch.nn.Module] = {}
        self.results: Dict[str, Dict] = {}

    def register(self, name: str, model):
        self._models[name] = model.to(self.device)

    def run(self, tasks: Optional[List[str]] = None) -> Dict:
        if tasks is None:
            tasks = list(self.TASK_FNS.keys())

        for task in tasks:
            print(f"\n[Task: {task.upper()}]")
            self.results[task] = {}
            fn = self.TASK_FNS[task]

            for name, model in self._models.items():
                print(f"  Evaluating: {name} ...", end=" ", flush=True)
                try:
                    result = fn(model, self.loader, self.device)
                    self.results[task][name] = result
                    print(f"→ {result}")
                except Exception as e:
                    print(f"FAILED: {e}")

        self._print_table()
        return self.results

    def _print_table(self):
        print("\n" + "=" * 72)
        print("  BENCHMARK SUMMARY")
        print("=" * 72)

        col_widths = {
            'classification':  {'Acc': 7, 'Top3': 7, 'F1': 7, 'AUC': 7},
            'regression':      {'MSE': 8, 'MAE': 8, 'R²': 8, 'Res': 8},
            'generative':      {'MMD': 8, 'pT-KL': 8, 'Marg-MSE': 10},
            'superresolution': {'Chamfer': 9, 'pT-EMD': 8, 'Mult-Err': 10},
        }
        header_row = {
            'classification':  lambda r: f"{r.accuracy:7.4f} {r.top3_acc:7.4f} {r.macro_f1:7.4f} {(r.auc_roc or 0):7.4f}",
            'regression':      lambda r: f"{r.mse:8.4f} {r.mae:8.4f} {r.r2:8.4f} {r.resolution:8.4f}",
            'generative':      lambda r: f"{r.mmd:8.4f} {r.marginal_kl:8.4f} {r.marginal_mse:10.4f}",
            'superresolution': lambda r: f"{r.chamfer:9.4f} {r.pt_emd:8.4f} {r.mult_err:10.4f}",
        }

        for task, task_results in self.results.items():
            print(f"\n  [{task.upper()}]")
            hcols = "  ".join(col_widths[task].keys())
            print(f"  {'Model':<38} {hcols}")
            print("  " + "-" * 66)
            for name, r in task_results.items():
                print(f"  {name:<38} {header_row[task](r)}")

        print("\n" + "=" * 72)

    def save_results(self, path: str):
        """Save results to JSON (scalars only)."""
        out = {}
        for task, task_res in self.results.items():
            out[task] = {}
            for name, r in task_res.items():
                out[task][name] = {k: float(v) if v is not None else None
                                   for k, v in r.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  ✓ Results saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Evaluate FoundationLorentzParT")
    p.add_argument('--config',    default='configs/foundation_config.yaml')
    p.add_argument('--data-path', default='./data/jetclass_100k')
    p.add_argument('--ckpt',      default=None,
                   help='Primary model checkpoint (or use --ckpt-cls, --ckpt-reg, etc.)')
    p.add_argument('--ckpt-cls',  default=None)
    p.add_argument('--ckpt-reg',  default=None)
    p.add_argument('--ckpt-gen',  default=None)
    p.add_argument('--ckpt-sr',   default=None)
    p.add_argument('--tasks',     nargs='+',
                   default=['classification', 'regression'],
                   choices=['classification', 'regression', 'generative', 'superresolution'])
    p.add_argument('--batch-size',  type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--save-results', default='./outputs/eval_results.json')
    return p.parse_args()


def _load_model(ckpt_path: str, device: torch.device) -> FoundationLorentzParT:
    """Load a FoundationLorentzParT from checkpoint."""
    state = torch.load(ckpt_path, map_location=device)
    # Infer model config from checkpoint or use defaults
    model = FoundationLorentzParT()
    sd    = state.get('model_state_dict', state)
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = create_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mask_particle=False,
    )

    harness = EvaluationHarness(test_loader, device)

    # Register models for each requested task
    task_ckpts = {
        'classification':  args.ckpt_cls or args.ckpt,
        'regression':      args.ckpt_reg or args.ckpt,
        'generative':      args.ckpt_gen or args.ckpt,
        'superresolution': args.ckpt_sr  or args.ckpt,
    }

    registered = set()
    for task in args.tasks:
        ckpt = task_ckpts.get(task)
        if ckpt and ckpt not in registered:
            try:
                name  = f"FoundationLorentzParT ({Path(ckpt).stem})"
                model = _load_model(ckpt, device)
                harness.register(name, model)
                registered.add(ckpt)
                print(f"  Registered: {name}")
            except Exception as e:
                print(f"  Could not load {ckpt}: {e}")

    if not registered:
        print("No checkpoints provided — running dummy sanity check with untrained model.")
        model = FoundationLorentzParT()
        harness.register("FoundationLorentzParT (untrained)", model)

    results = harness.run(tasks=args.tasks)
    harness.save_results(args.save_results)


if __name__ == "__main__":
    main()