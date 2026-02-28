"""
Main Training Script for Foundation LorentzParT Model
======================================================
Entry point for pre-training, multi-task training, fine-tuning, and evaluation.
Usage
-----
  # Full pipeline (pretrain → multitask → finetune all tasks → evaluate)
  python main.py --config configs/foundation_config.yaml --mode all

  # Original 3-task multi-task training (classification + regression + reconstruction)
  python main.py --config configs/foundation_config.yaml --mode train

  # Self-supervised MPA pre-training only (no labels needed)
  python main.py --config configs/foundation_config.yaml --mode pretrain

  # Fine-tune a single task from pre-trained encoder
  python main.py --config configs/foundation_config.yaml --mode finetune \
      --task classification --encoder-ckpt ./outputs/pretrain/encoder_best.pt

  # Evaluate on test set (benchmarks all registered tasks)
  python main.py --config configs/foundation_config.yaml --mode eval \
      --checkpoint ./outputs/foundation_model/checkpoints/best_model.pt

"""

import os
import sys
import argparse
import yaml
import warnings
import types
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts, ReduceLROnPlateau,
    StepLR, CosineAnnealingLR
)

from models.foundation_lorentz_part import FoundationLorentzParT
from loss.hybrid_loss               import HybridLoss, ConservationLoss
from engine.multi_task_trainer      import MultiTaskTrainer, FineTuneTrainer
from utils.data_factory             import create_dataloaders


# ─────────────────────────────────────────────────────────────────────────────
# Config + distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Setup distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://',
            world_size=world_size, rank=rank,
        )
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def create_model(config: Dict) -> FoundationLorentzParT:
    """Build FoundationLorentzParT from YAML config."""
    mc = config['model']
    dc = config['data']
    return FoundationLorentzParT(
        max_num_particles      = dc['max_particles'],
        num_particle_features  = dc['num_particle_features'],
        num_classes            = mc['classification']['num_classes'],
        num_regression_targets = mc['regression']['num_targets'],
        embed_dim              = mc['embed_dim'],
        num_heads              = mc['num_heads'],
        num_layers             = mc['num_layers'],
        dropout                = mc['dropout'],
        expansion_factor       = mc['expansion_factor'],
        pair_embed_dims        = mc['pair_embed_dims'],
        in_s_channels          = mc.get('in_s_channels'),
        out_s_channels         = mc.get('out_s_channels'),
        # New foundation-model params (with safe defaults for old configs)
        latent_dim             = mc.get('generative', {}).get('latent_dim', 32),
        n_low                  = mc.get('superresolution', {}).get('n_low', 30),
        n_high                 = mc.get('superresolution', {}).get('n_high', 128),
        vae_beta               = mc.get('generative', {}).get('beta', 1.0),
    )


def create_criterion(config: Dict) -> HybridLoss:
    """Build HybridLoss from YAML config."""
    lc = config['loss']
    cc = lc.get('conservation_loss', {})
    reconstruction_criterion = ConservationLoss(
        beta      = cc.get('beta', 1.0),
        gamma     = cc.get('gamma', 0.5),
        loss_coef = cc.get('loss_coef', [0.25, 0.25, 0.25, 0.25]),
    )
    return HybridLoss(
        task_weights             = lc['task_weights'],
        regression_criterion     = lc.get('regression_criterion', 'mse'),
        reconstruction_criterion = reconstruction_criterion,
        huber_delta              = lc.get('huber_delta', 1.0),
        label_smoothing          = config['training'].get('label_smoothing', 0.0),
    )


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Build optimizer from YAML config. Supports AdamW, Adam, SGD."""
    oc   = config['training']['optimizer']
    name = oc['name'].lower()
    if name == 'adamw':
        return AdamW(model.parameters(), lr=oc['lr'],
                     weight_decay=oc.get('weight_decay', 0.01),
                     betas=tuple(oc.get('betas', [0.9, 0.999])),
                     eps=oc.get('eps', 1e-8))
    elif name == 'adam':
        return Adam(model.parameters(), lr=oc['lr'],
                    weight_decay=oc.get('weight_decay', 0.0),
                    betas=tuple(oc.get('betas', [0.9, 0.999])),
                    eps=oc.get('eps', 1e-8))
    elif name == 'sgd':
        return SGD(model.parameters(), lr=oc['lr'],
                   momentum=oc.get('momentum', 0.9),
                   weight_decay=oc.get('weight_decay', 0.0))
    raise ValueError(f"Unknown optimizer: {oc['name']}")


def create_scheduler(optimizer, config: Dict):
    """Build LR scheduler from YAML config."""
    if 'scheduler' not in config['training']:
        return None
    sc   = config['training']['scheduler']
    name = sc['name'].lower()
    if name == 'cosineannealingwarmrestarts':
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=sc.get('T_0', 10),
            T_mult=sc.get('T_mult', 2), eta_min=sc.get('eta_min', 1e-6))
    elif name == 'reducelronplateau':
        return ReduceLROnPlateau(
            optimizer, mode=sc.get('mode', 'min'),
            factor=sc.get('factor', 0.5), patience=sc.get('patience', 5),
            min_lr=sc.get('min_lr', 1e-6))
    elif name == 'steplr':
        return StepLR(optimizer, step_size=sc.get('step_size', 10),
                      gamma=sc.get('gamma', 0.1))
    elif name == 'cosineannealinglr':
        return CosineAnnealingLR(
            optimizer, T_max=config['training']['num_epochs'],
            eta_min=sc.get('eta_min', 1e-6))
    raise ValueError(f"Unknown scheduler: {sc['name']}")


def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — MPA Pre-Training
# ─────────────────────────────────────────────────────────────────────────────

def stage_pretrain(config: Dict, device: torch.device) -> str:
    """
    Self-supervised Masked Particle Autoencoder pre-training.
    No labels required — learns from reconstruction loss only.
    Returns path to saved encoder checkpoint.
    """
    from pretrain import MaskedParticlePreTrainer

    print("\n" + "=" * 70)
    print("  STAGE 1 — Masked Particle Autoencoder Pre-Training")
    print("=" * 70)

    dc = config['data']
    tc = config['training']

    train_loader, val_loader, _ = create_dataloaders(
        data_path    = dc['data_path'],
        batch_size   = tc['batch_size'],
        num_workers  = config['hardware']['num_workers'],
        mask_particle= True,
        mask_strategy= dc.get('mask_strategy', 'biased'),
    )

    model    = create_model(config)
    save_dir = Path(config['tracking']['output_dir']) / 'pretrain'

    args = types.SimpleNamespace(
        mask_ratio      = dc.get('mask_ratio', 0.30),
        mask_strategy   = dc.get('mask_strategy', 'biased'),
        epochs          = tc.get('pretrain_epochs', 50),
        lr              = tc['optimizer']['lr'],
        warmup_epochs   = tc.get('pretrain_warmup', 5),
        grad_clip       = tc.get('gradient_clip_val', 1.0),
        mixed_precision = tc.get('mixed_precision', True),
        save_dir        = str(save_dir),
    )

    trainer = MaskedParticlePreTrainer(model, train_loader, val_loader, device, args)
    trainer.pretrain()

    return str(save_dir / 'encoder_best.pt')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Multi-Task Training (classification + regression + reconstruction)
# ─────────────────────────────────────────────────────────────────────────────

def stage_multitask(
    config: Dict,
    device: torch.device,
    encoder_ckpt: Optional[str] = None,
    is_distributed: bool = False,
    rank: int = 0,
    pretrain_mode: bool = False,
) -> str:
    """
    Original multi-task training loop.
    Optionally initialises encoder from MPA pre-trained weights.
    Returns path to best model checkpoint.
    """
    is_main = (rank == 0)

    if is_main:
        print("\n" + "=" * 70)
        stage = "Pre-Training (Reconstruction Only)" if pretrain_mode else \
                "STAGE 2 — Multi-Task Training"
        print(f"  {stage}")
        print("=" * 70)

    dc = config['data']
    tc = config['training']
    dy = config['dynamics']

    train_loader, val_loader, test_loader = create_dataloaders(
        data_path    = dc['data_path'],
        batch_size   = tc['batch_size'],
        num_workers  = config['hardware']['num_workers'],
        max_particles= dc['max_particles'],
        mask_particle= dc['mask_particle'] or pretrain_mode,
        mask_strategy= dc['mask_strategy'],
        distributed  = is_distributed,
        seed         = config['tracking']['seed'],
    )

    if is_main:
        print(f"  Train: {len(train_loader.dataset):,}  "
              f"Val: {len(val_loader.dataset):,}  "
              f"Test: {len(test_loader.dataset):,}")

    model     = create_model(config)
    criterion = create_criterion(config)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    # Load pre-trained encoder if provided
    if encoder_ckpt and os.path.exists(encoder_ckpt):
        try:
            model.encoder.load_state_dict(
                torch.load(encoder_ckpt, map_location='cpu'), strict=False)
            if is_main:
                print(f"  ✓ Loaded pre-trained encoder from {encoder_ckpt}")
        except Exception as e:
            if is_main:
                print(f"  [WARNING] Could not load encoder: {e}")

    if is_main:
        model.print_parameter_summary()

    # Switch to reconstruction-only mode for pre-training
    if pretrain_mode:
        criterion.update_task_weights(
            {'classification': 0.0, 'regression': 0.0, 'reconstruction': 1.0})

    task_modes = config['tasks'].copy()
    # Remove non-bool keys added for fine-tuning
    task_modes = {k: v for k, v in task_modes.items()
                  if k in ('classification', 'regression', 'reconstruction')
                  and isinstance(v, bool)}
    if pretrain_mode:
        task_modes = {'classification': False, 'regression': False, 'reconstruction': True}

    es = dy.get('early_stopping', {})
    trainer = MultiTaskTrainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        test_loader    = test_loader,
        criterion      = criterion,
        optimizer      = optimizer,
        scheduler      = scheduler,
        device         = str(device),
        num_epochs     = tc['num_epochs'],
        task_modes     = task_modes,
        mixed_precision= tc['mixed_precision'],
        gradient_clip_val = tc.get('gradient_clip_val'),
        save_dir       = config['tracking']['output_dir'],
        experiment_name= config['experiment']['name'],
        log_interval   = dy['logging']['log_interval'],
        save_checkpoint_interval = dy['checkpointing']['save_interval'],
        early_stopping_patience  = es.get('patience') if es.get('enabled') else None,
        verbose        = is_main,
    )
    trainer.train()

    out = Path(config['tracking']['output_dir']) / config['experiment']['name']
    return str(out / 'checkpoints' / 'best_model.pt')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Fine-Tuning (single task)
# ─────────────────────────────────────────────────────────────────────────────

def stage_finetune(
    config: Dict,
    device: torch.device,
    task: str,
    encoder_ckpt: Optional[str],
    finetune_mode: str = 'partial',
) -> str:
    """
    Fine-tune on a single downstream task:
      classification | regression | generative | superresolution
    """
    print(f"\n{'='*70}")
    print(f"  STAGE 3 — Fine-Tuning: {task.upper()}  (mode={finetune_mode})")
    print(f"{'='*70}")

    dc = config['data']
    tc = config['training']

    train_loader, val_loader, _ = create_dataloaders(
        data_path    = dc['data_path'],
        batch_size   = tc['batch_size'],
        num_workers  = config['hardware']['num_workers'],
        mask_particle= (task == 'reconstruction'),
        mask_strategy= dc['mask_strategy'],
    )

    model = create_model(config)

    # Separate LRs for encoder vs task head (standard fine-tuning practice)
    lr_base = tc['optimizer']['lr']
    optimizer = AdamW([
        {'params': model.encoder.parameters(),           'lr': lr_base * 0.1},
        {'params': model.classification_head.parameters(),'lr': lr_base},
        {'params': model.regression_head.parameters(),   'lr': lr_base},
        {'params': model.generative_head.parameters(),   'lr': lr_base},
        {'params': model.superresolution_head.parameters(),'lr': lr_base},
    ], weight_decay=tc['optimizer'].get('weight_decay', 1e-2))

    out_dir = Path(config['tracking']['output_dir']) / 'finetune'

    trainer = FineTuneTrainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        task            = task,
        optimizer       = optimizer,
        device          = str(device),
        num_epochs      = tc['num_epochs'],
        finetune_mode   = finetune_mode,
        unfreeze_last_k = config['tasks'].get('unfreeze_last_k', 4),
        mixed_precision = tc['mixed_precision'],
        gradient_clip_val = tc.get('gradient_clip_val', 1.0),
        save_dir        = str(out_dir),
        log_interval    = config['dynamics']['logging']['log_interval'],
        verbose         = True,
    )

    if encoder_ckpt:
        trainer.load_pretrained_encoder(encoder_ckpt)

    trainer.finetune()
    return str(out_dir / task / f'{task}_best.pt')


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Evaluation & Benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def stage_evaluate(
    config: Dict,
    device: torch.device,
    checkpoint: Optional[str] = None,
    tasks: Optional[list] = None,
):
    """
    Fully-implemented evaluation (replaces the TODO placeholder).
    Benchmarks all requested tasks and prints a comparison table.
    """
    from evaluate import EvaluationHarness

    print(f"\n{'='*70}")
    print(f"  STAGE 4 — Evaluation & Benchmarking")
    print(f"{'='*70}")

    dc = config['data']

    _, _, test_loader = create_dataloaders(
        data_path    = dc['data_path'],
        batch_size   = 256,
        num_workers  = config['hardware']['num_workers'],
        mask_particle= False,
    )

    harness = EvaluationHarness(test_loader, device)

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model = create_model(config)
        model.load_state_dict(state.get('model_state_dict', state), strict=False)
        model.eval()
        harness.register("FoundationLorentzParT (trained)", model)
        print(f"  ✓ Loaded checkpoint from {checkpoint}")
    else:
        print("  [WARNING] No valid checkpoint — evaluating untrained model.")
        harness.register("FoundationLorentzParT (untrained)", create_model(config))

    eval_tasks = tasks or ['classification', 'regression']
    results    = harness.run(tasks=eval_tasks)

    out_path = Path(config['tracking']['output_dir']) / 'eval_results.json'
    harness.save_results(str(out_path))
    return results


# Argument parsing

def parse_args():
    p = argparse.ArgumentParser(
        description="Foundation LorentzParT — Multi-Task Training for HEP"
    )
    p.add_argument('--config', default='configs/foundation_config.yaml')
    p.add_argument('--mode',   default='train',
                   choices=['train', 'pretrain', 'finetune', 'eval', 'all'],
                   help=(
                       'train    = original 3-task multi-task loop  |  '
                       'pretrain = MPA pre-training only  |  '
                       'finetune = single-task fine-tune  |  '
                       'eval     = evaluation + benchmarking  |  '
                       'all      = pretrain→train→finetune all→eval'
                   ))
    p.add_argument('--task',   default='all',
                   choices=['classification','regression',
                            'generative','superresolution','all'],
                   help='Task for --mode finetune or eval')
    p.add_argument('--checkpoint',    default=None,
                   help='Model checkpoint for eval or resuming training')
    p.add_argument('--encoder-ckpt',  default=None,
                   help='Pre-trained encoder checkpoint for fine-tuning')
    p.add_argument('--finetune-mode', default='partial',
                   choices=['frozen','partial','full'])
    p.add_argument('--data-path',     default=None)
    p.add_argument('--output-dir',    default=None)
    return p.parse_args()


# Main entry point

def main():
    args   = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.output_dir:
        config['tracking']['output_dir'] = args.output_dir

    # Reproducibility
    set_seed(
        config['tracking']['seed'],
        deterministic=config['tracking'].get('deterministic', False),
    )

    # Distributed setup
    is_distributed, rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    # Device
    hw_device = config['hardware']['device']
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
    elif torch.cuda.is_available() and hw_device != 'cpu':
        device = torch.device(hw_device)
    else:
        device = torch.device('cpu')

    if is_main:
        print("=" * 70)
        print("  Foundation LorentzParT")
        print("=" * 70)
        print(f"  Experiment : {config['experiment']['name']}")
        print(f"  Mode       : {args.mode}")
        print(f"  Device     : {device}")
        print(f"  Distributed: {is_distributed}  (world={world_size})")
        print("=" * 70)

    # Mode dispatch 

    encoder_ckpt = args.encoder_ckpt
    model_ckpt   = args.checkpoint

    # --mode pretrain  OR  original --mode pretrain shortcut
    if args.mode == 'pretrain':
        stage_pretrain(config, device)
        return

    # --mode train  (original 3-task multi-task, no separate pre-training)
    if args.mode == 'train':
        pretrain_only = config['tasks'].get('pretrain_reconstruction_only', False)
        stage_multitask(config, device,
                        encoder_ckpt   = encoder_ckpt,
                        is_distributed = is_distributed,
                        rank           = rank,
                        pretrain_mode  = pretrain_only)
        return

    # --mode finetune  (single task)
    if args.mode == 'finetune':
        tasks = (['classification','regression','generative','superresolution']
                 if args.task == 'all' else [args.task])
        for task in tasks:
            stage_finetune(config, device, task=task,
                           encoder_ckpt  = encoder_ckpt,
                           finetune_mode = args.finetune_mode)
        return

    # --mode eval  (fully implemented — no more TODO)
    if args.mode == 'eval':
        eval_tasks = (['classification','regression']
                      if args.task == 'all' else [args.task])
        stage_evaluate(config, device, checkpoint=model_ckpt, tasks=eval_tasks)
        return

    # --mode all  (complete pipeline)
    if args.mode == 'all':
        # Stage 1: MPA pre-training
        encoder_ckpt = stage_pretrain(config, device)

        # Stage 2: Multi-task fine-tuning from pre-trained encoder
        model_ckpt = stage_multitask(config, device,
                                     encoder_ckpt   = encoder_ckpt,
                                     is_distributed = is_distributed,
                                     rank           = rank)

        # Stage 3: Per-task fine-tuning
        finetune_tasks = config['tasks'].get(
            'finetune_tasks',
            ['classification', 'regression', 'generative', 'superresolution']
        )
        for task in finetune_tasks:
            mode = config['tasks'].get('finetune_modes', {}).get(task, 'partial')
            stage_finetune(config, device, task=task,
                           encoder_ckpt  = encoder_ckpt,
                           finetune_mode = mode)

        # Stage 4: Evaluate
        stage_evaluate(config, device, checkpoint=model_ckpt,
                       tasks=['classification', 'regression'])
        return


if __name__ == "__main__":
    main()