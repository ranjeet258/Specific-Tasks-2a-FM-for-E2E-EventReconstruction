"""
Train baseline models (ParT, ParticleNet) on 100k JetClass subset.

 Benchmark requirement
- Train vanilla ParT and ParticleNet on same 100k data
- Parameter-matched to FoundationLorentzParT (~2.4M)
- Fair comparison for evaluating improvements
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

# Import baseline models from evaluate.py
from evaluate_updated import ParticleTransformerBaseline, ParticleNetBaseline
from src.utils.data_factory import get_dataloaders


def train_baseline(model, train_loader, val_loader, args, device):
     # Optimizer (same as FoundationLorentzParT)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Training {args.model.upper()} Baseline")
    print(f"{'='*80}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"{'='*80}\n")
    
    for epoch in range(args.epochs):
        # ====================================================================
        # TRAINING
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch_gpu = {
                k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = model(batch_gpu)
            logits = outputs['classification_logits']
            labels = batch_gpu['label']
            
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # ====================================================================
        # VALIDATION
        # ====================================================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar_val:
                batch_gpu = {
                    k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()
                }
                
                outputs = model(batch_gpu)
                logits = outputs['classification_logits']
                labels = batch_gpu['label']
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
                
                pbar_val.set_postfix({
                    'val_acc': f'{val_correct/val_total:.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # ====================================================================
        # LOGGING & CHECKPOINTING
        # ====================================================================
        print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            save_path = Path(args.save_dir) / f"{args.model}_baseline_100k.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best! Saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
        
        print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"{'='*80}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Model saved to: {Path(args.save_dir) / f'{args.model}_baseline_100k.pt'}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models for GSoC Task 2a benchmarking"
    )
    parser.add_argument('--model', choices=['part', 'particlenet'], required=True,
                       help='Baseline model to train')
    parser.add_argument('--config', type=str, default='configs/foundation_config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for AdamW')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save-dir', type=str, default='outputs/baselines',
                       help='Directory to save checkpoints')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"GSoC 2026 Task 2a: Baseline Training")
    print(f"{'='*80}")
    print(f"Model: {args.model.upper()}")
    print(f"Device: {device}")
    print(f"Dataset: {config['data']['data_path']}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        config=config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"✓ Train set: {len(train_loader.dataset)} events")
    print(f"✓ Val set:   {len(val_loader.dataset)} events")
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    print(f"\nInitializing {args.model.upper()} model...")
    
    if args.model == 'part':
        model = ParticleTransformerBaseline(
            input_dim=4,
            num_classes=config['data']['num_classes'],
            embed_dim=128,
            num_layers=8,
            num_heads=8
        )
    else:  # particlenet
        model = ParticleNetBaseline(
            input_dim=4,
            num_classes=config['data']['num_classes'],
            hidden_dim=128,
            num_layers=8,
            k=16
        )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}")
    print(f"  (Target: ~2.4M to match FoundationLorentzParT)")
    
    # ========================================================================
    # TRAIN
    # ========================================================================
    train_baseline(model, train_loader, val_loader, args, device)
    
    print(f"\n Success! Baseline ready for evaluation.")
    print(f"   Next: Run comparative evaluation with:")
    print(f"   python evaluate.py --baseline-{args.model} {args.save_dir}/{args.model}_baseline_100k.pt")


if __name__ == '__main__':
    main()
