"""
Training pipeline for multimodal reasoning model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
from tqdm import tqdm
from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class Trainer:
    """Training and evaluation for multimodal model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "mps",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        save_dir: str = "outputs/checkpoints"
    ):
        """
        Args:
            model: MultimodalReasoningModel
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on ('mps', 'cuda', or 'cpu')
            lr: Learning rate
            weight_decay: AdamW weight decay
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function (BCE for multi-label classification)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        # Tracking
        self.best_val_auc = 0.0
        self.train_losses = []
        self.val_aucs = []
        
        print(f"✅ Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Learning rate: {lr}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            notes = batch['clinical_note']  # List of strings
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, notes)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        all_labels = []
        all_preds = []
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            images = batch['image'].to(self.device)
            notes = batch['clinical_note']
            labels = batch['labels'].to(self.device)
            
            # Forward
            outputs = self.model(images, notes)
            logits = outputs['logits']
            
            # Loss
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            # Predictions
            preds = torch.sigmoid(logits)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        
        # Concatenate
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        
        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        
        # AUC per class (handle cases with only one class)
        aucs = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
        
        mean_auc = np.mean(aucs) if aucs else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'val_auc': mean_auc
        }
        
        return metrics
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*80}")
        print("Starting Training")
        print(f"{'='*80}\n")
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            # Track
            self.train_losses.append(train_loss)
            self.val_aucs.append(val_metrics['val_auc'])
            
            # Print
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val AUC: {val_metrics['val_auc']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if val_metrics['val_auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['val_auc']
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"  ✅ Saved best model (AUC: {self.best_val_auc:.4f})")
            
            # Save latest
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)
        
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_aucs': self.val_aucs
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)