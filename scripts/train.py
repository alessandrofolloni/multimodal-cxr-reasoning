"""
Main training script - QUICK TEST VERSION
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import CheXpertMultimodalDataset
from models.reasoning_model import MultimodalReasoningModel
from training.trainer import Trainer


def main():
    # Config - QUICK TEST
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    LR = 1e-4
    NUM_WORKERS = 0
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Quick test settings
    QUICK_TEST = True
    TEST_SAMPLES = 1000
    
    print(f"{'='*80}")
    print("Training Multimodal Reasoning Model - QUICK TEST")
    print(f"{'='*80}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Test samples: {TEST_SAMPLES if QUICK_TEST else 'ALL'}")
    
    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = CheXpertMultimodalDataset(
        csv_path="data/processed/train_with_notes.csv",
        data_root="data/chexpert",
        image_size=224,
        split="train"
    )
    
    val_dataset = CheXpertMultimodalDataset(
        csv_path="data/processed/valid_with_notes.csv",
        data_root="data/chexpert",
        image_size=224,
        split="valid"
    )
    
    # Subset for quick testing
    if QUICK_TEST:
        print(f"⚡ QUICK TEST MODE: Using {TEST_SAMPLES} training samples")
        train_indices = np.random.choice(len(train_dataset), size=min(TEST_SAMPLES, len(train_dataset)), replace=False)
        val_indices = np.random.choice(len(val_dataset), size=min(200, len(val_dataset)), replace=False)
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    print(f"✅ Datasets loaded")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    # Initialize model
    print(f"\nInitializing model...")
    model = MultimodalReasoningModel(
        num_classes=14,
        freeze_encoders=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        lr=LR,
        num_epochs=NUM_EPOCHS
    )
    
    # Train!
    trainer.train()
    
    print(f"\n✅ Training complete! Check outputs/checkpoints/ for saved models.")


if __name__ == "__main__":
    main()