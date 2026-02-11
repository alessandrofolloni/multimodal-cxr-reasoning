"""
Evaluation script for trained multimodal reasoning model.
Tests the model on validation set and generates detailed metrics.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import json
from PIL import Image
from torchvision import transforms
from transformers import ViTModel, AutoModel, AutoTokenizer

# Set paths
project_root = Path(__file__).parent.parent


# =============================================================================
# MODEL COMPONENTS (inline to avoid import issues)
# =============================================================================

class CheXpertMultimodalDataset(Dataset):
    def __init__(self, csv_path, data_root, image_size=224, split="train"):
        self.data_root = Path(data_root)
        self.split = split
        self.df = pd.read_csv(csv_path)
        
        self.pathology_columns = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_root / row['Path']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        clinical_note = row['clinical_note']
        
        labels = []
        for col in self.pathology_columns:
            val = row[col]
            labels.append(1.0 if pd.notna(val) and val > 0 else 0.0)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return {'image': image, 'clinical_note': clinical_note, 'labels': labels}


class ImageEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224", output_dim=768, freeze=False):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        if freeze:
            for param in self.vit.parameters():
                param.requires_grad = False
        self.projection = nn.Identity()
    
    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state[:, 0, :]


class TextEncoder(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", max_length=128, output_dim=768, freeze=False):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.projection = nn.Identity()
    
    def forward(self, texts):
        encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                 max_length=self.max_length, return_tensors='pt')
        input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.img_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.text_to_img_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.ln_img = nn.LayerNorm(embed_dim)
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ffn_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout)
        )
    
    def forward(self, image_embed, text_embed):
        img = image_embed.unsqueeze(1)
        txt = text_embed.unsqueeze(1)
        
        img_attended, _ = self.img_to_text_attn(img, txt, txt)
        img = self.ln_img(image_embed + img_attended.squeeze(1))
        img = img + self.ffn_img(img)
        
        txt_attended, _ = self.text_to_img_attn(txt, img.unsqueeze(1), img.unsqueeze(1))
        txt = self.ln_text(text_embed + txt_attended.squeeze(1))
        txt = txt + self.ffn_text(txt)
        
        fused = torch.cat([img, txt], dim=-1)
        return self.final_proj(fused)


class MultimodalReasoningModel(nn.Module):
    def __init__(self, num_classes=14, freeze_encoders=False):
        super().__init__()
        self.image_encoder = ImageEncoder(freeze=freeze_encoders)
        self.text_encoder = TextEncoder(freeze=freeze_encoders)
        self.fusion = CrossAttentionFusion()
        self.classifier = nn.Sequential(
            nn.Linear(768, 384), nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(384, num_classes)
        )
    
    def forward(self, images, clinical_notes):
        image_embed = self.image_encoder(images)
        text_embed = self.text_encoder(clinical_notes)
        fused_embed = self.fusion(image_embed, text_embed)
        logits = self.classifier(fused_embed)
        return {'logits': logits, 'image_embed': image_embed, 
                'text_embed': text_embed, 'fused_embed': fused_embed}


# =============================================================================
# EVALUATOR
# =============================================================================

class ModelEvaluator:
    """Comprehensive evaluation of trained model."""
    
    def __init__(self, model_path, device="mps"):
        self.device = device
        self.pathology_names = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Load model
        print("Loading model...")
        self.model = MultimodalReasoningModel(num_classes=14, freeze_encoders=False)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")
        
        # Store checkpoint info
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'best_auc': checkpoint.get('metrics', {}).get('val_auc', 'N/A')
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Run evaluation on validation set."""
        print("\n" + "="*80)
        print("Running Evaluation")
        print("="*80)
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch['image'].to(self.device)
            notes = batch['clinical_note']
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(images, notes)
            probs = torch.sigmoid(outputs['logits'])
            preds = (probs > 0.5).float()
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        return all_labels, all_preds, all_probs
    
    def compute_metrics(self, labels, preds, probs):
        """Compute comprehensive metrics."""
        print("\n" + "="*80)
        print("Computing Metrics")
        print("="*80)
        
        metrics = {}
        
        for i, name in enumerate(self.pathology_names):
            class_labels = labels[:, i]
            class_probs = probs[:, i]
            
            if len(np.unique(class_labels)) < 2:
                continue
            
            auc = roc_auc_score(class_labels, class_probs)
            
            class_preds = preds[:, i]
            tp = np.sum((class_labels == 1) & (class_preds == 1))
            fp = np.sum((class_labels == 0) & (class_preds == 1))
            tn = np.sum((class_labels == 0) & (class_preds == 0))
            fn = np.sum((class_labels == 1) & (class_preds == 0))
            
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[name] = {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(np.sum(class_labels))
            }
        
        valid_aucs = [m['auc'] for m in metrics.values()]
        metrics['OVERALL'] = {
            'mean_auc': float(np.mean(valid_aucs)),
            'median_auc': float(np.median(valid_aucs)),
            'std_auc': float(np.std(valid_aucs))
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print metrics table."""
        print("\n" + "="*80)
        print("Per-Class Performance")
        print("="*80)
        print(f"{'Pathology':<30} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6}")
        print("-"*80)
        
        for name in self.pathology_names:
            if name in metrics:
                m = metrics[name]
                print(f"{name:<30} {m['auc']:>8.4f} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
                      f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['support']:>6}")
        
        print("-"*80)
        print(f"{'MEAN AUC':<30} {metrics['OVERALL']['mean_auc']:>8.4f}")
        print("="*80)
    
    def plot_roc_curves(self, labels, probs, save_path):
        """Plot ROC curves."""
        fig, axes = plt.subplots(4, 4, figsize=(16, 14))
        axes = axes.flatten()
        
        for i, name in enumerate(self.pathology_names):
            ax = axes[i]
            class_labels = labels[:, i]
            class_probs = probs[:, i]
            
            if len(np.unique(class_labels)) > 1:
                fpr, tpr, _ = roc_curve(class_labels, class_probs)
                auc = roc_auc_score(class_labels, class_probs)
                
                ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.set_title(name, fontsize=10)
                ax.legend(loc='lower right', fontsize=8)
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                ax.set_title(name, fontsize=10)
        
        for i in range(len(self.pathology_names), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ ROC curves saved to {save_path}")
        plt.close()
    
    def save_results(self, metrics, save_dir):
        """Save results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        df_metrics = pd.DataFrame.from_dict(
            {k: v for k, v in metrics.items() if k != 'OVERALL'},
            orient='index'
        )
        df_metrics.to_csv(save_dir / 'metrics.csv')
        
        print(f"✅ Results saved to {save_dir}")


def main():
    MODEL_PATH = "outputs/checkpoints/best_model.pth"
    VAL_CSV = "data/processed/valid_with_notes.csv"
    DATA_ROOT = "data/chexpert"
    BATCH_SIZE = 16
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    OUTPUT_DIR = "outputs/evaluation"
    
    print(f"{'='*80}")
    print("Model Evaluation")
    print(f"{'='*80}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    print("\nLoading validation dataset...")
    val_dataset = CheXpertMultimodalDataset(csv_path=VAL_CSV, data_root=DATA_ROOT, split="valid")
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"✅ Loaded {len(val_dataset)} samples")
    
    evaluator = ModelEvaluator(MODEL_PATH, device=DEVICE)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    labels, preds, probs = evaluator.evaluate(val_loader)
    metrics = evaluator.compute_metrics(labels, preds, probs)
    evaluator.print_metrics(metrics)
    evaluator.plot_roc_curves(labels, probs, f"{OUTPUT_DIR}/roc_curves.png")
    evaluator.save_results(metrics, OUTPUT_DIR)
    
    print(f"\n✅ Evaluation Complete! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()