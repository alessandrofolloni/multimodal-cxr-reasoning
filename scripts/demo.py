"""
Interactive demo for multimodal reasoning model.
Upload a chest X-ray image and optional clinical note to see predictions.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from transformers import ViTModel, AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set paths
project_root = Path(__file__).parent.parent


# =============================================================================
# MODEL COMPONENTS (same as evaluation)
# =============================================================================

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
# DEMO CLASS
# =============================================================================

class CXRDemo:
    """Interactive demo for chest X-ray reasoning."""
    
    def __init__(self, model_path, device="mps"):
        self.device = device
        self.pathology_names = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        print("Loading model...")
        self.model = MultimodalReasoningModel(num_classes=14, freeze_encoders=False)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"✅ Model loaded from {model_path}")
    
    @torch.no_grad()
    def predict(self, image_path, clinical_note=None):
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to chest X-ray image
            clinical_note: Optional clinical context (str)
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Use default note if none provided
        if clinical_note is None:
            clinical_note = "Chief complaint: Routine chest evaluation. History: unremarkable medical history."
        
        # Predict
        outputs = self.model(image_tensor, [clinical_note])
        logits = outputs['logits']
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create results
        results = []
        for i, name in enumerate(self.pathology_names):
            results.append({
                'pathology': name,
                'probability': float(probs[i]),
                'prediction': 'Positive' if probs[i] > 0.5 else 'Negative'
            })
        
        # Sort by probability
        results = sorted(results, key=lambda x: x['probability'], reverse=True)
        
        return {
            'image': image,
            'clinical_note': clinical_note,
            'predictions': results
        }
    
    def visualize_predictions(self, results, save_path=None):
        """Visualize image with predictions."""
        fig = plt.figure(figsize=(16, 6))
        
        # Image
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(results['image'], cmap='gray')
        ax1.set_title('Chest X-Ray', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Clinical Note
        ax2 = plt.subplot(1, 3, 2)
        ax2.text(0.05, 0.95, 'Clinical Context:', 
                fontsize=12, fontweight='bold', va='top')
        ax2.text(0.05, 0.85, results['clinical_note'], 
                fontsize=10, va='top', wrap=True)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Top Predictions
        ax3 = plt.subplot(1, 3, 3)
        
        # Get top 10 predictions
        top_preds = results['predictions'][:10]
        
        y_pos = np.arange(len(top_preds))
        probs = [p['probability'] for p in top_preds]
        names = [p['pathology'] for p in top_preds]
        colors = ['#d62728' if p > 0.5 else '#1f77b4' for p in probs]
        
        bars = ax3.barh(y_pos, probs, color=colors, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(names, fontsize=9)
        ax3.set_xlabel('Probability', fontsize=11)
        ax3.set_title('Model Predictions', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
        
        # Add probability values
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax3.text(prob + 0.02, i, f'{prob:.3f}', 
                    va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
    
    def print_predictions(self, results):
        """Print predictions in a nice format."""
        print("\n" + "="*80)
        print("PREDICTIONS")
        print("="*80)
        print(f"\nClinical Note:\n{results['clinical_note']}")
        print("\n" + "-"*80)
        print(f"{'Pathology':<30} {'Probability':>12} {'Prediction':>12}")
        print("-"*80)
        
        for pred in results['predictions'][:10]:
            prob = pred['probability']
            color = '\033[91m' if prob > 0.5 else '\033[94m'  # Red if positive, blue if negative
            reset = '\033[0m'
            print(f"{pred['pathology']:<30} {color}{prob:>12.4f}{reset} {pred['prediction']:>12}")
        
        print("="*80)


# =============================================================================
# INTERACTIVE DEMO
# =============================================================================

def interactive_demo():
    """Run interactive demo."""
    MODEL_PATH = "outputs/checkpoints/best_model.pth"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("="*80)
    print("MULTIMODAL CHEST X-RAY REASONING - INTERACTIVE DEMO")
    print("="*80)
    
    # Initialize demo
    demo = CXRDemo(MODEL_PATH, device=DEVICE)
    
    while True:
        print("\n" + "="*80)
        print("Options:")
        print("  1. Test on validation sample")
        print("  2. Upload your own image")
        print("  3. Exit")
        print("="*80)
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Test on random validation sample
            val_csv = "data/processed/valid_with_notes.csv"
            df = pd.read_csv(val_csv)
            
            idx = np.random.randint(0, len(df))
            row = df.iloc[idx]
            
            image_path = Path("data/chexpert") / row['Path']
            clinical_note = row['clinical_note']
            
            print(f"\n📸 Testing on validation sample {idx}")
            print(f"   Image: {image_path}")
            
            # Predict
            results = demo.predict(image_path, clinical_note)
            
            # Print results
            demo.print_predictions(results)
            
            # Visualize
            save_path = f"outputs/demo/sample_{idx}_prediction.png"
            Path("outputs/demo").mkdir(parents=True, exist_ok=True)
            demo.visualize_predictions(results, save_path=save_path)
        
        elif choice == "2":
            # Upload custom image
            image_path = input("\nEnter path to chest X-ray image: ").strip()
            
            if not Path(image_path).exists():
                print(f"❌ File not found: {image_path}")
                continue
            
            # Optional clinical note
            use_note = input("Provide clinical context? (y/n): ").strip().lower()
            
            if use_note == 'y':
                print("\nEnter clinical note (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                clinical_note = " ".join(lines)
            else:
                clinical_note = None
            
            # Predict
            print(f"\n📸 Analyzing image: {image_path}")
            results = demo.predict(image_path, clinical_note)
            
            # Print and visualize
            demo.print_predictions(results)
            
            save_path = f"outputs/demo/custom_prediction.png"
            Path("outputs/demo").mkdir(parents=True, exist_ok=True)
            demo.visualize_predictions(results, save_path=save_path)
        
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def batch_demo():
    """Test on multiple validation samples at once."""
    MODEL_PATH = "outputs/checkpoints/best_model.pth"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("="*80)
    print("BATCH DEMO - Test on 5 random validation samples")
    print("="*80)
    
    demo = CXRDemo(MODEL_PATH, device=DEVICE)
    
    # Load validation data
    val_csv = "data/processed/valid_with_notes.csv"
    df = pd.read_csv(val_csv)
    
    # Random samples
    indices = np.random.choice(len(df), size=5, replace=False)
    
    Path("outputs/demo").mkdir(parents=True, exist_ok=True)
    
    for i, idx in enumerate(indices):
        row = df.iloc[idx]
        image_path = Path("data/chexpert") / row['Path']
        clinical_note = row['clinical_note']
        
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/5 (Index: {idx})")
        print(f"{'='*80}")
        
        results = demo.predict(image_path, clinical_note)
        demo.print_predictions(results)
        
        save_path = f"outputs/demo/batch_sample_{i+1}.png"
        demo.visualize_predictions(results, save_path=save_path)
    
    print(f"\n✅ Batch demo complete! Check outputs/demo/ for visualizations.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo for multimodal CXR reasoning")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive",
                       help="Demo mode: interactive or batch")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_demo()
    else:
        batch_demo()


if __name__ == "__main__":
    main()