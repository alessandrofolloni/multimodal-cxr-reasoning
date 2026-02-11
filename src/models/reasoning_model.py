"""
Complete multimodal reasoning model for chest X-ray diagnosis.
Combines image encoder, text encoder, fusion, and classification head.
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion import CrossAttentionFusion


class MultimodalReasoningModel(nn.Module):
    """
    Full pipeline: Image + Text → Fusion → Pathology Classification
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        image_model: str = "google/vit-base-patch16-224",
        text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 768,
        fusion_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoders: bool = False
    ):
        """
        Args:
            num_classes: Number of pathology classes (14 for CheXpert)
            image_model: HuggingFace model for images
            text_model: HuggingFace model for text
            embed_dim: Embedding dimension
            fusion_heads: Number of attention heads in fusion
            dropout: Dropout rate
            freeze_encoders: Freeze image/text encoders (for fast prototyping)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoders
        self.image_encoder = ImageEncoder(
            model_name=image_model,
            pretrained=True,
            freeze_backbone=freeze_encoders,
            output_dim=embed_dim
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_model,
            max_length=128,
            freeze_backbone=freeze_encoders,
            output_dim=embed_dim
        )
        
        # Fusion
        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=fusion_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        print(f"✅ MultimodalReasoningModel initialized")
        print(f"   - Image encoder: {image_model}")
        print(f"   - Text encoder: {text_model}")
        print(f"   - Num classes: {num_classes}")
        print(f"   - Encoders frozen: {freeze_encoders}")
    
    def forward(
        self,
        images: torch.Tensor,
        clinical_notes: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [batch, 3, 224, 224]
            clinical_notes: List of strings [batch]
        
        Returns:
            Dict with:
                - logits: [batch, num_classes] pathology predictions
                - image_embed: [batch, embed_dim] 
                - text_embed: [batch, embed_dim]
                - fused_embed: [batch, embed_dim]
        """
        # Encode image
        image_embed = self.image_encoder(images)  # [batch, 768]
        
        # Encode text
        text_embed = self.text_encoder(clinical_notes)  # [batch, 768]
        
        # Fuse modalities
        fused_embed = self.fusion(image_embed, text_embed)  # [batch, 768]
        
        # Classify
        logits = self.classifier(fused_embed)  # [batch, num_classes]
        
        return {
            'logits': logits,
            'image_embed': image_embed,
            'text_embed': text_embed,
            'fused_embed': fused_embed
        }
    
    def get_num_params(self) -> Dict[str, int]:
        """Count parameters by component."""
        return {
            'image_encoder': sum(p.numel() for p in self.image_encoder.parameters()),
            'text_encoder': sum(p.numel() for p in self.text_encoder.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


# Test script
if __name__ == "__main__":
    print("="*80)
    print("Testing MultimodalReasoningModel")
    print("="*80)
    
    # Initialize model
    model = MultimodalReasoningModel(
        num_classes=14,
        freeze_encoders=False  # Set True for faster testing
    )
    
    # Create dummy batch
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_notes = [
        "Chief complaint: Patient presenting with dyspnea for 3 days. History: heart failure.",
        "Chief complaint: Routine chest evaluation. History: unremarkable medical history."
    ]
    
    print(f"\nInput:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Notes: {len(dummy_notes)} strings")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_images, dummy_notes)
    
    print(f"\nOutput:")
    print(f"  Logits: {outputs['logits'].shape} (pathology predictions)")
    print(f"  Image embed: {outputs['image_embed'].shape}")
    print(f"  Text embed: {outputs['text_embed'].shape}")
    print(f"  Fused embed: {outputs['fused_embed'].shape}")
    
    # Sample predictions
    probs = torch.sigmoid(outputs['logits'])
    print(f"\nSample predictions (first sample):")
    pathologies = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    for i, (path, prob) in enumerate(zip(pathologies, probs[0])):
        if prob > 0.5:
            print(f"  {path}: {prob:.3f}")
    
    # Model statistics
    print(f"\n{'='*80}")
    print("Model Parameters:")
    print(f"{'='*80}")
    params = model.get_num_params()
    for name, count in params.items():
        print(f"{name:20s}: {count:,}")
    
    print(f"\n✅ Model test completed successfully!")