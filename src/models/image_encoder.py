"""
Image encoder using Vision Transformer (ViT) for chest X-rays.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class ImageEncoder(nn.Module):
    """Vision Transformer encoder for chest X-ray images."""
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 768
    ):
        """
        Args:
            model_name: HuggingFace model name
            pretrained: Use pretrained ImageNet weights
            freeze_backbone: Freeze ViT weights (for quick experiments)
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Load ViT
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
            print(f"✅ Loaded pretrained ViT: {model_name}")
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
            print(f"✅ Initialized ViT from scratch")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print("❄️  ViT backbone frozen")
        
        # Projection head (if output_dim != 768)
        self.hidden_dim = self.vit.config.hidden_size  # 768 for base
        
        if output_dim != self.hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, 3, 224, 224]
        
        Returns:
            embeddings: [batch, output_dim]
        """
        # Get ViT outputs
        outputs = self.vit(pixel_values=images)
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Project to output_dim
        embedding = self.projection(cls_embedding)  # [batch, output_dim]
        
        return embedding


# Test script
if __name__ == "__main__":
    # Test image encoder
    print("Testing ImageEncoder...")
    
    encoder = ImageEncoder(
        model_name="google/vit-base-patch16-224",
        pretrained=True,
        freeze_backbone=False,
        output_dim=768
    )
    
    # Create dummy batch
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    with torch.no_grad():
        embeddings = encoder(dummy_images)
    
    print(f"\nInput shape: {dummy_images.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: [{batch_size}, 768]")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")