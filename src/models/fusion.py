
"""
Fusion module to combine image and text embeddings using cross-attention.
"""

import torch
import torch.nn as nn
import math


class CrossAttentionFusion(nn.Module):
    """Cross-attention between image and text embeddings."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: Embedding dimension (768 for ViT-base + BERT-base)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention for image attending to text
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head attention for text attending to image
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.ln_img = nn.LayerNorm(embed_dim)
        self.ln_text = nn.LayerNorm(embed_dim)
        
        # FFN for image
        self.ffn_img = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # FFN for text
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_embed: [batch, embed_dim] from image encoder
            text_embed: [batch, embed_dim] from text encoder
        
        Returns:
            fused_embed: [batch, embed_dim] multimodal embedding
        """
        batch_size = image_embed.size(0)
        
        # Add sequence dimension for attention
        # [batch, embed_dim] → [batch, 1, embed_dim]
        img = image_embed.unsqueeze(1)
        txt = text_embed.unsqueeze(1)
        
        # Image attends to text (query=img, key=txt, value=txt)
        img_attended, _ = self.img_to_text_attn(
            query=img,
            key=txt,
            value=txt
        )
        img_attended = img_attended.squeeze(1)  # [batch, embed_dim]
        
        # Residual + LayerNorm
        img = self.ln_img(image_embed + img_attended)
        
        # FFN
        img = img + self.ffn_img(img)
        
        # Text attends to image (query=txt, key=img, value=img)
        txt_attended, _ = self.text_to_img_attn(
            query=txt,
            key=img.unsqueeze(1),
            value=img.unsqueeze(1)
        )
        txt_attended = txt_attended.squeeze(1)  # [batch, embed_dim]
        
        # Residual + LayerNorm
        txt = self.ln_text(text_embed + txt_attended)
        
        # FFN
        txt = txt + self.ffn_text(txt)
        
        # Concatenate and project
        fused = torch.cat([img, txt], dim=-1)  # [batch, embed_dim * 2]
        fused = self.final_proj(fused)  # [batch, embed_dim]
        
        return fused


# Test script
if __name__ == "__main__":
    print("Testing CrossAttentionFusion...")
    
    fusion = CrossAttentionFusion(
        embed_dim=768,
        num_heads=8,
        dropout=0.1
    )
    
    # Create dummy embeddings
    batch_size = 4
    image_embed = torch.randn(batch_size, 768)
    text_embed = torch.randn(batch_size, 768)
    
    print(f"Image embedding shape: {image_embed.shape}")
    print(f"Text embedding shape: {text_embed.shape}")
    
    # Forward pass
    with torch.no_grad():
        fused = fusion(image_embed, text_embed)
    
    print(f"\nFused embedding shape: {fused.shape}")
    print(f"Expected: [{batch_size}, 768]")
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\nFusion module parameters: {total_params:,}")