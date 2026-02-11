"""
Text encoder using BioClinicalBERT for clinical notes.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """BERT-based encoder for clinical text."""
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 128,
        freeze_backbone: bool = False,
        output_dim: int = 768
    ):
        """
        Args:
            model_name: HuggingFace model name (BioClinicalBERT recommended)
            max_length: Max token length for input
            freeze_backbone: Freeze BERT weights
            output_dim: Output embedding dimension
        """
        super().__init__()
        
        self.max_length = max_length
        self.output_dim = output_dim
        
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        print(f"✅ Loaded clinical BERT: {model_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("❄️  BERT backbone frozen")
        
        # Projection head
        self.hidden_dim = self.bert.config.hidden_size  # 768 for BERT-base
        
        if output_dim != self.hidden_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: List of clinical notes (strings)
        
        Returns:
            embeddings: [batch, output_dim]
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        input_ids = encoded['input_ids'].to(next(self.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Project
        embedding = self.projection(cls_embedding)  # [batch, output_dim]
        
        return embedding


# Test script
if __name__ == "__main__":
    print("Testing TextEncoder...")
    
    encoder = TextEncoder(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=128,
        freeze_backbone=False,
        output_dim=768
    )
    
    # Test clinical notes
    test_notes = [
        "Chief complaint: Patient presenting with dyspnea for 3 days. History: heart failure.",
        "Chief complaint: Routine chest evaluation. History: unremarkable medical history.",
        "Chief complaint: Patient presenting with fever and cough. History: COPD.",
    ]
    
    # Forward pass
    with torch.no_grad():
        embeddings = encoder(test_notes)
    
    print(f"\nInput: {len(test_notes)} clinical notes")
    print(f"Output shape: {embeddings.shape}")
    print(f"Expected: [{len(test_notes)}, 768]")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")