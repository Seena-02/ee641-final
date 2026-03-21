"""
ResNet + GPT2 Prefix-Tuning Model - WITH ORIGINAL LOSS COMPUTATION
Vision encoder (ResNet18) -> Prefix Generator -> LLM Decoder (GPT2)
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
from torchvision import models


class ResNetGPT2PrefixModel(nn.Module):
    """
    Vision-to-Language model using prefix-tuning.
    
    Architecture:
    1. ResNet18: Extract visual features (512-dim)
    2. Projection: 512-dim -> hidden_size
    3. Prefix Generator: hidden_size -> (num_prefix_tokens × hidden_size)
    4. GPT2 Decoder: Generate sequence from prefix + tokens
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size=None,
        gpt2_hidden_size=None,  # Backward compatibility
        num_layers=2,
        num_attention_heads=2,
        num_prefix_tokens=8,
        dropout=0.4,
        resnet_frozen=True
    ):
        super().__init__()
        
        # Handle backward compatibility: accept either parameter name
        if gpt2_hidden_size is not None:
            hidden_size = gpt2_hidden_size
        elif hidden_size is None:
            hidden_size = 128  # Default
        
        self.hidden_size = hidden_size
        self.num_prefix_tokens = num_prefix_tokens
        self.vocab_size = vocab_size
        
        # Vision Encoder: ResNet18 (pretrained on ImageNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet if specified
        if resnet_frozen:
            for param in self.resnet_features.parameters():
                param.requires_grad = False
        
        # Project ResNet features (512-dim) to hidden size
        self.feature_projection = nn.Linear(512, hidden_size)
        
        # Prefix Generator: Maps image features to prefix embeddings
        self.prefix_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, num_prefix_tokens * hidden_size)
        )
        
        # GPT2 Decoder Configuration
        gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=128,  # Match original training
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_attention_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        
        self.gpt2 = GPT2LMHeadModel(gpt2_config)
    
    def encode_image_to_prefix(self, images):
        """Convert images to prefix embeddings"""
        batch_size = images.shape[0]
        features = self.resnet_features(images).squeeze(-1).squeeze(-1)
        features = self.feature_projection(features)
        prefix_flat = self.prefix_generator(features)
        prefix_embeds = prefix_flat.view(batch_size, self.num_prefix_tokens, self.hidden_size)
        return prefix_embeds
    
    def forward(self, images, input_ids, labels=None):
        """
        Forward pass with ORIGINAL loss computation method.
        
        Args:
            images: (batch, 3, 224, 224) - Input images
            input_ids: (batch, seq_len) - Token IDs
            labels: (batch, seq_len) - Target labels (with -100 for padding)
        
        Returns:
            outputs with loss and logits
        """
        prefix_embeds = self.encode_image_to_prefix(images)
        token_embeds = self.gpt2.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        if labels is not None:
            # Get outputs WITHOUT labels first
            outputs = self.gpt2(inputs_embeds=inputs_embeds, return_dict=True)
            
            # Extract logits for NON-PREFIX positions only
            # outputs.logits shape: [batch, prefix_len + seq_len, vocab]
            # We want logits starting from position num_prefix_tokens
            logits = outputs.logits[:, self.num_prefix_tokens:, :]  # [batch, seq_len, vocab]
            
            # Compute loss manually on these logits
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            
            # Add loss to outputs
            outputs.loss = loss
            return outputs
        else:
            return self.gpt2(inputs_embeds=inputs_embeds, return_dict=True)
    
    def generate(self, images, max_length=20, num_beams=1, 
                 pad_token_id=0, eos_token_id=2, bos_token_id=1):
        """
        Generate sequences from images using autoregressive decoding.
        FIXED: Properly appends EOS token instead of replacing it with PAD.
        
        Args:
            images: (batch, 3, 224, 224) - Input images
            max_length: Maximum sequence length
            num_beams: Number of beams (unused, for compatibility)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            bos_token_id: Beginning-of-sequence token ID
        
        Returns:
            generated_ids: (batch, seq_len) - Generated token IDs
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Encode images to prefix
        prefix_embeds = self.encode_image_to_prefix(images)
        
        # Start with BOS token
        generated_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Autoregressive generation
        for step in range(max_length - 1):
            # Get token embeddings for generated sequence so far
            token_embeds = self.gpt2.transformer.wte(generated_ids)
            
            # Combine prefix + token embeddings
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            
            # Forward pass
            outputs = self.gpt2(inputs_embeds=inputs_embeds, return_dict=True)
            
            # Get logits for the last TOKEN position (not last overall position)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)
            
            # FIXED: Append token first (including EOS if generated)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            
            # Mark finished sequences (after appending the token)
            finished = finished | (next_token == eos_token_id)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated_ids