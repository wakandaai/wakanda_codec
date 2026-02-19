"""
Sub-transformer for predicting codebooks 1 to N-1.
Based on Qwen3-TTS architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import PretrainedConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
)


class DepthTransformerConfig(PretrainedConfig):
    """Configuration for the sub-transformer."""
    
    model_type = "sub_transformer"
    
    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_vocab_size: int = 1024,
        hidden_size: int = 2048,
        num_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        intermediate_size: int = 5632,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_codebooks = num_codebooks
        self.codebook_vocab_size = codebook_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout


class DepthTransformer(nn.Module):
    """
    Sub-transformer that predicts codebooks 1 to N-1 sequentially.
    
    For each codebook i (from 1 to N-1):
        - Input: hidden_states from main model + embeddings of codebooks 0 to i-1
        - Output: logits for codebook i
    
    This follows the Qwen3-TTS design where RVQ codebooks are predicted
    autoregressively within each timestep.
    """
    
    def __init__(self, config: DepthTransformerConfig, base_model_config):
        super().__init__()
        self.config = config
        self.num_codebooks = config.num_codebooks
        
        # Create a pseudo-config for Llama layers
        self.layer_config = type('Config', (), {
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': config.num_key_value_heads,
            'intermediate_size': config.intermediate_size,
            'hidden_act': config.hidden_act,
            'max_position_embeddings': config.max_position_embeddings,
            'rms_norm_eps': config.rms_norm_eps,
            'rope_theta': config.rope_theta,
            'attention_dropout': config.attention_dropout,
            'rope_scaling': None,
            'attention_bias': False,
            '_attn_implementation': base_model_config._attn_implementation,
        })()
        
        # Separate embedding for each codebook 1 to N-1
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(config.codebook_vocab_size, config.hidden_size)
            for _ in range(config.num_codebooks - 1)
        ])
        
        # Projection if main model hidden size differs
        if base_model_config.hidden_size != config.hidden_size:
            self.input_projection = nn.Linear(
                base_model_config.hidden_size,
                config.hidden_size,
                bias=True
            )
        else:
            self.input_projection = nn.Identity()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(self.layer_config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Separate prediction head for each codebook
        self.lm_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.codebook_vocab_size, bias=False)
            for _ in range(config.num_codebooks - 1)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = self.config.hidden_size ** -0.5
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        codebook_labels: Optional[torch.Tensor] = None,
        speech_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) from main model's last layer
            codebook_labels: (batch_size, seq_len, num_codebooks-1) ground truth for codebooks 1..N-1
            speech_mask: (batch_size, seq_len) boolean mask indicating speech token positions
            attention_mask: (batch_size, seq_len) attention mask
        
        Returns:
            all_logits: (batch_size, seq_len, num_codebooks-1, vocab_size)
            loss: scalar tensor if codebook_labels provided, else None
        """
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project input if needed
        hidden_states = self.input_projection(hidden_states)
        
        # Store logits for all codebooks
        all_logits = []
        total_loss = 0.0
        
        # Sequentially predict each codebook
        for codebook_idx in range(self.num_codebooks - 1):
            # Build input: main hidden + sum of previous codebook embeddings
            layer_input = hidden_states.clone()
            
            if codebook_idx > 0 and codebook_labels is not None:
                # Add embeddings from previously predicted codebooks
                for prev_idx in range(codebook_idx):
                    prev_emb = self.codebook_embeddings[prev_idx](
                        codebook_labels[:, :, prev_idx]
                    )
                    layer_input = layer_input + prev_emb
            
            # Forward through transformer layers
            current_hidden = layer_input
            for layer in self.layers:
                layer_outputs = layer(
                    current_hidden,
                    attention_mask=attention_mask,
                    position_ids=None,
                )
                current_hidden = layer_outputs[0]
            
            current_hidden = self.norm(current_hidden)
            
            # Predict current codebook
            logits = self.lm_heads[codebook_idx](current_hidden)
            all_logits.append(logits)
            
            # Compute loss if labels provided
            if codebook_labels is not None and speech_mask is not None:
                # Only compute loss on speech positions
                valid_logits = logits[speech_mask]  # (num_speech_tokens, vocab_size)
                valid_labels = codebook_labels[:, :, codebook_idx][speech_mask]  # (num_speech_tokens,)
                
                loss = nn.functional.cross_entropy(
                    valid_logits,
                    valid_labels,
                    ignore_index=-100
                )
                total_loss = total_loss + loss
        
        # Stack logits: (B, T, N-1, V)
        all_logits = torch.stack(all_logits, dim=2)
        
        if codebook_labels is not None:
            # Average loss across codebooks
            avg_loss = total_loss / (self.num_codebooks - 1)
            return all_logits, avg_loss
        
        return all_logits, None
    
    def get_input_embeddings(self):
        """Return the first codebook embedding for compatibility."""
        return self.codebook_embeddings[0]
    
    def get_output_embeddings(self):
        """Return the first lm_head for compatibility."""
        return self.lm_heads[0]