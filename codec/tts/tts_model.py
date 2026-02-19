"""
LlamaTTS: Llama-based TTS model with speaker conditioning and multi-codebook support.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .sub_transformer import SubTransformer, SubTransformerConfig


@dataclass
class LlamaTTSOutput(CausalLMOutputWithPast):
    """
    Extended output with sub-transformer loss.
    
    Attributes:
        main_loss: Loss from main model (codebook 0 prediction)
        sub_loss: Loss from sub-transformer (codebooks 1-N prediction)
        total_loss: Weighted sum of main_loss and sub_loss
    """
    main_loss: Optional[torch.FloatTensor] = None
    sub_loss: Optional[torch.FloatTensor] = None
    total_loss: Optional[torch.FloatTensor] = None


class LlamaTTSConfig(PretrainedConfig):
    """Configuration for LlamaTTS model."""
    
    model_type = "llama_tts"
    
    def __init__(
        self,
        # Base Llama config
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        
        # Codec config
        num_codebooks: int = 8,
        codebook_vocab_size: int = 1024,
        
        # Speaker embedding config
        speaker_embedding_dim: int = 256,
        freeze_speaker_projection: bool = True,
        
        # Sub-transformer config
        use_sub_transformer: bool = True,
        sub_transformer_config: Optional[dict] = None,
        sub_transformer_loss_weight: float = 0.3,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_codebooks = num_codebooks
        self.codebook_vocab_size = codebook_vocab_size
        self.speaker_embedding_dim = speaker_embedding_dim
        self.freeze_speaker_projection = freeze_speaker_projection
        self.use_sub_transformer = use_sub_transformer
        self.sub_transformer_loss_weight = sub_transformer_loss_weight
        
        # Create sub-transformer config
        if use_sub_transformer and num_codebooks > 1:
            if sub_transformer_config is None:
                sub_transformer_config = {}
            
            self.sub_transformer_config = SubTransformerConfig(
                num_codebooks=num_codebooks,
                codebook_vocab_size=codebook_vocab_size,
                **sub_transformer_config
            )
        else:
            self.sub_transformer_config = None


class LlamaTTSForCausalLM(LlamaForCausalLM):
    """
    Llama-based TTS model with:
    1. Speaker embedding injection (frozen projection)
    2. Extended vocabulary for speech tokens
    3. Optional sub-transformer for multi-codebook prediction
    """
    
    config_class = LlamaTTSConfig
    
    def __init__(self, config: LlamaTTSConfig):
        # Initialize base Llama model
        super().__init__(config)
        
        self.tts_config = config
        
        # Speaker embedding projection (frozen by default)
        self.speaker_projection = nn.Linear(
            config.speaker_embedding_dim,
            config.hidden_size,
            bias=False
        )
        
        if config.freeze_speaker_projection:
            self.speaker_projection.requires_grad_(False)
        
        # Sub-transformer for multi-codebook prediction
        if config.use_sub_transformer and config.num_codebooks > 1:
            self.sub_transformer = SubTransformer(
                config.sub_transformer_config,
                base_model_config=config
            )
        else:
            self.sub_transformer = None
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # TTS-specific inputs
        speaker_embedding: Optional[torch.FloatTensor] = None,
        codebook_labels: Optional[torch.LongTensor] = None,
        speech_mask: Optional[torch.BoolTensor] = None,
        **kwargs
    ) -> Union[Tuple, LlamaTTSOutput]:
        """
        Args:
            speaker_embedding: (batch_size, speaker_embedding_dim) speaker embeddings
            codebook_labels: (batch_size, seq_len, num_codebooks-1) labels for codebooks 1..N-1
            speech_mask: (batch_size, seq_len) boolean mask for speech token positions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Inject speaker embedding at position 0 (replacing <|speaker|> token)
        if speaker_embedding is not None:
            batch_size = inputs_embeds.shape[0]
            speaker_emb = self.speaker_projection(speaker_embedding)  # (B, hidden_size)
            
            # Replace first token embedding with speaker embedding
            inputs_embeds[:, 0, :] = speaker_emb
        
        # Forward through main Llama model
        # Force output_hidden_states=True if we need sub-transformer
        if self.sub_transformer is not None and codebook_labels is not None:
            output_hidden_states = True
        
        outputs = super().forward(
            input_ids=None,  # We're using inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        main_loss = outputs.loss
        sub_loss = None
        total_loss = main_loss
        
        # Sub-transformer forward pass for codebooks 1..N-1
        if self.sub_transformer is not None and codebook_labels is not None:
            # Get last hidden states from main model
            hidden_states = outputs.hidden_states[-1]  # (B, T, D)
            
            # Forward through sub-transformer
            _, sub_loss = self.sub_transformer(
                hidden_states=hidden_states,
                codebook_labels=codebook_labels,
                speech_mask=speech_mask,
                attention_mask=attention_mask,
            )
            
            # Combine losses
            if main_loss is not None and sub_loss is not None:
                total_loss = main_loss + self.tts_config.sub_transformer_loss_weight * sub_loss
        
        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return LlamaTTSOutput(
            loss=total_loss,
            main_loss=main_loss,
            sub_loss=sub_loss,
            total_loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        speaker_embedding=None,
        **kwargs
    ):
        # Get base inputs
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
        # Add speaker embedding for first forward pass
        if speaker_embedding is not None and past_key_values is None:
            model_inputs["speaker_embedding"] = speaker_embedding
        
        return model_inputs