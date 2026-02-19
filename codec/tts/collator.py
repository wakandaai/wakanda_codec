"""
Data collator for TTS training with padding.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer


@dataclass
class TTSDataCollator:
    """
    Data collator that pads sequences to the same length within a batch.
    """
    
    tokenizer: PreTrainedTokenizer
    num_codebooks: int
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get max length in this batch
        max_length = max(len(f['input_ids']) for f in features)
        
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        batch_size = len(features)
        
        # Initialize tensors
        input_ids = torch.full(
            (batch_size, max_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        speech_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        speaker_embeddings = []
        
        # Codebook labels (if multi-codebook)
        if self.num_codebooks > 1 and features[0]['codebook_labels'] is not None:
            codebook_labels = torch.full(
                (batch_size, max_length, self.num_codebooks - 1),
                -100,
                dtype=torch.long
            )
        else:
            codebook_labels = None
        
        # Fill tensors
        for i, feature in enumerate(features):
            seq_len = len(feature['input_ids'])
            
            input_ids[i, :seq_len] = feature['input_ids']
            labels[i, :seq_len] = feature['labels']
            attention_mask[i, :seq_len] = 1
            speech_mask[i, :seq_len] = feature['speech_mask']
            speaker_embeddings.append(feature['speaker_embedding'])
            
            if codebook_labels is not None:
                codebook_labels[i, :seq_len, :] = feature['codebook_labels']
        
        speaker_embeddings = torch.stack(speaker_embeddings)
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'speaker_embedding': speaker_embeddings,
            'speech_mask': speech_mask,
        }
        
        if codebook_labels is not None:
            batch['codebook_labels'] = codebook_labels
        
        return batch