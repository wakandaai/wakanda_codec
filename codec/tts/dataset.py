"""
TTS Dataset for loading pre-extracted codes, speaker embeddings, and text.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TTSDataset(Dataset):
    """
    Dataset for TTS training with pre-extracted codes and speaker embeddings.
    
    CSV format:
        path_to_codes, path_to_spk_embed, text
    
    Code file format:
        NumPy array of shape (num_codebooks, time)
    
    Speaker embedding format:
        NumPy array of shape (1, speaker_embedding_dim)
    """
    
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        config: Dict,
        max_seq_length: int = 4096,
    ):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = max_seq_length
        
        # Get special token IDs
        self.speaker_token_id = tokenizer.convert_tokens_to_ids('<|speaker|>')
        self.text_start_id = tokenizer.convert_tokens_to_ids('<|text_start|>')
        self.text_end_id = tokenizer.convert_tokens_to_ids('<|text_end|>')
        self.speech_start_id = tokenizer.convert_tokens_to_ids('<|speech_start|>')
        self.speech_end_id = tokenizer.convert_tokens_to_ids('<|speech_end|>')
        self.pad_token_id = tokenizer.pad_token_id
        
        # Base vocab size (for shifting codebook tokens)
        # This is original vocab + special tokens, before codebook tokens
        self.base_vocab_size = len(tokenizer) - config['codebook_vocab_size']
        
        self.num_codebooks = config['num_codebooks']
        self.codebook_vocab_size = config['codebook_vocab_size']
        
        logger.info(f"Loaded {len(self.df)} samples from {csv_path}")
        logger.info(f"Base vocab size: {self.base_vocab_size}")
        logger.info(f"Codebook vocab size: {self.codebook_vocab_size}")
        logger.info(f"Num codebooks: {self.num_codebooks}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        try:
            # Load speaker embedding
            speaker_emb = np.load(row['path_to_spk_embed'])  # (1, speaker_embedding_dim)
            speaker_emb = torch.tensor(speaker_emb, dtype=torch.float32).squeeze()  # (speaker_embedding_dim,)
            
            # Load codes
            codes = np.load(row['path_to_codes'])  # (num_codebooks, time)
            
            # Validate shape
            if codes.ndim != 2:
                raise ValueError(f"Expected 2D codes, got shape {codes.shape}")
            
            num_codebooks, time_steps = codes.shape
            
            # Handle single vs multi-codebook
            if num_codebooks != self.num_codebooks:
                raise ValueError(
                    f"Expected {self.num_codebooks} codebooks, got {num_codebooks} in sample {idx}"
                )
            
            # Get text
            text = row['text']
            
            # Tokenize text (without special tokens, we'll add them manually)
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Build sequence:
            # [<|speaker|>] [<|text_start|>] text_tokens [<|text_end|>] 
            # [<|speech_start|>] codebook_0_tokens [<|speech_end|>]
            
            codebook_0 = codes[0, :]  # First codebook
            codebook_0_shifted = codebook_0 + self.base_vocab_size
            
            # Check for vocab overflow
            if codebook_0_shifted.max() >= len(self.tokenizer):
                raise ValueError(
                    f"Codebook token overflow: max shifted ID {codebook_0_shifted.max()} "
                    f">= vocab size {len(self.tokenizer)}"
                )
            
            input_ids = [
                self.speaker_token_id,
                self.text_start_id,
                *text_tokens,
                self.text_end_id,
                self.speech_start_id,
                *codebook_0_shifted.tolist(),
                self.speech_end_id
            ]
            
            # Truncate if too long
            if len(input_ids) > self.max_seq_length:
                # Keep speaker + text markers, truncate speech
                text_len = len(text_tokens) + 4  # speaker + text_start + text + text_end
                max_speech_len = self.max_seq_length - text_len - 2  # -2 for speech_start/end
                
                if max_speech_len < 1:
                    # Text itself too long, truncate text
                    logger.warning(f"Text too long, truncating. Sample idx={idx}")
                    max_text_len = self.max_seq_length - 6  # Minimal speech
                    text_tokens = text_tokens[:max_text_len]
                    max_speech_len = 1
                
                codebook_0_shifted = codebook_0_shifted[:max_speech_len]
                codes = codes[:, :max_speech_len]
                
                input_ids = [
                    self.speaker_token_id,
                    self.text_start_id,
                    *text_tokens,
                    self.text_end_id,
                    self.speech_start_id,
                    *codebook_0_shifted.tolist(),
                    self.speech_end_id
                ]
            
            # Create labels: -100 for non-speech tokens
            speech_start_pos = 4 + len(text_tokens)  # After speaker + text_start + text + text_end
            labels = [-100] * speech_start_pos
            labels += codebook_0_shifted.tolist()
            labels += [self.speech_end_id]
            
            # Create speech mask
            speech_mask = [False] * speech_start_pos
            speech_mask += [True] * len(codebook_0_shifted)
            speech_mask += [False]  # speech_end is not a speech token
            
            # Multi-codebook labels (codebooks 1 to N-1)
            if self.num_codebooks > 1:
                # Shape: (time, num_codebooks-1)
                codebook_1_to_n = codes[1:, :len(codebook_0_shifted)].T
                
                # Pad to match sequence length (with -100 for non-speech positions)
                padded_codebook_labels = np.full(
                    (len(input_ids), self.num_codebooks - 1),
                    -100,
                    dtype=np.int64
                )
                padded_codebook_labels[
                    speech_start_pos:speech_start_pos + len(codebook_0_shifted), :
                ] = codebook_1_to_n
                
                codebook_labels = torch.tensor(padded_codebook_labels, dtype=torch.long)
            else:
                codebook_labels = None
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'speaker_embedding': speaker_emb,
                'codebook_labels': codebook_labels,
                'speech_mask': torch.tensor(speech_mask, dtype=torch.bool),
            }
        
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            logger.error(f"Row: {row}")
            raise