# codec/tts/dataset.py

"""
Dataset classes for Zero-shot TTS training.

Supports both single-codebook and multi-codebook audio codecs.
Loads precomputed codes and text on-the-fly.

Expected manifest CSV format:
    code_path,text,speaker_id (optional)
    /path/to/codes.npy,"Hello world",spk_001

Expected code format (.npy files):
    Single codebook: shape (T,) or (1, T)
    Multi-codebook: shape (K, T) where K is number of codebooks
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from codec.tts.config import DataConfig

logger = logging.getLogger(__name__)


class SpeechLMDataset(Dataset):
    """
    Dataset for Speech Language Model training.
    
    Handles:
    - Loading precomputed audio codes from .npy files
    - Tokenizing text with the LLM tokenizer
    - Combining text + speech tokens with proper special tokens
    - Creating labels with masking for text portion
    
    For multi-codebook codecs:
    - Returns all codebook codes for embedding summation (input)
    - Returns separate targets for temporal (codebook 1) and depth (codebooks 2..K) transformers
    """
    
    def __init__(
        self,
        manifest_path: str,
        tokenizer: AutoTokenizer,
        config: DataConfig,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            manifest_path: Path to CSV manifest file
            tokenizer: HuggingFace tokenizer (should already have special tokens added)
            config: DataConfig with dataset parameters
            split: "train" or "val" (affects logging)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Load manifest
        self.manifest = self._load_manifest(manifest_path)
        logger.info(f"Loaded {split} manifest with {len(self.manifest)} samples")
        
        # Cache special token IDs
        self._cache_special_tokens()
        
        # Compute vocab offset for speech tokens
        # Speech tokens start after text vocab + special tokens
        self.speech_token_offset = len(tokenizer) - (config.num_codebooks * config.codebook_size)
        logger.info(f"Speech token offset: {self.speech_token_offset}")
        
        # Ignore index for loss computation
        self.ignore_index = -100
        
    def _load_manifest(self, manifest_path: str) -> pd.DataFrame:
        """Load and validate manifest CSV."""
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        df = pd.read_csv(manifest_path)
        
        # Validate required columns
        required_cols = [self.config.code_path_col, self.config.text_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter out rows with missing code files
        valid_mask = df[self.config.code_path_col].apply(lambda x: Path(x).exists())
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            logger.warning(f"Skipping {n_invalid} samples with missing code files")
            df = df[valid_mask].reset_index(drop=True)
        
        return df
    
    def _cache_special_tokens(self):
        """Cache special token IDs for efficiency."""
        self.speech_start_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_START|>")
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_END|>")
        self.text_start_id = self.tokenizer.convert_tokens_to_ids("<|TEXT_START|>")
        self.text_end_id = self.tokenizer.convert_tokens_to_ids("<|TEXT_END|>")
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # Validate tokens were added
        for name, token_id in [
            ("SPEECH_START", self.speech_start_id),
            ("SPEECH_END", self.speech_end_id),
            ("TEXT_START", self.text_start_id),
            ("TEXT_END", self.text_end_id),
        ]:
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"Special token {name} not found in tokenizer")
    
    def __len__(self) -> int:
        return len(self.manifest)
    
    def _load_codes(self, code_path: str) -> np.ndarray:
        """
        Load audio codes from .npy file.
        
        Returns:
            np.ndarray of shape (K, T) where K is num_codebooks
        """
        codes = np.load(code_path)
        
        # Normalize shape to (K, T)
        if codes.ndim == 2:
            # Multi-codebook: (K, T) - already correct
            pass
        elif codes.ndim == 1:
            # Single codebook: (T,) -> (1, T)
            codes = codes[np.newaxis, :]
        else:
            raise ValueError(f"Unexpected code shape: {codes.shape}")
        
        # Validate codebook count
        if codes.shape[0] != self.config.num_codebooks:
            raise ValueError(
                f"Code file has {codes.shape[0]} codebooks, "
                f"expected {self.config.num_codebooks}"
            )
        
        return codes.astype(np.int64)
    
    def _codes_to_token_ids(self, codes: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert codec codes to token IDs in the expanded vocabulary.
        
        For single codebook:
            Returns (temporal_ids, None)
            temporal_ids shape: (T,)
            
        For multi-codebook:
            Returns (temporal_ids, depth_targets)
            temporal_ids shape: (T,) - first codebook only, for LLM input/output
            depth_targets shape: (T, K-1) - remaining codebooks for depth transformer
        
        Token ID computation:
            token_id = speech_token_offset + (codebook_index * codebook_size) + code_value
        """
        K, T = codes.shape
        
        # First codebook -> temporal transformer tokens
        temporal_ids = self.speech_token_offset + codes[0]  # Shape: (T,)
        
        if K == 1:
            return temporal_ids, None
        
        # Remaining codebooks -> depth transformer targets
        # Shape: (K-1, T) -> transpose to (T, K-1) for easier batching
        depth_targets = np.zeros((T, K - 1), dtype=np.int64)
        for k in range(1, K):
            # Each codebook has its own offset in the vocabulary
            depth_targets[:, k - 1] = self.speech_token_offset + (k * self.config.codebook_size) + codes[k]
        
        return temporal_ids, depth_targets
    
    def _create_input_embeddings_indices(self, codes: np.ndarray) -> np.ndarray:
        """
        For multi-codebook: create indices for summed embeddings.
        
        At each timestep, we sum embeddings from all K codebooks.
        Returns array of shape (T, K) with token IDs for each codebook.
        """
        K, T = codes.shape
        
        embedding_indices = np.zeros((T, K), dtype=np.int64)
        for k in range(K):
            embedding_indices[:, k] = self.speech_token_offset + (k * self.config.codebook_size) + codes[k]
        
        return embedding_indices
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training sample.
        
        Returns dict with:
            - input_ids: (seq_len,) - token IDs for LLM input
            - attention_mask: (seq_len,) - attention mask
            - labels: (seq_len,) - labels for temporal transformer (-100 for text portion)
            - speech_mask: (seq_len,) - boolean mask indicating speech token positions
            
            For multi-codebook only:
            - embedding_indices: (speech_len, K) - indices for summed embeddings
            - depth_targets: (speech_len, K-1) - targets for depth transformer
        """
        row = self.manifest.iloc[idx]
        
        # Load text and codes
        text = str(row[self.config.text_col])
        code_path = row[self.config.code_path_col]
        codes = self._load_codes(code_path)  # Shape: (K, T)
        
        # Truncate codes if needed
        max_speech = self.config.max_speech_frames
        if codes.shape[1] > max_speech:
            # raise warning
            logger.warning(f"Sample {idx}: codes length {codes.shape[1]} exceeds max_speech_frames {max_speech}, truncating")
            codes = codes[:, :max_speech]
        
        # Tokenize text
        text_tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.config.max_text_tokens,
            truncation=True,
        )
        
        # Convert codes to token IDs
        temporal_ids, depth_targets = self._codes_to_token_ids(codes)
        
        # Build sequence: [TEXT_START] text [TEXT_END] [SPEECH_START] speech [SPEECH_END]
        sequence = (
            [self.text_start_id]
            + text_tokens
            + [self.text_end_id, self.speech_start_id]
            + temporal_ids.tolist()
            + [self.speech_end_id]
        )
        
        # if total length exceeds max, truncate
        if len(sequence) > self.config.max_total_length:
            logger.warning(f"Sample {idx}: total sequence length {len(sequence)} exceeds max_total_length {self.config.max_total_length}, truncating")
            sequence = sequence[:self.config.max_total_length]
        
        # Create labels: only predict speech tokens
        # Text portion is masked with ignore_index
        labels = [self.ignore_index] * len(sequence)
        
        # Find speech start position and set labels for speech portion
        speech_start_pos = len(text_tokens) + 3  # After TEXT_START, text, TEXT_END, SPEECH_START
        for i in range(speech_start_pos, len(sequence)):
            labels[i] = sequence[i]
        
        # Create attention mask (all 1s, padding handled by collator)
        attention_mask = [1] * len(sequence)
        
        # Create speech mask for identifying speech positions
        speech_mask = [False] * len(sequence)
        for i in range(speech_start_pos, len(sequence) - 1):  # Exclude SPEECH_END
            speech_mask[i] = True
        
        # Build return dict
        result = {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speech_mask": torch.tensor(speech_mask, dtype=torch.bool),
        }
        
        # Add multi-codebook specific fields
        if self.config.num_codebooks > 1:
            embedding_indices = self._create_input_embeddings_indices(codes)
            
            # Truncate to match temporal_ids length
            actual_speech_len = len(temporal_ids) if isinstance(temporal_ids, list) else len(temporal_ids.tolist())
            embedding_indices = embedding_indices[:actual_speech_len]
            if depth_targets is not None:
                depth_targets = depth_targets[:actual_speech_len]
            
            result["embedding_indices"] = torch.tensor(embedding_indices, dtype=torch.long)
            if depth_targets is not None:
                result["depth_targets"] = torch.tensor(depth_targets, dtype=torch.long)
        
        return result


class SpeechLMCollator:
    """
    Collator for batching SpeechLM samples.
    
    Handles:
    - Padding sequences to max length in batch
    - Properly padding multi-codebook fields
    """
    
    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int = -100,
        num_codebooks: int = 1,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.num_codebooks = num_codebooks
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        
        # Find max lengths
        max_seq_len = max(sample["input_ids"].size(0) for sample in batch)
        
        # Initialize batch tensors
        batch_size = len(batch)
        
        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), self.ignore_index, dtype=torch.long)
        speech_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        
        # Multi-codebook fields
        if self.num_codebooks > 1:
            max_speech_len = max(
                sample["embedding_indices"].size(0) 
                for sample in batch 
                if "embedding_indices" in sample
            )
            embedding_indices = torch.zeros(
                batch_size, max_speech_len, self.num_codebooks, dtype=torch.long
            )
            depth_targets = torch.full(
                (batch_size, max_speech_len, self.num_codebooks - 1),
                self.ignore_index,
                dtype=torch.long,
            )
        
        # Fill batch tensors
        for i, sample in enumerate(batch):
            seq_len = sample["input_ids"].size(0)
            
            input_ids[i, :seq_len] = sample["input_ids"]
            attention_mask[i, :seq_len] = sample["attention_mask"]
            labels[i, :seq_len] = sample["labels"]
            speech_mask[i, :seq_len] = sample["speech_mask"]
            
            if self.num_codebooks > 1 and "embedding_indices" in sample:
                speech_len = sample["embedding_indices"].size(0)
                embedding_indices[i, :speech_len] = sample["embedding_indices"]
                if "depth_targets" in sample:
                    depth_targets[i, :speech_len] = sample["depth_targets"]
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "speech_mask": speech_mask,
        }
        
        if self.num_codebooks > 1:
            result["embedding_indices"] = embedding_indices
            result["depth_targets"] = depth_targets
        
        return result


def create_tokenizer(
    model_name: str,
    num_codebooks: int,
    codebook_size: int,
    special_tokens: Optional[List[str]] = None,
) -> AutoTokenizer:
    """
    Create tokenizer with expanded vocabulary for speech tokens.
    
    Args:
        model_name: Base model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        num_codebooks: Number of codebooks in the codec
        codebook_size: Size of each codebook vocabulary
        special_tokens: Additional special tokens to add
        
    Returns:
        Tokenizer with speech tokens added
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Default special tokens
    if special_tokens is None:
        special_tokens = [
            "<|SPEECH_START|>",
            "<|SPEECH_END|>",
            "<|TEXT_START|>",
            "<|TEXT_END|>",
        ]
    
    # Add special tokens
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    logger.info(f"Added {num_added} special tokens")
    
    # Add speech tokens for all codebooks
    # Format: <|s_{codebook}_{code}|> for clarity, but we'll use simpler format
    # <|s_0|>, <|s_1|>, ... for codebook 0
    # <|s_1024|>, <|s_1025|>, ... for codebook 1 (if codebook_size=1024)
    speech_tokens = []
    for k in range(num_codebooks):
        for c in range(codebook_size):
            token_idx = k * codebook_size + c
            speech_tokens.append(f"<|s_{token_idx}|>")
    
    num_speech_added = tokenizer.add_tokens(speech_tokens)
    logger.info(f"Added {num_speech_added} speech tokens ({num_codebooks} codebooks × {codebook_size} codes)")
    
    return tokenizer


def build_datasets(
    config: DataConfig,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple[SpeechLMDataset, Optional[SpeechLMDataset], AutoTokenizer]:
    """
    Build train and validation datasets.
    
    Args:
        config: DataConfig with paths and parameters
        tokenizer: Optional pre-created tokenizer
        
    Returns:
        (train_dataset, val_dataset, tokenizer)
    """
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = create_tokenizer(
            model_name=config.text_tokenizer,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
        )
    
    # Create datasets
    train_dataset = SpeechLMDataset(
        manifest_path=config.train_manifest,
        tokenizer=tokenizer,
        config=config,
        split="train",
    )
    
    val_dataset = None
    if config.val_manifest and Path(config.val_manifest).exists():
        val_dataset = SpeechLMDataset(
            manifest_path=config.val_manifest,
            tokenizer=tokenizer,
            config=config,
            split="val",
        )
    
    return train_dataset, val_dataset, tokenizer