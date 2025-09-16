# codec/evaluation/speaker_sim.py

"""
Speaker Similarity Evaluation using ESPNet

This module provides speaker similarity metrics using ESPNet's pre-trained
speaker embedding models.

Based on ESPnet-SPK
https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/spk1
"""

import os
import warnings
import logging
from typing import Union, Optional, Dict, Any, List
from espnet2.bin.spk_inference import Speech2Embedding
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

logger = logging.getLogger(__name__)

def _load_espnet_speaker_model(model_name: str, device: str = "cpu"):
    """
    Load ESPNet pre-trained speaker verification model
    
    Args:
        model_name: ESPNet model identifier from HuggingFace Hub
        device: Device to load model on
        
    Returns:
        Loaded ESPNet speaker model
    """
        
    # Initialize the speaker embedding extractor
    speech2embedding = Speech2Embedding.from_pretrained(
        model_tag=model_name,
        device=device,
    )
    
    logger.info(f"Loaded ESPNet speaker model: {model_name}")
    return speech2embedding

def _extract_speaker_embedding(model, audio_path: str, 
                              target_sr: int = 16000) -> torch.Tensor:
    """
    Extract speaker embedding from audio file using ESPNet model
    
    Args:
        model: Loaded ESPNet Speech2Embedding model
        audio_path: Path to audio file
        target_sr: Target sample rate for the model
        
    Returns:
        Speaker embedding tensor
    """
    try:
        # Load audio
        audio, sr = sf.read(audio_path, dtype=np.float32)
        
        # Convert to torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = Resample(orig_freq=sr, new_freq=target_sr)
            audio = resampler(audio)
        
        if audio.dim() > 1:
            audio = audio.mean(dim=0)  # Convert to mono
        
        # Extract embedding
        embedding = model(audio)
        
        return embedding
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract embedding from {audio_path}: {e}")

def init_model(model_name: str, use_gpu: bool = True):
    """
    Initialize ESPNet speaker verification model
    
    Args:
        model_name: ESPNet model name (e.g., "espnet/voxceleb12_rawnet3")
        use_gpu: Whether to use GPU if available
        
    Returns:
        Loaded ESPNet speaker model
    """
    
    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load model
    model = _load_espnet_speaker_model(model_name, device)
    return model

def compute_speaker_similarity(reference: Union[str, np.ndarray],
                               decoded: Union[str, np.ndarray],
                               model: Any,
                               sample_rate: Optional[int] = 16000) -> float:
    """
    Compute speaker similarity between reference and decoded audio using ESPNet
    
    Args:
        reference: Reference audio (file path or numpy array)
        decoded: Decoded audio (file path or numpy array)
        model: Pre-loaded ESPNet Speech2Embedding model
        sample_rate: Sample rate (required if using numpy arrays)
        
    Returns:
        Cosine similarity score between speaker embeddings (-1.0 to 1.0)
        Higher values indicate more similar speakers
        
    Example:
        >>> model = init_model("espnet/voxceleb12_rawnet3")
        >>> sim_score = compute_speaker_similarity("reference.wav", "decoded.wav", model)
        >>> print(f"Speaker similarity: {sim_score:.4f}")
    """
    try:
        # Handle file path vs numpy array inputs differently for ESPNet
        if isinstance(reference, str) and isinstance(decoded, str):
            # Use ESPNet's direct file processing
            ref_embedding = _extract_speaker_embedding(model, reference, sample_rate)
            dec_embedding = _extract_speaker_embedding(model, decoded, sample_rate)
        else:
            raise NotImplementedError(
                "ESPNet speaker similarity currently requires file paths. "
                "Numpy array input is not yet supported with ESPNet models."
            )
        
        # Compute cosine similarity between embeddings
        similarity = F.cosine_similarity(ref_embedding, dec_embedding)
        
        return float(similarity.item())
        
    except Exception as e:
        logger.error(f"Error computing speaker similarity: {e}")
        raise