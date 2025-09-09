# codec/evaluation/utmos.py

"""
UTMOS Evaluation for Neural Audio Codecs

This module provides UTMOS (UTokyo-SaruLab MOS) evaluation for neural audio codecs.
UTMOS is a state-of-the-art neural MOS predictor trained on human ratings.

Based on: https://github.com/tarepan/SpeechMOS
Paper: "UTMOS: UTokyo-SaruLab Mean Opinion Score Prediction System"
"""

import os
import warnings
from typing import Union, Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import torch
from torchaudio.transforms import Resample

# =============================================================================
# UTMOS Model Loading and Management
# Based on https://github.com/tarepan/SpeechMOS
# =============================================================================

class UTMOSPredictor:
    """
    UTMOS predictor wrapper for consistent interface
    """
    
    def __init__(self, model_name: str = "utmos22_strong", device: str = 'auto'):
        """
        Initialize UTMOS predictor
        
        Args:
            model_name: UTMOS model variant to use
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.predictor = None
        self._load_model()
    
    def _get_device(self, device: str) -> str:
        """Determine appropriate device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """Load UTMOS model from torch.hub"""
        try:
            # Load model from torch hub
            self.predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", 
                self.model_name, 
                trust_repo=True
            )
            self.predictor = self.predictor.to(self.device)
            self.predictor.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load UTMOS model '{self.model_name}': {e}")
    
    def predict(self, audio: torch.Tensor, sample_rate: int) -> float:
        """
        Predict MOS score for audio
        
        Args:
            audio: Audio tensor (1D)
            sample_rate: Sample rate of audio
            
        Returns:
            MOS score (1.0 - 5.0, higher is better)
        """
        if self.predictor is None:
            raise RuntimeError("UTMOS predictor not initialized")
        
        # Ensure audio is on correct device
        audio = audio.to(self.device)
        
        # UTMOS expects specific input format
        with torch.no_grad():
            score = self.predictor(audio, sample_rate)
        
        # Extract scalar score
        if isinstance(score, torch.Tensor):
            score = score.item()
        
        return float(score)


# Global predictor instance for efficient reuse
_global_predictor = None


def _get_predictor(model_name: str = "utmos22_strong", device: str = 'auto') -> UTMOSPredictor:
    """Get global predictor instance (create if needed)"""
    global _global_predictor
    
    if _global_predictor is None or _global_predictor.model_name != model_name:
        _global_predictor = UTMOSPredictor(model_name, device)
    
    return _global_predictor


# =============================================================================
# Standardized Interface Functions
# =============================================================================

def _load_and_validate_audio(audio_input: Union[str, np.ndarray], 
                           sample_rate: Optional[int] = None) -> tuple:
    """
    Load and validate audio input
    
    Args:
        audio_input: Audio file path or numpy array
        sample_rate: Sample rate (required for numpy arrays)
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        
        audio, sr = sf.read(audio_input)
        audio = torch.from_numpy(audio).float()
        
        # Handle multi-channel audio by taking first channel
        if audio.ndim > 1:
            audio = audio[:, 0]
            warnings.warn("Multi-channel audio detected, using first channel for UTMOS evaluation")
        
        return audio, sr
        
    elif isinstance(audio_input, np.ndarray):
        if sample_rate is None:
            raise ValueError("Sample rate must be provided when using numpy arrays")
        
        audio = torch.from_numpy(audio_input).float()
        
        # Handle multi-channel audio by taking first channel
        if audio.ndim > 1:
            audio = audio[:, 0]
            warnings.warn("Multi-channel audio detected, using first channel for UTMOS evaluation")
        
        return audio, sample_rate
        
    else:
        raise TypeError("Audio input must be file path (str) or numpy array")


def compute_utmos(reference: Union[str, np.ndarray],
                          decoded: Union[str, np.ndarray], 
                          sample_rate: Optional[int] = None,
                          model_name: str = "utmos22_strong",
                          use_gpu: bool = True,
                          return_both: bool = False) -> Union[float, Dict[str, float]]:
    """
    Compute UTMOS scores for reference and decoded audio
    
    Args:
        reference: Reference audio (file path or numpy array)
        decoded: decoded/decoded audio (file path or numpy array)
        sample_rate: Sample rate (required if using numpy arrays)
        model_name: UTMOS model variant
        use_gpu: Whether to use GPU for inference
        return_both: If True, return dict with both scores; if False, return decoded score only
        
    Returns:
        decoded UTMOS score (float) or dict with both scores
        
    Example:
        >>> # Get decoded score only
        >>> score = compute_utmos_reference("ref.wav", "decoded.wav")
        >>> print(f"decoded UTMOS: {score:.3f}")
        
        >>> # Get both scores
        >>> scores = compute_utmos_reference("ref.wav", "decoded.wav", return_both=True)
        >>> print(f"Reference: {scores['reference']:.3f}, decoded: {scores['decoded']:.3f}")
    """
    ref_score = compute_utmos(reference, sample_rate, model_name, use_gpu)
    dec_score = compute_utmos(decoded, sample_rate, model_name, use_gpu)
    
    if return_both:
        return {
            'reference': ref_score,
            'decoded': dec_score,
        }
    else:
        return dec_score