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