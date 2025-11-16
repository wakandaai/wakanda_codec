# codec/model/base.py

""" Base classes for codec models.

    Defines BaseCodec abstract class with required methods:

    encode() - takes audio, returns latents/codes
    decode() - takes latents/codes, returns audio
    forward() - full encode-quantize-decode pass
    and:
    Properties for sample rate and hop length
"""

from abc import ABC, abstractmethod
import torch
from typing import Dict

class BaseCodec(ABC, torch.nn.Module):
    """ Abstract base class for neural audio codec models.
    
    All codec implementations must inherit from this class and implement the required abstract methods.
    This ensures a consistent interface across different codec architectures.
    
    The base class provides:
    - Abstract methods for encode/decode operations
    - Properties for sample rate and hop length
    """
    
    def __init__(self):
        super().__init__()
        self._sample_rate: int
        self._hop_length: int


    @abstractmethod
    def encode(self, audio: torch.Tensor, audio_sample_rate: int) -> torch.Tensor:
        """ Encodes audio into continuous latents

        Args:
            audio: Tensor [B, 1, T]
                Input audio data.
            audio_sample_rate: int
                Sample rate of the input audio.

        Returns:
            h: Tensor [B, C, T']
                The continuous latents from the encoder
        """
        pass

    @abstractmethod
    def quantize(self, h: torch.Tensor, n_quantizers: int = 1) -> Dict[str, torch.Tensor]:
        """ Quantizes continuous latents into discrete codes.

        Args:
            h: Tensor [B, C, T']
                Continuous latents from the encoder.
            n_quantizers: int
                Number of quantizers to use. Default is 1.
        Returns:
            dict
                A dictionary with the following:
                "z" : Tensor[B x D x T']
                    Quantized continuous representation of input
                "codes" : Tensor[B x N x T']
                    Codebook indices for each codebook (quantized discrete representation of input)
                "latents" : Tensor[B x N*D x T']
                    Projected latents continuous representation of input before quantization
                "vq/commitment_loss" : Tensor[1]
                    Commitment loss to train encoder to predict vectors closer to codebook
                    entries
                "vq/codebook_loss" : Tensor[1]
                    Codebook loss to update the codebook
                "length" : int
                    Number of samples in input audio
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """ Decodes quantized latents back into audio.

        Args:
            z: Quantized continuous representation of input.

        Returns:
            recon_audio: Tensor [B, 1, T]
                Reconstructed audio data.
        """
        pass

    @abstractmethod
    def forward(self, audio: torch.Tensor, audio_sample_rate: int, n_quantizers: int = 1) -> Dict[str, torch.Tensor]:
        """ Full encode-quantize-decode pass.

        Args:
            audio: Tensor [B, 1, T]
                Input audio data.
            audio_sample_rate: int
                Sample rate of the input audio.
            n_quantizers: int
                Number of quantizers to use.

        Returns:
            dict
                A dictionary with the following:
                "recon_audio" : Tensor[B x 1 x T]
                    Reconstructed audio data.
                "codes" : Tensor[B x N x T']
                    Codebook indices for each codebook (quantized discrete representation of input)
                "latents" : Tensor[B x N*D x T']
                    Projected latents continuous representation of input before quantization
                "vq/commitment_loss" : Tensor[1]
                    Commitment loss to train encoder to predict vectors closer to codebook
                    entries
                "vq/codebook_loss" : Tensor[1]
                    Codebook loss to update the codebook
                "length" : int
                    Number of samples in input audio
        """
        pass