# codec/model/transformer_codec.py

"""
Transformer-based Neural Audio Codec

Architecture:
    Encoder: Conv Downsampling → Transformer → Latent Projection
    Decoder: Latent Projection → Transformer → Conv Upsampling

Inspired by Mimi (Kyutai) but with flexible configuration.
"""

import math
from typing import List, Union, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from codec.model.base import BaseCodec
from codec.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d
from codec.nn.quantize import ResidualVectorQuantize
from codec.nn.transformer import SinusoidalPositionalEncoding

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

class ConvDownsampleBlock(nn.Module):
    """Convolutional downsampling block with residual units"""
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)

class ConvUpsampleBlock(nn.Module):
    """Convolutional upsampling block with residual units"""
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=0 if stride % 2 == 0 else 1
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)

class ConvEncoder(nn.Module):
    """
    Convolutional encoder with downsampling
    
    Args:
        d_model: Base channel dimension
        strides: List of stride values for downsampling
        d_latent: Output dimension after conv encoding
    """
    def __init__(
        self,
        d_model: int = 64,
        strides: List[int] = [2, 4, 8, 8],
        d_latent: int = 512,
    ):
        super().__init__()
        
        # Initial convolution
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        
        # Downsampling blocks
        for stride in strides:
            d_model *= 2
            layers.append(ConvDownsampleBlock(d_model, stride=stride))
        
        # Final projection
        layers.extend([
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ])
        
        self.block = nn.Sequential(*layers)
        self.output_dim = d_latent
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, T]
        Returns:
            [B, d_latent, T']
        """
        return self.block(x)

class ConvDecoder(nn.Module):
    """
    Convolutional decoder with upsampling
    
    Args:
        input_dim: Input dimension from transformer
        channels: Base channel dimension for decoder
        strides: List of stride values for upsampling
        d_out: Output channels (usually 1 for mono audio)
    """
    def __init__(
        self,
        input_dim: int = 512,
        channels: int = 1536,
        strides: List[int] = [8, 8, 4, 2],
        d_out: int = 1,
    ):
        super().__init__()
        
        # Initial convolution
        layers = [WNConv1d(input_dim, channels, kernel_size=7, padding=3)]
        
        # Upsampling blocks
        for i, stride in enumerate(strides):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers.append(ConvUpsampleBlock(input_dim, output_dim, stride))
        
        # Final projection
        layers.extend([
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, input_dim, T']
        Returns:
            [B, d_out, T]
        """
        return self.block(x)

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing conv features
    
    Args:
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        causal: Whether to use causal (autoregressive) attention
    """
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.causal = causal
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN architecture (more stable)
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B, D, T] - output from conv encoder
        Returns:
            [B, D, T] - transformed features
        """
        # Reshape: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask if causal
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Apply transformer
        x = self.transformer(x, mask=mask, is_causal=self.causal)
        
        # Final norm
        x = self.norm(x)
        
        # Reshape back: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for processing quantized latents
    
    Args:
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        causal: Whether to use causal (autoregressive) attention
    """
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.causal = causal
        
        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder layers (using encoder-only architecture like BERT)
        # For a true encoder-decoder, we'd use TransformerDecoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: [B, D, T'] - quantized latents
        Returns:
            [B, D, T'] - transformed features
        """
        # Reshape: [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask if causal
        mask = None
        if self.causal:
            seq_len = x.size(1)
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Apply transformer
        x = self.transformer(x, mask=mask, is_causal=self.causal)
        
        # Final norm
        x = self.norm(x)
        
        # Reshape back: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        
        return x

class TransformerCodec(BaseCodec):
    """
    Transformer-based neural audio codec
    
    Architecture:
        Input Audio [B, 1, T]
        ↓
        Conv Encoder (downsampling) → [B, D, T']
        ↓
        Transformer Encoder → [B, D, T']
        ↓
        Residual Vector Quantization → [B, D, T']
        ↓
        Transformer Decoder → [B, D, T']
        ↓
        Conv Decoder (upsampling) → [B, 1, T]
        ↓
        Output Audio
    
    Args:
        # Encoder config
        encoder_dim: Base dimension for conv encoder
        encoder_rates: Downsampling strides for encoder
        
        # Transformer config
        latent_dim: Dimension of latent space
        encoder_transformer_layers: Number of transformer layers in encoder
        decoder_transformer_layers: Number of transformer layers in decoder
        transformer_num_heads: Number of attention heads
        transformer_dim_feedforward: Dimension of feedforward network
        transformer_dropout: Dropout rate
        encoder_causal: Whether encoder uses causal attention
        decoder_causal: Whether decoder uses causal attention
        
        # Decoder config
        decoder_dim: Base dimension for conv decoder
        decoder_rates: Upsampling strides for decoder
        
        # Quantization config
        n_codebooks: Number of codebooks for RVQ
        codebook_size: Size of each codebook
        codebook_dim: Dimension of codebook entries
        quantizer_dropout: Quantizer dropout probability
        
        # Audio config
        sample_rate: Sample rate of audio
    """
    
    def __init__(
        self,
        # Encoder config
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        
        # Transformer config
        latent_dim: int = 512,
        encoder_transformer_layers: int = 6,
        decoder_transformer_layers: int = 6,
        transformer_num_heads: int = 8,
        transformer_dim_feedforward: int = 2048,
        transformer_dropout: float = 0.1,
        encoder_causal: bool = False,
        decoder_causal: bool = False,
        
        # Decoder config
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        
        # Quantization config
        n_codebooks: int = 12,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        quantizer_dropout: float = 0.0,
        
        # Audio config
        sample_rate: int = 16000,
    ):
        super().__init__()
        
        # Store config
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        
        # Calculate hop length
        self.hop_length = np.prod(encoder_rates)
        
        # Build encoder
        self.conv_encoder = ConvEncoder(
            d_model=encoder_dim,
            strides=encoder_rates,
            d_latent=latent_dim,
        )
        
        self.transformer_encoder = TransformerEncoder(
            d_model=latent_dim,
            num_layers=encoder_transformer_layers,
            num_heads=transformer_num_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            causal=encoder_causal,
        )
        
        # Build quantizer
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        
        # Build decoder
        self.transformer_decoder = TransformerDecoder(
            d_model=latent_dim,
            num_layers=decoder_transformer_layers,
            num_heads=transformer_num_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            causal=decoder_causal,
        )
        
        self.conv_decoder = ConvDecoder(
            input_dim=latent_dim,
            channels=decoder_dim,
            strides=decoder_rates,
            d_out=1,
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def preprocess(self, audio: torch.Tensor, sample_rate: int = None) -> torch.Tensor:
        """Pad audio to be divisible by hop length"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate, \
            f"Sample rate mismatch: {sample_rate} != {self.sample_rate}"
        
        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio = F.pad(audio, (0, right_pad))
        
        return audio
    
    def encode(self, audio: torch.Tensor, audio_sample_rate: int) -> torch.Tensor:
        """
        Encode audio into continuous latents
        
        Args:
            audio: [B, 1, T] - input audio
            audio_sample_rate: Sample rate of input
            
        Returns:
            h: [B, D, T'] - continuous latents
        """
        assert audio_sample_rate == self.sample_rate, \
            f"Sample rate mismatch: {audio_sample_rate} != {self.sample_rate}"
        
        # Preprocess (pad to hop length)
        audio = self.preprocess(audio, audio_sample_rate)
        
        # Conv encoding with downsampling
        h = self.conv_encoder(audio)  # [B, D, T']
        
        # Transformer encoding
        h = self.transformer_encoder(h)  # [B, D, T']
        
        return h
    
    def quantize(self, h: torch.Tensor, n_quantizers: int = None) -> Dict[str, torch.Tensor]:
        """
        Quantize continuous latents
        
        Args:
            h: [B, D, T'] - continuous latents
            n_quantizers: Number of quantizers to use (None = use all)
            
        Returns:
            Dictionary with:
                - z: [B, D, T'] - quantized latents
                - codes: [B, N, T'] - codebook indices
                - latents: [B, N*D, T'] - projected latents before quantization
                - vq/commitment_loss: commitment loss
                - vq/codebook_loss: codebook loss
        """
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(h, n_quantizers)
        
        return {
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to audio
        
        Args:
            z: [B, D, T'] - quantized latents
            
        Returns:
            audio: [B, 1, T] - reconstructed audio
        """
        # Transformer decoding
        h = self.transformer_decoder(z)  # [B, D, T']
        
        # Conv decoding with upsampling
        audio = self.conv_decoder(h)  # [B, 1, T]
        
        return audio
    
    def forward(
        self, 
        audio: torch.Tensor, 
        audio_sample_rate: int, 
        n_quantizers: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full encode-quantize-decode pass
        
        Args:
            audio: [B, 1, T] - input audio
            audio_sample_rate: Sample rate
            n_quantizers: Number of quantizers to use
            
        Returns:
            Dictionary with:
                - recon_audio: [B, 1, T] - reconstructed audio
                - z: [B, D, T'] - quantized latents
                - codes: [B, N, T'] - codebook indices
                - latents: [B, N*D, T'] - projected latents
                - vq/commitment_loss: commitment loss
                - vq/codebook_loss: codebook loss
                - length: original audio length
        """
        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        
        length = audio.shape[-1]
        
        # Encode
        h = self.encode(audio, audio_sample_rate)
        
        # Quantize
        quant_result = self.quantize(h, n_quantizers)
        z = quant_result["z"]
        
        # Decode
        recon_audio = self.decode(z)
        
        # Trim to original length
        recon_audio = recon_audio[..., :length]
        
        return {
            "recon_audio": recon_audio,
            "z": z,
            "codes": quant_result["codes"],
            "latents": quant_result["latents"],
            "vq/commitment_loss": quant_result["vq/commitment_loss"],
            "vq/codebook_loss": quant_result["vq/codebook_loss"],
            "length": length,
        }


if __name__ == "__main__":
    from functools import partial
    
    # Test TransformerCodec 16kHz model
    sample_rate = 16000
    encoder_dim = 64
    encoder_rates = [2, 4, 5, 8]
    latent_dim = 512
    encoder_transformer_layers = 4
    decoder_transformer_layers = 4
    transformer_num_heads = 8
    transformer_dim_feedforward = 2048
    decoder_dim = 1536
    decoder_rates = [8, 5, 4, 2]
    n_codebooks = 12
    codebook_size = 1024
    codebook_dim = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TransformerCodec(
        encoder_dim=encoder_dim,
        encoder_rates=encoder_rates,
        latent_dim=latent_dim,
        encoder_transformer_layers=encoder_transformer_layers,
        decoder_transformer_layers=decoder_transformer_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_dim_feedforward=transformer_dim_feedforward,
        decoder_dim=decoder_dim,
        decoder_rates=decoder_rates,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        sample_rate=sample_rate,
    ).to(device)
    
    # Print model info
    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]) / 1e6, "M")
    
    # Test forward pass
    length = 16001
    x = torch.randn(2, 1, length).to(device)
    x.requires_grad_(True)
    x.retain_grad()
    
    print("\n" + "="*60)
    print("Testing forward pass...")
    print("="*60)
    
    out = model(audio=x, audio_sample_rate=sample_rate, n_quantizers=n_codebooks)
    recon_audio = out["recon_audio"]
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {recon_audio.shape}")
    print(f"Codes shape: {out['codes'].shape}")
    print(f"Latents shape: {out['latents'].shape}")
    
    # Test backward pass
    print("\n" + "="*60)
    print("Testing backward pass...")
    print("="*60)
    
    grad = torch.zeros_like(recon_audio)
    grad[:, :, grad.shape[-1] // 2] = 1
    recon_audio.backward(grad)
    
    # Check receptive field
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)
    rf = (gradmap != 0).sum()
    
    print(f"Receptive field: {rf.item()} samples")
    
    # Check codec properties
    hop_length = np.prod(encoder_rates)
    print(f"\nCodec hop length (samples): {hop_length}")
    print(f"Codec frame rate (Hz): {sample_rate / hop_length:.2f}")
    print(f"Compression ratio: {hop_length}x")