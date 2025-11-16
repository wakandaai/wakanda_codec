# codec/model/dac.py

""" 
    DAC codec model implementation.

    Original code by Descript, Inc.
    Source: https://github.com/Alexgichamba/descript-audio-codec/blob/main/dac/model/dac.py
"""

from typing import List, Union, Dict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from codec.model.base import BaseCodec
from codec.nn.layers import Snake1d
from codec.nn.layers import WNConv1d
from codec.nn.layers import WNConvTranspose1d
from codec.nn.quantize import ResidualVectorQuantize

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

class EncoderBlock(nn.Module):
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

class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
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
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DACCodec(BaseCodec):
    """ DAC codec model implementation.

    Args:
        d_model: int
            Base channel dimension for the encoder.
        strides: list
            List of strides for each encoder block.
        d_latent: int
            Dimension of the latent representation.
        n_quantizers: int
            Number of quantizers to use.
        codebook_size: int
            Size of each codebook.
        codebook_dim: int
            Dimension of each codebook entry.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 1,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

    def preprocess(self, audio: torch.Tensor, sample_rate: int = None) -> torch.Tensor:
        """ Preprocesses audio data to ensure compatibility with the model's hop length."""
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio = nn.functional.pad(audio, (0, right_pad))

        return audio

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
        assert audio_sample_rate == self.sample_rate, "Input audio sample rate does not match model sample rate"
        audio = self.preprocess(audio, audio_sample_rate)
        h = self.encoder(audio)
        return h
    
    def quantize(self, h: torch.Tensor, n_quantizers: int = 1) -> Dict[str, torch.Tensor]:
        """ Quantizes continuous latents into discrete codes.

        Args:
            h: Tensor [B, C, T]
                Continuous latents from the encoder.
            n_quantizers: int
                Number of quantizers to use.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing quantized codes and other relevant information.
        """
        quantize_result = {}
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(h, n_quantizers)
        quantize_result["z"] = z
        quantize_result["codes"] = codes
        quantize_result["latents"] = latents
        quantize_result["vq/commitment_loss"] = commitment_loss
        quantize_result["vq/codebook_loss"] = codebook_loss
        return quantize_result

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """ Decodes quantized latents back into audio.

        Args:
            z: Quantized continuous representation of input.

        Returns:
            audio: Tensor [B, 1, T]
                Reconstructed audio from the decoder.
        """
        return self.decoder(z)

    def forward(self, audio: torch.Tensor, audio_sample_rate: int, n_quantizers: int = 1) -> Dict[str, torch.Tensor]:
        """ Forward pass through the DAC codec model.

        Args:
            audio: Tensor [B, 1, T]
                Input audio data.
            audio_sample_rate: int
                Sample rate of the input audio.
            n_quantizers: int
                Number of quantizers to use.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing reconstructed audio and other relevant information.
        """
        length = audio.shape[-1]
        h = self.encode(audio, audio_sample_rate)
        quantize_result = self.quantize(h, n_quantizers)
        z = quantize_result["z"]
        recon_audio = self.decode(z)

        output = {
            "recon_audio": recon_audio[..., :length],
            "z": quantize_result["z"],
            "codes": quantize_result["codes"],
            "latents": quantize_result["latents"],
            "vq/commitment_loss": quantize_result["vq/commitment_loss"],
            "vq/codebook_loss": quantize_result["vq/codebook_loss"],
            "length": audio.shape[-1],
        }
        return output

if __name__ == "__main__":
    from functools import partial

    # test DAC 16k model
    sample_rate= 16000
    encoder_dim = 64
    encoder_rates = [2, 4, 5, 8]
    decoder_dim = 1536
    decoder_rates = [8, 5, 4, 2]
    n_codebooks = 12
    codebook_size = 1024
    codebook_dim = 8
    quantizer_dropout = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DACCodec(
        encoder_dim=encoder_dim,
        encoder_rates=encoder_rates,
        decoder_dim=decoder_dim,
        decoder_rates=decoder_rates,
        n_codebooks=n_codebooks,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
        quantizer_dropout=quantizer_dropout,
        sample_rate=sample_rate,
    ).to(device)

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 16001
    x = torch.randn(1, 1, length).to(device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(audio=x, audio_sample_rate=sample_rate, n_quantizers=n_codebooks)
    recon_audio = out["recon_audio"]
    print("Input shape:", x.shape)
    print("Output shape:", recon_audio.shape)

    # Create gradient variable
    grad = torch.zeros_like(recon_audio)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    recon_audio.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    # check codec frame rate
    hop_length = np.prod(encoder_rates)
    print("Codec hop length (samples):", hop_length)
    print("Codec frame rate (Hz):", sample_rate / hop_length)

    # check shape of codes
    codes = out["codes"]
    print("Codes shape:", codes.shape)
