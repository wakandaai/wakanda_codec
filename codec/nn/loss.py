# codec/nn/loss.py

# Code by Descript, Inc.
# Original source: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py
# Modified to remove audiotools dependency

import typing
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn
import torchaudio


class STFTConfig:
    def __init__(
        self,
        window_length: int,
        hop_length: int,
        window_type: Optional[str] = "hann"
    ):
        self.window_length = window_length
        self.hop_length = hop_length
        self.window_type = window_type


def compute_stft(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window_type: str = "hann"
) -> torch.Tensor:
    """Compute STFT magnitude.
    
    Parameters
    ----------
    audio : torch.Tensor
        Audio tensor (B, C, T)
    n_fft : int
        FFT size
    hop_length : int
        Hop length
    window_type : str
        Window type
        
    Returns
    -------
    torch.Tensor
        STFT magnitude (B, C, F, T)
    """
    B, C, T = audio.shape
    audio = audio.reshape(B * C, T)
    
    # Create window
    if window_type == "hann":
        window = torch.hann_window(n_fft, device=audio.device)
    elif window_type == "hamming":
        window = torch.hamming_window(n_fft, device=audio.device)
    else:
        window = torch.ones(n_fft, device=audio.device)
    
    # Compute STFT
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True
    )
    
    # Compute magnitude
    magnitude = torch.abs(stft)
    magnitude = magnitude.reshape(B, C, magnitude.shape[-2], magnitude.shape[-1])
    
    return magnitude


def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    window_type: str = "hann"
) -> torch.Tensor:
    """Compute mel spectrogram.
    
    Parameters
    ----------
    audio : torch.Tensor
        Audio tensor (B, C, T)
    sample_rate : int
        Sample rate
    n_fft : int
        FFT size
    hop_length : int
        Hop length
    n_mels : int
        Number of mel bins
    f_min : float
        Minimum frequency
    f_max : Optional[float]
        Maximum frequency
    window_type : str
        Window type
        
    Returns
    -------
    torch.Tensor
        Mel spectrogram (B, C, n_mels, T)
    """
    # Compute STFT magnitude
    magnitude = compute_stft(audio, n_fft, hop_length, window_type)
    # magnitude is (B, C, F, T)
    
    B, C, F, T_frames = magnitude.shape
    
    # Create mel filterbank
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=F,
        f_min=f_min,
        f_max=f_max if f_max is not None else sample_rate / 2,
        n_mels=n_mels,
        sample_rate=sample_rate
    ).to(audio.device)
    
    # Reshape for batch matrix multiplication
    magnitude = magnitude.reshape(B * C, F, T_frames)
    
    # Apply mel filterbank: (n_mels, F) @ (F, T) = (n_mels, T)
    mel_spec = torch.matmul(mel_fb.T, magnitude)
    
    # Reshape back to (B, C, n_mels, T)
    mel_spec = mel_spec.reshape(B, C, n_mels, T_frames)
    
    return mel_spec


class L1Loss(nn.L1Loss):
    """L1 Loss between audio tensors.
    
    Parameters
    ----------
    weight : float, optional
        Weight of this loss, defaults to 1.0.
    """
    def __init__(self, weight: float = 1.0, **kwargs):
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Estimate audio tensor (B, C, T)
        y : torch.Tensor
            Reference audio tensor (B, C, T)

        Returns
        -------
        torch.Tensor
            L1 loss between audio tensors.
        """
        return super().forward(x, y)


class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : bool, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or None), by default 'mean'
    zero_mean : bool, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : float, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.
    """

    def __init__(
        self,
        scaling: bool = True,
        reduction: str = "mean",
        zero_mean: bool = True,
        clip_min: Optional[float] = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Estimate audio tensor (B, C, T)
        y : torch.Tensor
            Reference audio tensor (B, C, T)

        Returns
        -------
        torch.Tensor
            SI-SDR loss
        """
        eps = 1e-8
        
        references = x
        estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of audio, by default 24000
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    window_type : str, optional
        Type of window to use, by default "hann"

    References
    ----------
    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        window_type: str = "hann",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.stft_configs = [
            STFTConfig(
                window_length=w,
                hop_length=w // 4,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes multi-scale STFT between estimate and reference.
        
        Parameters
        ----------
        x : torch.Tensor
            Estimate signal (B, C, T)
        y : torch.Tensor
            Reference signal (B, C, T)

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for config in self.stft_configs:
            x_mag = compute_stft(
                x, 
                config.window_length, 
                config.hop_length,
                config.window_type
            )
            y_mag = compute_stft(
                y,
                config.window_length,
                config.hop_length,
                config.window_type
            )
            
            loss += self.log_weight * self.loss_fn(
                x_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mag.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mag, y_mag)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of audio, by default 24000
    n_mels : List[int], optional
        Number of mels per STFT, by default [150, 80]
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    mel_fmin : List[float], optional
        Minimum frequency for mel filterbank, by default [0.0, 0.0]
    mel_fmax : List[Optional[float]], optional
        Maximum frequency for mel filterbank, by default [None, None]
    window_type : str, optional
        Type of window to use, by default "hann"
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[Optional[float]] = [None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.stft_configs = [
            STFTConfig(
                window_length=w,
                hop_length=w // 4,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn if loss_fn is not None else nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Computes mel loss between estimate and reference.
        
        Parameters
        ----------
        x : torch.Tensor
            Estimate signal (B, C, T)
        y : torch.Tensor
            Reference signal (B, C, T)

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        for n_mels, fmin, fmax, config in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_configs
        ):
            x_mels = compute_mel_spectrogram(
                x,
                self.sample_rate,
                config.window_length,
                config.hop_length,
                n_mels,
                fmin,
                fmax,
                config.window_type
            )
            y_mels = compute_mel_spectrogram(
                y,
                self.sample_rate,
                config.window_length,
                config.hop_length,
                n_mels,
                fmin,
                fmax,
                config.window_type
            )

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake: torch.Tensor, real: torch.Tensor):
        """
        Parameters
        ----------
        fake : torch.Tensor
            Generated audio (B, C, T)
        real : torch.Tensor
            Real audio (B, C, T)
        
        Returns
        -------
        Tuple[List, List]
            Discriminator outputs for fake and real audio
        """
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        """Compute discriminator loss.
        
        Parameters
        ----------
        fake : torch.Tensor
            Generated audio (B, C, T)
        real : torch.Tensor
            Real audio (B, C, T)
            
        Returns
        -------
        torch.Tensor
            Discriminator loss
        """
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        """Compute generator loss.
        
        Parameters
        ----------
        fake : torch.Tensor
            Generated audio (B, C, T)
        real : torch.Tensor
            Real audio (B, C, T)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Generator adversarial loss and feature matching loss
        """
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature