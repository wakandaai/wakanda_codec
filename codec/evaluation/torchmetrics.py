# codec/evaluation/torchmetrics.py

import torch
from typing import Optional
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


def compute_NISQA(dec: torch.Tensor, fs: int = 16000) -> float:
    """
    Compute Non-Intrusive Speech Quality Assessment (NISQA) score.

    Args:
        dec (torch.Tensor): Decoded audio tensor of shape (batch_size, time).
        fs (int): Sampling frequency of the audio signals. Default is 16000 Hz.

    Returns:
        float: NISQA score.
    """
    nisqa_metric = NonIntrusiveSpeechQualityAssessment(fs=fs)
    score = nisqa_metric(dec)
    return score.item()

def compute_DNSMOS(dec: torch.Tensor, fs: int = 16000, personalized: bool=False, device: Optional[str]="cuda:0", num_threads: Optional[int] = None, cache_sessions: bool = True) -> float:
    """
    Compute Deep Noise Suppression Mean Opinion Score (DNSMOS).

    Args:
        dec (torch.Tensor): Decoded audio tensor of shape (batch_size, time).
        fs (int): Sampling frequency of the audio signals. Default is 16000 Hz.
        personalized (bool): Whether to penalize for interfering (undesired neighboring) speakers. Default is False.
        device (Optional[str]): Device to run the model on. Default is "cuda:0".
        num_threads (Optional[int]): Number of threads to use for computation. Default is None.
        cache_sessions (bool): Whether to cache sessions. Default is True.

    Returns:
        float: DNSMOS score.
    """
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(fs=fs, personalized=personalized, device=device, num_threads=num_threads, cache_sessions=cache_sessions)
    score = dnsmos_metric(dec)
    return score.item()

def compute_PESQ(ref: torch.Tensor, dec: torch.Tensor, fs: int = 16000, mode: str = 'nb', n_processes: int = 1) -> float:
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ) score.

    Args:
        ref (torch.Tensor): Reference audio tensor of shape (batch_size, time).
        dec (torch.Tensor): Decoded audio tensor of shape (batch_size, time).
        fs (int): Sampling frequency of the audio signals. Default is 16000 Hz.
        mode (str): Mode of PESQ ('nb' for narrowband, 'wb' for wideband). Default is 'nb'.
        n_processes (int): Number of processes to use for computation. Default is 1.

    Returns:
        float: PESQ score.
    """
    pesq_metric = PerceptualEvaluationSpeechQuality(fs=fs, mode=mode, n_processes=n_processes)
    score = pesq_metric(dec, ref)
    return score.item()