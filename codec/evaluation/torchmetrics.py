# codec/evaluation/torchmetrics.py

import torch
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment, DeepNoiseSuppressionMeanOpinionScore

def compute_NISQA(enh: torch.Tensor, fs: int = 16000) -> float:
    """
    Compute Non-Intrusive Speech Quality Assessment (NISQA) score.

    Args:
        ref (torch.Tensor): Reference audio tensor of shape (batch_size, time).
        enh (torch.Tensor): Enhanced audio tensor of shape (batch_size, time).
        fs (int): Sampling frequency of the audio signals. Default is 16000 Hz.

    Returns:
        float: NISQA score.
    """
    nisqa_metric = NonIntrusiveSpeechQualityAssessment(fs=fs)
    score = nisqa_metric(enh, ref)
    return score.item()

def compute_DNSMOS(enh: torch.Tensor, fs: int = 16000, personalized: bool=False, device: Optional[str]="cuda:0", num_threads: Optional[int] = None, cache_sessions: bool = True) -> float:
    """
    Compute Deep Noise Suppression Mean Opinion Score (DNSMOS).

    Args:
        enh (torch.Tensor): Enhanced audio tensor of shape (batch_size, time).
        fs (int): Sampling frequency of the audio signals. Default is 16000 Hz.
        personalized (bool): Whether to penalize for interfering (undesired neighboring) speakers. Default is False.
        device (Optional[str]): Device to run the model on. Default is "cuda:0".
        num_threads (Optional[int]): Number of threads to use for computation. Default is None.
        cache_sessions (bool): Whether to cache sessions. Default is True.

    Returns:
        float: DNSMOS score.
    """
    dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(fs=fs, personalized=personalized, device=device, num_threads=num_threads, cache_sessions=cache_sessions)
    score = dnsmos_metric(enh)
    return score.item()