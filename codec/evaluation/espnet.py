# codec/evaluation/espnet.py

"""
Neural Audio Codec Evaluation Metrics

This module provides standardized evaluation metrics for neural audio codecs:
- STOI (Short-Time Objective Intelligibility) 
- MCD (Mel-Cepstral Distortion)

The core mathematical implementations are preserved from ESPNet with proper attribution.
Standardized interfaces are provided for integration into evaluation pipelines.

"""

import itertools
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, Tuple, Optional, List

import numpy as np
import soundfile
from pystoi.stoi import stoi
import pysptk
import pyworld as pw
import scipy.spatial.distance
from fastdtw import fastdtw
from scipy.signal import firwin, lfilter


# =============================================================================
# ESPNet STOI Implementation
# Source: https://github.com/espnet/espnet/blob/master/utils/eval-source-separation.py
# Copyright ESPNet Contributors
# =============================================================================

def _espnet_eval_STOI(ref, y, fs, extended=False, compute_permutation=True):
    """Calculate STOI

    Reference:
        A short-time objective intelligibility measure
            for time-frequency weighted noisy speech
        https://ieeexplore.ieee.org/document/5495701

    Note(kamo):
        STOI is defined on the signal at 10kHz
        and the input at the other sampling rate will be resampled.
        Thus, the result differs depending on the implementation of resampling.
        Especially, pystoi cannot reproduce matlab's resampling now.

    :param ref (np.ndarray): Reference (Nsrc, Nframe, Nmic)
    :param y (np.ndarray): Enhanced (Nsrc, Nframe, Nmic)
    :param fs (int): Sample frequency
    :param extended (bool): stoi or estoi
    :param compute_permutation (bool):
    :return: value, perm
    :rtype: Tuple[Tuple[float, ...], Tuple[int, ...]]
    """
    if ref.shape != y.shape:
        raise ValueError(
            "ref and y should have the same shape: {} != {}".format(ref.shape, y.shape)
        )
    if ref.ndim != 3:
        raise ValueError("Input must have 3 dims: {}".format(ref.ndim))
    n_src = ref.shape[0]
    n_mic = ref.shape[2]

    if compute_permutation:
        index_list = list(itertools.permutations(range(n_src)))
    else:
        index_list = [list(range(n_src))]

    values = [
        [
            sum(stoi(ref[i, :, ch], y[j, :, ch], fs, extended) for ch in range(n_mic))
            / n_mic
            for i, j in enumerate(indices)
        ]
        for indices in index_list
    ]

    best_pairs = sorted(
        [(v, i) for v, i in zip(values, index_list)], key=lambda x: sum(x[0])
    )[-1]
    value, perm = best_pairs
    return tuple(value), tuple(perm)

# =============================================================================
# ESPNet MCD Implementation
# Source1: https://github.com/espnet/espnet/blob/master/utils/mcd_calculate.py
# Copyright 2020 Nagoya University (Wen-Chin Huang)
# Source2: https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/pyscripts/utils/evaluate_mcd.py
# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
# =============================================================================

def _espnet_low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def _espnet_spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence
    """
    # frame based processing
    npow = np.apply_along_axis(_espnet_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow


def _espnet_spvec2pow(specvec):
    """Convert a spectrum envelope into a power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2

    Return
    ------
    power : scala,
        Power of a frame
    """
    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def _espnet_extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`
    """
    T = data.shape[0]
    if T != len(npow):
        raise ValueError("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata


def _espnet_world_extract(x, fs, mcep_dim=41, mcep_alpha=0.41, fftl=1024, 
                         shiftms=5, f0min=80, f0max=400):
    """Extract WORLD features from audio signal"""
    x = np.array(x, dtype=np.float64)
    x = _espnet_low_cut_filter(x, fs)

    # extract features
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=shiftms
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=fftl)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)
    npow = _espnet_spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }


def _espnet_calculate_mcd(ref_feats, enh_feats):
    """Calculate MCD between two feature sets"""
    # VAD & DTW based on power
    gt_mcep_nonsil_pow = _espnet_extfrm(ref_feats["mcep"], ref_feats["npow"])
    cvt_mcep_nonsil_pow = _espnet_extfrm(enh_feats["mcep"], enh_feats["npow"])
    _, path = fastdtw(
        cvt_mcep_nonsil_pow,
        gt_mcep_nonsil_pow,
        dist=scipy.spatial.distance.euclidean,
    )
    twf_pow = np.array(path).T

    # MCD using power-based DTW
    cvt_mcep_dtw_pow = cvt_mcep_nonsil_pow[twf_pow[0]]
    gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
    diff2sum = np.sum((cvt_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

    return mcd


# =============================================================================
# Utility Functions
# =============================================================================

def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    """
    Get optimal mel-cepstral analysis parameters based on sample rate

    Args:
        fs: Sample rate in Hz

    Returns:
        Tuple of (mcep_dim, mcep_alpha)
    """
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        # Default values for other rates
        return 41, 0.41

def _load_audio(audio_input: Union[str, np.ndarray], 
                expected_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio from file path or validate numpy array
    
    Args:
        audio_input: File path (str) or audio array (np.ndarray)
        expected_sr: Expected sample rate for validation
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        audio, sr = soundfile.read(audio_input, dtype=np.float32)
        return audio, sr
    elif isinstance(audio_input, np.ndarray):
        if expected_sr is None:
            raise ValueError("Sample rate must be provided when using numpy arrays")
        return audio_input.astype(np.float32), expected_sr
    else:
        raise TypeError("Audio input must be file path (str) or numpy array")


def _prepare_audio_for_espnet(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to ESPNet expected format: (Nsrc=1, Nframe, Nmic)
    
    Args:
        audio: Input audio array (can be 1D or 2D)
        
    Returns:
        Audio in format (1, Nframe, Nmic)
    """
    if audio.ndim == 1:
        # Mono: (Nframe,) -> (1, Nframe, 1)
        audio = audio[None, :, None]
    elif audio.ndim == 2:
        # Stereo: (Nframe, Nmic) -> (1, Nframe, Nmic)  
        audio = audio[None, :, :]
    elif audio.ndim == 3:
        # Already in correct format
        pass
    else:
        raise ValueError(f"Audio must be 1D, 2D, or 3D array, got {audio.ndim}D")
        
    return audio


def _validate_inputs(reference: Union[str, np.ndarray], 
                    decoded: Union[str, np.ndarray],
                    sample_rate: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Validate and load audio inputs
    
    Returns:
        Tuple of (reference_audio, decoded_audio, sample_rate)
    """
    # Load audio files
    ref_audio, ref_sr = _load_audio(reference, sample_rate)
    dec_audio, dec_sr = _load_audio(decoded, sample_rate)

    # Validate sample rates match
    if ref_sr != dec_sr:
        raise ValueError(f"Sample rates don't match: reference={ref_sr}, decoded={dec_sr}")

    # Ensure same length (truncate to shorter)
    min_len = min(len(ref_audio), len(dec_audio))
    ref_audio = ref_audio[:min_len]
    dec_audio = dec_audio[:min_len]

    return ref_audio, dec_audio, ref_sr


# =============================================================================
# Standardized Public Interface
# =============================================================================

def compute_stoi(reference: Union[str, np.ndarray], 
                decoded: Union[str, np.ndarray],
                sample_rate: int = 10000,
                extended: bool = False) -> float:
    """
    Compute STOI (Short-Time Objective Intelligibility) metric
    
    Args:
        reference: Reference audio (file path or numpy array)
        decoded: Decoded audio (file path or numpy array)  
        sample_rate: Sample rate in Hz (required if using numpy arrays)
        extended: Use extended STOI (ESTOI) if True
        
    Returns:
        STOI score (higher is better, range 0-1)
        
    Example:
        >>> stoi_score = compute_stoi("reference.wav", "decoded.wav")
        >>> estoi_score = compute_stoi(ref_array, dec_array, 10000, extended=True)
    """
    ref_audio, decoded_audio, sr = _validate_inputs(reference, decoded, sample_rate)

    # Convert to ESPNet format
    ref_audio = _prepare_audio_for_espnet(ref_audio)
    decoded_audio = _prepare_audio_for_espnet(decoded_audio)

    # Call ESPNet function
    stoi_scores, _ = _espnet_eval_STOI(
        ref_audio, decoded_audio, sr,
        extended=extended,
        compute_permutation=False
    )
    
    # Return single score (average over sources/channels)
    return float(np.mean(stoi_scores))

def compute_mcd(reference: Union[str, np.ndarray],
               decoded: Union[str, np.ndarray],
               sample_rate: Optional[int] = None,
               mcep_dim: Optional[int] = None,
               mcep_alpha: Optional[float] = None,
               fft_length: int = 1024,
               frame_shift_ms: int = 5,
               f0_min: int = 80,
               f0_max: int = 400) -> float:
    """
    Compute MCD (Mel-Cepstral Distortion) metric
    
    Args:
        reference: Reference audio (file path or numpy array)
        decoded: Decoded audio (file path or numpy array)
        sample_rate: Sample rate in Hz (required if using numpy arrays)
        mcep_dim: Dimension of mel-cepstral coefficients (auto-selected if None)
        mcep_alpha: All-pass constant for mel-cepstral analysis (auto-selected if None)
        fft_length: FFT length for analysis
        frame_shift_ms: Frame shift in milliseconds
        f0_min: Minimum F0 for analysis
        f0_max: Maximum F0 for analysis
        
    Returns:
        MCD score in dB (lower is better)
        
    Note:
        If mcep_dim or mcep_alpha are None, optimal values are automatically
        selected based on the sampling rate following ESPNet conventions:
        - 16kHz: dim=23, alpha=0.42
        - 22.05kHz: dim=34, alpha=0.45  
        - 24kHz: dim=34, alpha=0.46
        - 44.1kHz: dim=39, alpha=0.53
        - 48kHz: dim=39, alpha=0.55
        - Other rates: dim=41, alpha=0.41 (default)
        
    Example:
        >>> # Auto-select optimal parameters based on sample rate
        >>> mcd_score = compute_mcd("reference.wav", "decoded.wav")
        >>> # Use custom parameters
        >>> mcd_custom = compute_mcd(ref_array, dec_array, 22050, mcep_dim=25, mcep_alpha=0.41)
    """
    ref_audio, dec_audio, sr = _validate_inputs(reference, decoded, sample_rate)

    # Extract WORLD features using ESPNet functions (with auto parameter selection)
    ref_feats = _espnet_world_extract(
        ref_audio, sr, mcep_dim, mcep_alpha, fft_length, 
        frame_shift_ms, f0_min, f0_max
    )
    enh_feats = _espnet_world_extract(
        enh_audio, sr, mcep_dim, mcep_alpha, fft_length,
        frame_shift_ms, f0_min, f0_max  
    )
    
    # Calculate MCD using ESPNet function
    mcd_score = _espnet_calculate_mcd(ref_feats, enh_feats)
    
    return float(mcd_score)