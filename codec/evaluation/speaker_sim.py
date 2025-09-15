#!/usr/bin/env python3

"""
Speaker Similarity Evaluation for Neural Audio Codecs

This module provides speaker similarity metrics using ECAPA-TDNN speaker verification models.
Based on: https://github.com/microsoft/UniSpeech/blob/main/downstreams/speaker_verification/verification.py

The ECAPA-TDNN implementation is adapted from:
https://github.com/lawlict/ECAPA-TDNN
"""

import os
import warnings
from typing import Union, Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from torchaudio.transforms import Resample

# =============================================================================
# ECAPA-TDNN Model Implementation 
# Adapted from https://github.com/lawlict/ECAPA-TDNN
# =============================================================================

class Res2Conv1dReluBn(nn.Module):
    """Res2Conv1d + BatchNorm1d + ReLU"""
    
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    """Conv1d + BatchNorm1d + ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """The SE connection of 1D case"""
    
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(nn.Module):
    """SE-Res2Block of the ECAPA-TDNN architecture"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


class AttentiveStatsPool(nn.Module):
    """Attentive weighted mean and standard deviation pooling"""
    
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN Speaker Verification Model"""
    
    def __init__(self, feat_dim=80, channels=512, emb_dim=192, global_context_att=False,
                 feat_type='fbank', sr=16000, feature_selection="hidden_states", update_extract=False, config_path=None):
        super().__init__()

        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        if feat_type == "fbank" or feat_type == "mfcc":
            self.update_extract = False

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == 'fbank':
            self.feature_extract = trans.MelSpectrogram(
                sample_rate=sr, n_fft=512, win_length=win_len,
                hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                pad=0, n_mels=feat_dim
            )
        elif feat_type == 'mfcc':
            melkwargs = {
                'n_fft': 512,
                'win_length': win_len,
                'hop_length': hop_len,
                'f_min': 0.0,
                'f_max': sr // 2,
                'pad': 0
            }
            self.feature_extract = trans.MFCC(
                sample_rate=sr, n_mfcc=feat_dim, log_mels=False,
                melkwargs=melkwargs
            )
        else:
            # Advanced models using S3PRL
            try:
                if config_path is None:
                    self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)
                else:
                    # This would require UpstreamExpert import - user can implement if needed
                    raise NotImplementedError(f"Config path support for {feat_type} requires UpstreamExpert")
                
                # Handle attention precision issues for large models
                if hasattr(self.feature_extract, 'model') and hasattr(self.feature_extract.model, 'encoder'):
                    if len(self.feature_extract.model.encoder.layers) == 24:
                        if hasattr(self.feature_extract.model.encoder.layers[23].self_attn, "fp32_attention"):
                            self.feature_extract.model.encoder.layers[23].self_attn.fp32_attention = False
                        if hasattr(self.feature_extract.model.encoder.layers[11].self_attn, "fp32_attention"):
                            self.feature_extract.model.encoder.layers[11].self_attn.fp32_attention = False

                self.feat_num = self.get_feat_num()
                self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
                
            except Exception as e:
                warnings.warn(f"Failed to load {feat_type} from S3PRL: {e}. Falling back to fbank.")
                feat_type = 'fbank'
                feat_dim = 40
                self.feat_type = feat_type
                self.feature_extract = trans.MelSpectrogram(
                    sample_rate=sr, n_fft=512, win_length=win_len,
                    hop_length=hop_len, f_min=0.0, f_max=sr // 2,
                    pad=0, n_mels=feat_dim
                )

        # Freeze specific layers for advanced models
        if feat_type != 'fbank' and feat_type != 'mfcc':
            freeze_list = ['final_proj', 'label_embs_concat', 'mask_emb', 'project_q', 'quantizer']
            for name, param in self.feature_extract.named_parameters():
                for freeze_val in freeze_list:
                    if freeze_val in name:
                        param.requires_grad = False
                        break

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)

        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def get_feat_num(self):
        """Get number of features for S3PRL models"""
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select_feature = features[self.feature_selection]
        if isinstance(select_feature, (list, tuple)):
            return len(select_feature)
        else:
            return 1

    def get_feat(self, x):
        """Extract features from audio"""
        if self.update_extract:
            x = self.feature_extract([sample for sample in x])
        else:
            with torch.no_grad():
                if self.feat_type == 'fbank' or self.feat_type == 'mfcc':
                    x = self.feature_extract(x) + 1e-6  # B x feat_dim x time_len
                else:
                    x = self.feature_extract([sample for sample in x])

        if self.feat_type == 'fbank':
            x = x.log()

        if self.feat_type != "fbank" and self.feat_type != "mfcc":
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (norm_weights * x).sum(dim=0)
            x = torch.transpose(x, 1, 2) + 1e-6

        x = self.instance_norm(x)
        return x

    def forward(self, x):
        x = self.get_feat(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        return out


def ECAPA_TDNN_SMALL(feat_dim, emb_dim=256, feat_type='fbank', sr=16000, feature_selection="hidden_states", 
                     update_extract=False, config_path=None):
    """Create small ECAPA-TDNN model"""
    return ECAPA_TDNN(
        feat_dim=feat_dim, 
        channels=512, 
        emb_dim=emb_dim,
        feat_type=feat_type, 
        sr=sr, 
        feature_selection=feature_selection, 
        update_extract=update_extract, 
        config_path=config_path
    )


# =============================================================================
# Model Configuration and Initialization (Preserving Original Functionality)
# =============================================================================

MODEL_LIST = ['ecapa_tdnn', 'hubert_large', 'wav2vec2_xlsr', 'unispeech_sat', "wavlm_base_plus", "wavlm_large"]


def init_model(model_name, use_gpu: bool = True):
    """
    Initialize model exactly as in the original script
    """
    if model_name == 'wavlm_base_plus':
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus', config_path=None)
    elif model_name == 'wavlm_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    elif model_name == 'hubert_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='hubert_large_ll60k', config_path=None)
    elif model_name == 'wav2vec2_xlsr':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wav2vec2_xlsr', config_path=None)
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if use_gpu and torch.cuda.is_available():
        model = model.to('cuda')

    return model


# =============================================================================
# Standardized Interface for Integration
# =============================================================================

def _load_and_preprocess_audio(audio_input: Union[str, np.ndarray], 
                              sample_rate: Optional[int] = None,
                              target_sr: int = 16000):
    """
    Load and preprocess audio for speaker verification
    """
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        wav, sr = sf.read(audio_input)
        wav = torch.from_numpy(wav).unsqueeze(0).float()
    elif isinstance(audio_input, np.ndarray):
        if sample_rate is None:
            raise ValueError("Sample rate must be provided when using numpy arrays")
        wav = torch.from_numpy(audio_input).unsqueeze(0).float()
        sr = sample_rate
    else:
        raise TypeError("Audio input must be file path (str) or numpy array")
    
    if sr != target_sr:
        resample = Resample(orig_freq=sr, new_freq=target_sr)
        wav = resample(wav)
    
    return wav


def compute_speaker_similarity(reference: Union[str, np.ndarray],
                               decoded: Union[str, np.ndarray],
                               model: Any,
                               sample_rate: Optional[int] = 16000) -> float:
    """
    Compute speaker similarity between reference and decoded audio
    
    Args:
        reference: Reference audio (file path or numpy array)
        decoded: decoded/decoded audio (file path or numpy array)
        model_name: Speaker verification model to use (from MODEL_LIST)
        sample_rate: Sample rate (required if using numpy arrays)
        
    Returns:
        Cosine similarity score between speaker embeddings (-1.0 to 1.0)
        Higher values indicate more similar speakers
        
    Example:
        >>> sim_score = compute_speaker_similarity("reference.wav", "decoded.wav")
        >>> print(f"Speaker similarity: {sim_score:.4f}")
        
        >>> # Use advanced model
        >>> sim_wavlm = compute_speaker_similarity(
        ...     "ref.wav", "dec.wav", 
        ...     model_name='wavlm_large'
        ... )
    """  

    # Load and preprocess audio
    ref_audio = _load_and_preprocess_audio(reference, sample_rate, target_sr=sample_rate)
    dec_audio = _load_and_preprocess_audio(decoded, sample_rate, target_sr=sample_rate)

    # Get device from model parameters
    model_device = next(model.parameters()).device
    ref_audio = ref_audio.to(model_device)
    dec_audio = dec_audio.to(model_device)

    # Extract embeddings
    model.eval()
    with torch.no_grad():
        ref_embedding = model(ref_audio)
        dec_embedding = model(dec_audio)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(ref_embedding, dec_embedding)
    return float(similarity.item())

# =============================================================================
# Utility Functions
# =============================================================================

def get_available_speaker_models() -> list:
    """Get list of available speaker verification models"""
    return MODEL_LIST.copy()