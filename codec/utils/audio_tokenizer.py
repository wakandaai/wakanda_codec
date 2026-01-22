# codec/utils/audio_tokenizer.py

""" Abstract Audio Tokenizer and implementations for DAC, Mimi, and SNAC """

import torch
import numpy as np
import os
from abc import ABC, abstractmethod

class AudioTokenizer(ABC):
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def encode(self, audio_path: str) -> np.ndarray:
        """
        Reads audio and returns discrete codes.
        Returns: np.ndarray of shape (Num_Codebooks, Time_Steps)
        """
        pass

    @staticmethod
    def get_tokenizer(name: str, **kwargs):
        if name.lower() == 'dac':
            return DacTokenizer(**kwargs)
        elif name.lower() == 'mimi':
            return MimiTokenizer(**kwargs)
        elif name.lower() == 'snac':
            return SnacTokenizer(**kwargs)
        else:
            raise ValueError(f"Unknown tokenizer: {name}")

# --- DAC Implementation ---
class DacTokenizer(AudioTokenizer):
    def __init__(self, model_type='16khz', model_path=None, device='cuda'):
        super().__init__(device)
        import dac
        from audiotools import AudioSignal
        
        self.AudioSignal = AudioSignal
        if model_path:
            self.model = dac.DAC.load(model_path)
        else:
            path = dac.utils.download(model_type=model_type)
            self.model = dac.DAC.load(path)
        self.model.to(self.device).eval()

    def encode(self, audio_path):
        signal = self.AudioSignal(audio_path)
        signal.resample(self.model.sample_rate)
        signal.to(self.device)
        
        x = self.model.preprocess(signal.audio_data, signal.sample_rate)
        with torch.no_grad():
            _, codes, _, _, _ = self.model.encode(x)
        
        # DAC output: (B, K, T) -> remove batch
        return codes.squeeze(0).cpu().numpy().astype(np.int16)

# --- Mimi Implementation ---
class MimiTokenizer(AudioTokenizer):
    def __init__(self, model_name='kyutai/mimi', device='cuda'):
        super().__init__(device)
        from transformers import MimiModel, AutoFeatureExtractor
        import librosa
        import soundfile as sf
        
        self.librosa = librosa
        self.sf = sf
        self.model = MimiModel.from_pretrained(model_name).to(self.device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def encode(self, audio_path):
        # Load and Resample
        audio, sr = self.sf.read(audio_path)
        if audio.ndim > 1: audio = self.librosa.to_mono(audio.T)
        
        if sr != self.feature_extractor.sampling_rate:
            audio = self.librosa.resample(audio, orig_sr=sr, target_sr=self.feature_extractor.sampling_rate)
            
        inputs = self.feature_extractor(
            raw_audio=audio, 
            sampling_rate=self.feature_extractor.sampling_rate, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            encoder_outputs = self.model.encode(inputs["input_values"])
            # Mimi quantizes in the bottleneck
            # Note: We need the discrete codes, not the continuous vectors.
            # HuggingFace MimiModel output usually provides `audio_codes` in the encoder output
            # if we look at the source, but standard forward returns quantized_representation.
            # We might need to access the quantizer indices directly.
            
            # Accessing codes from the vector quantization layer:
            # shape: (B, K, T)
            codes = encoder_outputs.audio_codes 
            
        return codes.squeeze(0).cpu().numpy().astype(np.int16)

# --- SNAC (BigCodec) Implementation ---
class SnacTokenizer(AudioTokenizer):
    def __init__(self, model_name='hubertsiuzdak/snac_24khz', device='cuda'):
        super().__init__(device)
        from snac import SNAC
        import librosa
        import soundfile as sf
        
        self.librosa = librosa
        self.sf = sf
        self.model = SNAC.from_pretrained(model_name).to(self.device).eval()
        self.sampling_rate = 24000 # Default for SNAC

    def encode(self, audio_path):
        audio, sr = self.sf.read(audio_path)
        if audio.ndim > 1: audio = self.librosa.to_mono(audio.T)
        if sr != self.sampling_rate:
            audio = self.librosa.resample(audio, orig_sr=sr, target_sr=self.sampling_rate)
            
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        # Add Batch and Channel dims: (1, 1, T)
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        with torch.inference_mode():
            # SNAC forward returns (audio_hat, codes)
            # codes is a list of codebooks (hierarchical) or a tensor depending on version
            # Usually SNAC returns a list of codes at different temporal resolutions.
            _, codes_list = self.model(audio_tensor)
        
        # Normalize SNAC output to (K, T)
        # SNAC often has hierarchical codes (different lengths). 
        # For a standard LM, we usually repeat the coarse codes to match the finest resolution
        # OR we flatten them. 
        # Strategy: Upsample coarse codes to match the longest sequence (finest detail).
        
        # Example logic for hierarchical list:
        # codes_list[0]: (B, 1, T_coarse)
        # codes_list[1]: (B, 1, T_mid)
        # codes_list[2]: (B, 1, T_fine)
        
        target_len = codes_list[-1].shape[-1]
        aligned_codes = []
        
        for i, c in enumerate(codes_list):
            # c shape: (B, N_codebooks_at_layer, T)
            # Repeat to match target_len
            scale_factor = target_len // c.shape[-1]
            if scale_factor > 1:
                c = torch.repeat_interleave(c, scale_factor, dim=-1)
            aligned_codes.append(c)
            
        # Concatenate along codebook dimension
        # Result: (B, Total_Codebooks, T_fine)
        full_codes = torch.cat(aligned_codes, dim=1)
        
        return full_codes.squeeze(0).cpu().numpy().astype(np.int16)