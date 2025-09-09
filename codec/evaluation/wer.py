# codec/evaluation/wer.py

#!/usr/bin/env python3

"""
WER (Word Error Rate) Evaluation using Whisper ASR

Simple script to compute WER between reference text and Whisper transcriptions.
"""

import os
import warnings
from typing import Union, Optional, List

import torch
import soundfile as sf
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from jiwer import wer, cer


def load_whisper_model(model_name: str = "openai/whisper-large-v3", device: str = "auto"):
    """Load Whisper model and create ASR pipeline"""
    
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe


def transcribe_audio(pipe, audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio file using Whisper pipeline"""
    
    generate_kwargs = {}
    if language:
        generate_kwargs["language"] = language
    
    result = pipe(audio_path, generate_kwargs=generate_kwargs)
    return result["text"].strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis"""
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis"""
    return cer(reference, hypothesis)
