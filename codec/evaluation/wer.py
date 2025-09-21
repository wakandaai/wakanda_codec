# codec/evaluation/wer.py


"""
MMS ASR for WER evaluation
"""

import torch
import librosa
import soundfile as sf
from transformers import Wav2Vec2ForCTC, AutoProcessor
from jiwer import wer, cer


def load_mms_model(model_name: str = "facebook/mms-1b-all", device: str = "auto"):
    """Load MMS model"""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.to(device).eval()
    
    return {"processor": processor, "model": model, "device": device, "current_lang": None}


def transcribe_audio(mms_dict, audio_path: str, language: str = "eng") -> str:
    """Transcribe audio with MMS"""
    processor = mms_dict["processor"]
    model = mms_dict["model"]
    device = mms_dict["device"]
    
    # Switch language if needed
    if language != mms_dict["current_lang"]:
        processor.tokenizer.set_target_lang(language)
        model.load_adapter(language)
        mms_dict["current_lang"] = language
    
    # Load and process audio
    audio, sr = sf.read(audio_path)
    # resample to 16kHz
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    # if stereo, convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    ids = torch.argmax(outputs, dim=-1)[0]
    return processor.decode(ids).strip()


def compute_wer(model, decoded_path: str, reference_text: str, language: str = "eng") -> float:
    """Compute WER"""
    hypothesis = transcribe_audio(model, decoded_path, language)
    return wer(reference_text.lower(), hypothesis.lower())


def compute_cer(model, decoded_path: str, reference_text: str, language: str = "eng") -> float:
    """Compute CER"""
    hypothesis = transcribe_audio(model, decoded_path, language)
    return cer(reference_text.lower(), hypothesis.lower())