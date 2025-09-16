import os
import librosa
import torch
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join, exists
from argparse import ArgumentParser
from time import time
from transformers import MimiModel, AutoFeatureExtractor

def process_audio_file(audio_path, model, feature_extractor):
    """Process a single audio file with Mimi model"""
    # Load the audio sample
    audio_sample, original_sr = sf.read(audio_path)
    
    # Handle stereo to mono conversion if needed
    if audio_sample.ndim > 1:
        audio_sample = librosa.to_mono(audio_sample.T)
    
    # Resample if needed
    if original_sr != feature_extractor.sampling_rate:
        audio_sample = librosa.resample(
            audio_sample, 
            orig_sr=original_sr, 
            target_sr=feature_extractor.sampling_rate
        )
    
    # Pre-process the inputs
    inputs = feature_extractor(
        raw_audio=audio_sample, 
        sampling_rate=feature_extractor.sampling_rate, 
        return_tensors="pt"
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Get reconstructed audio
    with torch.no_grad():
        audio_values = model(inputs["input_values"]).audio_values
    
    # Fix the tensor shape for saving
    # Remove batch dimension and channel dimension if present
    audio_to_save = audio_values.squeeze().detach().cpu().numpy()
    
    # Ensure it's 1D for mono audio
    if audio_to_save.ndim > 1:
        audio_to_save = audio_to_save.flatten()
    
    return audio_to_save

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='.')
    parser.add_argument('--output-dir', required=True, type=str, default='outputs')
    parser.add_argument('--model-name', type=str, default='kyutai/mimi', 
                       help='Mimi model name from HuggingFace')
    
    args = parser.parse_args()

    print(f'Loading Mimi model: {args.model_name}')
    # Load the model + feature extractor
    model = MimiModel.from_pretrained(args.model_name)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print('Using GPU for inference')
    else:
        print('Using CPU for inference')
    
    model = model.eval()

    wav_dir = args.output_dir
    os.makedirs(wav_dir, exist_ok=True)
    
    # Support wav and flac recursively (same as BigCodec)
    wav_paths = glob(join(args.input_dir, '**', '*.wav'), recursive=True) \
            + glob(join(args.input_dir, '**', '*.flac'), recursive=True)
    print(f'Found {len(wav_paths)} audio files in {args.input_dir}')

    if len(wav_paths) == 0:
        print("No audio files found! Please check your input directory.")
        exit(1)

    st = time()
    successful = 0
    failed = 0
    
    for wav_path in tqdm(wav_paths, desc="Processing audio files"):
        try:
            # Preserve relative directory structure (same as BigCodec)
            rel_path = os.path.relpath(wav_path, args.input_dir)
            target_wav_path = join(wav_dir, rel_path)

            # Ensure subdir exists
            os.makedirs(os.path.dirname(target_wav_path), exist_ok=True)

            # Process audio file
            reconstructed_audio = process_audio_file(wav_path, model, feature_extractor)
            
            # Save audio with correct sample rate
            sf.write(target_wav_path, reconstructed_audio, feature_extractor.sampling_rate)
            successful += 1
            
        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            failed += 1
            continue

    et = time()
    print(f'Inference completed!')
    print(f'Successfully processed: {successful} files')
    print(f'Failed: {failed} files')
    print(f'Total time: {(et-st)/60:.2f} mins')
    print(f'Average time per file: {(et-st)/len(wav_paths):.2f} seconds')