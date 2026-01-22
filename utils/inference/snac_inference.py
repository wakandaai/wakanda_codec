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
from snac import SNAC

def process_audio_file(audio_path, model, sampling_rate=24000):
    """Process a single audio file with SNAC model"""
    # Load the audio sample
    audio_sample, original_sr = sf.read(audio_path)
    
    # Handle stereo to mono conversion if needed
    if audio_sample.ndim > 1:
        audio_sample = librosa.to_mono(audio_sample.T)
    
    # Resample if needed
    if original_sr != sampling_rate:
        audio_sample = librosa.resample(
            audio_sample, 
            orig_sr=original_sr, 
            target_sr=sampling_rate
        )
    
    # Convert to torch tensor and move to GPU
    audio_tensor = torch.from_numpy(audio_sample).float()
    if torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()
    
    # Ensure correct shape: (B, C, T)
    if len(audio_tensor.shape) == 1:
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    elif len(audio_tensor.shape) == 2:
        audio_tensor = audio_tensor.unsqueeze(1)  # (B, 1, T)
    
    # Get reconstructed audio
    with torch.inference_mode():
        audio_hat, codes = model(audio_tensor)
    
    # Fix the tensor shape for saving
    # Remove batch dimension and channel dimension if present
    audio_to_save = audio_hat.squeeze().detach().cpu().numpy()
    
    # Ensure it's 1D for mono audio
    if audio_to_save.ndim > 1:
        audio_to_save = audio_to_save.flatten()
    
    return audio_to_save

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='.')
    parser.add_argument('--output-dir', required=True, type=str, default='outputs')
    parser.add_argument('--model-name', type=str, default='hubertsiuzdak/snac_24khz', 
                       help='SNAC model name from HuggingFace')
    parser.add_argument('--sampling-rate', type=int, default=24000,
                       help='Target sampling rate for audio processing')
    
    args = parser.parse_args()

    print(f'Loading SNAC model: {args.model_name}')
    # Load the model
    model = SNAC.from_pretrained(args.model_name).eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print('Using GPU for inference')
    else:
        print('Using CPU for inference')

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
            reconstructed_audio = process_audio_file(wav_path, model, args.sampling_rate)
            
            # Save audio with correct sample rate
            sf.write(target_wav_path, reconstructed_audio, args.sampling_rate)
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