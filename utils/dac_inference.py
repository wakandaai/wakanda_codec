import os
import torch
import dac
import numpy as np
from audiotools import AudioSignal
from glob import glob
from tqdm import tqdm
from os.path import basename, join, exists
from argparse import ArgumentParser
from time import time

def process_audio_file(audio_path, model):
    """Process a single audio file with DAC model"""
    # Load audio signal file
    signal = AudioSignal(audio_path)
    
    # Resample to model's sample rate and move to device
    signal.resample(model.sample_rate)
    signal.to(model.device)
    
    # Preprocess and encode
    x = model.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, _, _ = model.encode(x)
    
    # Decode audio signal
    with torch.no_grad():
        y = model.decode(z)
    
    # Convert back to AudioSignal
    output_signal = AudioSignal(y.cpu(), model.sample_rate)
    
    return output_signal

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='.')
    parser.add_argument('--output-dir', required=True, type=str, default='outputs')
    parser.add_argument('--model-type', type=str, default='16khz', 
                       help='DAC model type (16khz, 24khz, 44khz)')
    
    args = parser.parse_args()

    print(f'Loading DAC model: {args.model_type}')
    # Download and load the model
    model_path = dac.utils.download(model_type=args.model_type)
    model = dac.DAC.load(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        print('Using GPU for inference')
    else:
        print('Using CPU for inference')

    wav_dir = args.output_dir
    os.makedirs(wav_dir, exist_ok=True)
    
    # Support wav and flac recursively (same as Mimi script)
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
            # Preserve relative directory structure (same as Mimi script)
            rel_path = os.path.relpath(wav_path, args.input_dir)
            target_wav_path = join(wav_dir, rel_path)

            # Ensure subdir exists
            os.makedirs(os.path.dirname(target_wav_path), exist_ok=True)

            # Process audio file
            output_signal = process_audio_file(wav_path, model)
            
            # Write to file
            output_signal.write(target_wav_path)
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