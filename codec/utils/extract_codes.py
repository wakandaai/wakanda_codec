# codec/utils/extract_codes.py

import argparse
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from codec.utils.audio_tokenizer import AudioTokenizer

def main():
    parser = argparse.ArgumentParser(description="Precompute speech tokens for training.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input audio folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save .npy files")
    parser.add_argument("--codec", type=str, required=True, choices=['dac', 'mimi', 'snac'], help="Which codec to use")
    parser.add_argument("--model_name", type=str, default=None, help="Optional model name/path override")
    args = parser.parse_args()

    # Initialize Tokenizer
    kwargs = {}
    if args.model_name:
        kwargs['model_name'] = args.model_name
        # Handle parameter naming differences if necessary
        if args.codec == 'dac':
             kwargs = {'model_type': args.model_name}
    
    print(f"Initializing {args.codec}...")
    tokenizer = AudioTokenizer.get_tokenizer(args.codec, **kwargs)

    # Find Files
    extensions = ['*.wav', '*.flac', '*.mp3']
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    
    print(f"Found {len(audio_files)} files.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Process
    success = 0
    for file_path in tqdm(audio_files):
        try:
            # Generate output path (preserving structure could be added, here flat for simplicity)
            # Or preserve structure:
            rel_path = os.path.relpath(file_path, args.input_dir)
            base_name = os.path.splitext(rel_path)[0]
            save_path = os.path.join(args.output_dir, f"{base_name}.npy")
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            if os.path.exists(save_path):
                continue
                
            # Encode
            codes = tokenizer.encode(file_path)
            
            # Save: Shape (K, T)
            np.save(save_path, codes)
            success += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Finished. Processed {success}/{len(audio_files)} files.")

if __name__ == "__main__":
    main()