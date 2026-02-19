#!/usr/bin/env python3
"""
Extract speaker embeddings from audio files using ESPnet.
Fixed based on ESPnet Speech2Embedding implementation.
"""

import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging

from espnet2.bin.spk_inference import Speech2Embedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_embeddings(
    audio_root: Path,
    output_root: Path,
    model_name: str = "espnet/voxcelebs12_rawnet3",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1,
):
    """
    Extract speaker embeddings for all audio files.
    
    Args:
        audio_root: Root directory containing audio files
        output_root: Root directory to save embeddings
        model_name: ESPnet model name
        device: Device to run on
        batch_size: Batch size (currently only supports 1)
    """
    # Load ESPnet model
    logger.info(f"Loading ESPnet model: {model_name}")
    model = Speech2Embedding.from_pretrained(
        model_tag=model_name,
        device=device,
    )
    logger.info("Model loaded")
    
    # Find all audio files
    audio_extensions = {'.flac', '.wav', '.mp3'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_root.rglob(f'*{ext}'))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each file
    processed = 0
    skipped = 0
    
    for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
        # Determine output path (same structure, different root)
        rel_path = audio_path.relative_to(audio_root)
        output_path = output_root / rel_path.with_suffix('.npy')
        
        # Skip if already exists
        if output_path.exists():
            skipped += 1
            continue
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Extract embedding
            # Speech2Embedding.__call__ expects numpy array and returns tensor on device
            # Based on ESPnet source: it converts to tensor, moves to device, and returns raw output
            with torch.no_grad():
                embedding = model(audio)  # Returns tensor on self.device (GPU)
            
            # CRITICAL: Move to CPU before converting to numpy
            # ESPnet returns the tensor on whatever device the model is on
            embedding_np = embedding.cpu().numpy()
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embedding_np)
            
            processed += 1
            
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped (already exist): {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings using ESPnet"
    )
    parser.add_argument(
        '--audio_root',
        type=Path,
        required=True,
        help='Root directory containing audio files'
    )
    parser.add_argument(
        '--output_root',
        type=Path,
        required=True,
        help='Root directory to save embeddings'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='espnet/voxcelebs12_rawnet3',
        help='ESPnet model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.audio_root.exists():
        raise ValueError(f"Audio root does not exist: {args.audio_root}")
    
    # Create output directory
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Extract embeddings
    extract_embeddings(
        audio_root=args.audio_root,
        output_root=args.output_root,
        model_name=args.model_name,
        device=args.device,
    )


if __name__ == "__main__":
    main()