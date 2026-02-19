#!/usr/bin/env python3
"""
Prepare dataset CSV files from pre-extracted codes, speaker embeddings, and transcripts.
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_librispeech_transcripts(librispeech_root: Path) -> dict:
    """
    Load LibriSpeech transcripts from all trans.txt files.
    
    Returns:
        dict: {utterance_id: transcript_text}
    """
    transcripts = {}
    
    # Find all trans.txt files
    trans_files = list(librispeech_root.rglob('*.trans.txt'))
    
    logger.info(f"Found {len(trans_files)} transcript files")
    
    for trans_file in tqdm(trans_files, desc="Loading transcripts"):
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text
    
    logger.info(f"Loaded {len(transcripts)} transcripts")
    return transcripts


def create_dataset_csv(
    codes_root: Path,
    spk_embeds_root: Path,
    librispeech_root: Path,
    output_csv: Path,
    partition: str = None,
):
    """
    Create CSV mapping codes, speaker embeddings, and text.
    
    Args:
        codes_root: data/codes/bigcodec/LibriSpeech
        spk_embeds_root: data/spk_embds/espnet_ecapa/LibriSpeech
        librispeech_root: data/LibriSpeech
        output_csv: output CSV path
        partition: e.g., 'train-clean-100', 'dev-clean', or None for all
    """
    # Load transcripts
    transcripts = load_librispeech_transcripts(librispeech_root)
    
    # Find all code files
    if partition:
        code_pattern = f"{partition}/**/*.npy"
    else:
        code_pattern = "**/*.npy"
    
    code_files = list(codes_root.glob(code_pattern))
    logger.info(f"Found {len(code_files)} code files")
    
    rows = []
    missing_spk = 0
    missing_text = 0
    
    for code_path in tqdm(code_files, desc="Processing files"):
        # Extract utterance ID from path
        # e.g., .../84/121123/84-121123-0000.npy -> 84-121123-0000
        utt_id = code_path.stem
        
        # Find corresponding speaker embedding
        rel_path = code_path.relative_to(codes_root)
        spk_path = spk_embeds_root / rel_path
        
        if not spk_path.exists():
            missing_spk += 1
            continue
        
        # Get text
        text = transcripts.get(utt_id)
        if not text:
            missing_text += 1
            continue
        
        rows.append({
            'path_to_codes': str(code_path.absolute()),
            'path_to_spk_embed': str(spk_path.absolute()),
            'text': text
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    
    logger.info(f"Created {output_csv} with {len(df)} samples")
    logger.info(f"Missing speaker embeddings: {missing_spk}")
    logger.info(f"Missing transcripts: {missing_text}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TTS dataset CSV from codes, speaker embeddings, and transcripts"
    )
    parser.add_argument(
        '--codes_root',
        type=Path,
        required=True,
        help='Root directory for codes (e.g., data/codes/bigcodec/LibriSpeech)'
    )
    parser.add_argument(
        '--spk_embeds_root',
        type=Path,
        required=True,
        help='Root directory for speaker embeddings (e.g., data/spk_embds/espnet_ecapa/LibriSpeech)'
    )
    parser.add_argument(
        '--librispeech_root',
        type=Path,
        required=True,
        help='Root directory for LibriSpeech audio and transcripts (e.g., data/LibriSpeech)'
    )
    parser.add_argument(
        '--output_csv',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default=None,
        help='Specific partition (e.g., train-clean-100, dev-clean) or None for all'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.codes_root.exists():
        raise ValueError(f"Codes root does not exist: {args.codes_root}")
    
    if not args.spk_embeds_root.exists():
        raise ValueError(f"Speaker embeddings root does not exist: {args.spk_embeds_root}")
    
    if not args.librispeech_root.exists():
        raise ValueError(f"LibriSpeech root does not exist: {args.librispeech_root}")
    
    # Create output directory if needed
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV
    create_dataset_csv(
        codes_root=args.codes_root,
        spk_embeds_root=args.spk_embeds_root,
        librispeech_root=args.librispeech_root,
        output_csv=args.output_csv,
        partition=args.partition,
    )


if __name__ == "__main__":
    main()