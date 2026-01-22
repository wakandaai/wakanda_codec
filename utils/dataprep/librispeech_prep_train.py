# utils/dataprep/librispeech_prep_train.py

"""
Prepare LibriSpeech metadata for Speech LM training.

Generates CSV files with columns:
- code_path: Path to precomputed .npy audio codes
- text: Transcript text
- speaker_id: Speaker identifier
- duration_frames: Number of frames in the audio codes

Usage:
    python utils/dataprep/librispeech_prep_train.py \
        --librispeech-root /path/to/LibriSpeech \
        --codes-root /path/to/precomputed_codes \
        --output-dir data/librispeech_lm \
        --splits train-clean-100 train-clean-360 dev-clean test-clean
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UtteranceInfo:
    """Information about a single utterance"""
    utterance_id: str
    speaker_id: str
    chapter_id: str
    code_path: str
    text: str
    num_frames: int
    num_codebooks: int


def parse_librispeech_path(audio_path: Path) -> Tuple[str, str, str]:
    """
    Parse LibriSpeech path to extract speaker_id, chapter_id, utterance_id.
    
    LibriSpeech structure: {split}/{speaker_id}/{chapter_id}/{speaker_id}-{chapter_id}-{utterance_num}.flac
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (speaker_id, chapter_id, utterance_id)
    """
    # Filename format: {speaker_id}-{chapter_id}-{utterance_num}.flac
    filename = audio_path.stem
    parts = filename.split('-')
    
    if len(parts) >= 3:
        speaker_id = parts[0]
        chapter_id = parts[1]
        utterance_id = filename  # Full filename as utterance_id
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    
    return speaker_id, chapter_id, utterance_id


def load_transcripts(librispeech_split_path: Path) -> Dict[str, str]:
    """
    Load all transcripts from a LibriSpeech split.
    
    Args:
        librispeech_split_path: Path to LibriSpeech split (e.g., train-clean-100)
        
    Returns:
        Dictionary mapping utterance_id -> transcript
    """
    transcripts = {}
    
    trans_files = list(librispeech_split_path.rglob("*.trans.txt"))
    logger.info(f"Found {len(trans_files)} transcript files in {librispeech_split_path.name}")
    
    for trans_file in trans_files:
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    utterance_id = parts[0]
                    transcript = parts[1]
                    transcripts[utterance_id] = transcript
    
    return transcripts


def find_code_files(codes_split_path: Path) -> Dict[str, Path]:
    """
    Find all precomputed code files.
    
    Args:
        codes_split_path: Path to codes directory for a split
        
    Returns:
        Dictionary mapping utterance_id -> code_path
    """
    code_files = {}
    
    for npy_file in codes_split_path.rglob("*.npy"):
        utterance_id = npy_file.stem
        code_files[utterance_id] = npy_file
    
    return code_files


def get_code_info(code_path: Path) -> Tuple[int, int]:
    """
    Get shape information from a code file.
    
    Args:
        code_path: Path to .npy file
        
    Returns:
        Tuple of (num_codebooks, num_frames)
    """
    codes = np.load(code_path)
    
    # Expected shape: (num_codebooks, num_frames) or (num_frames, num_codebooks)
    if codes.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {codes.shape}")
    
    # Assume (num_codebooks, num_frames) where num_codebooks is typically 8-12
    if codes.shape[0] < codes.shape[1]:
        num_codebooks, num_frames = codes.shape
    else:
        # Might be transposed
        num_frames, num_codebooks = codes.shape
        logger.warning(f"Code file may be transposed: {code_path}, shape={codes.shape}")
    
    return num_codebooks, num_frames


def process_split(
    librispeech_root: Path,
    codes_root: Path,
    split_name: str,
    min_frames: int = 10,
    max_frames: int = 3000,
) -> List[UtteranceInfo]:
    """
    Process a single LibriSpeech split.
    
    Args:
        librispeech_root: Root of LibriSpeech dataset
        codes_root: Root of precomputed codes
        split_name: Name of the split (e.g., "train-clean-100")
        min_frames: Minimum number of frames to include
        max_frames: Maximum number of frames to include
        
    Returns:
        List of UtteranceInfo objects
    """
    librispeech_split = librispeech_root / split_name
    codes_split = codes_root / split_name
    
    if not librispeech_split.exists():
        logger.warning(f"LibriSpeech split not found: {librispeech_split}")
        return []
    
    if not codes_split.exists():
        logger.warning(f"Codes split not found: {codes_split}")
        return []
    
    # Load transcripts
    transcripts = load_transcripts(librispeech_split)
    logger.info(f"Loaded {len(transcripts)} transcripts for {split_name}")
    
    # Find code files
    code_files = find_code_files(codes_split)
    logger.info(f"Found {len(code_files)} code files for {split_name}")

    assert len(code_files) == len(transcripts), \
        f"Mismatch between code files ({len(code_files)}) and transcripts ({len(transcripts)})"
    
    # Match and create utterance info
    utterances = []
    skipped_no_transcript = 0
    skipped_no_codes = 0
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_error = 0
    
    for utterance_id, code_path in tqdm(code_files.items(), desc=f"Processing {split_name}"):
        # Check transcript exists
        if utterance_id not in transcripts:
            skipped_no_transcript += 1
            continue
        
        try:
            # Parse speaker and chapter from utterance_id
            parts = utterance_id.split('-')
            if len(parts) >= 2:
                speaker_id = parts[0]
                chapter_id = parts[1]
            else:
                logger.warning(f"Could not parse utterance_id: {utterance_id}")
                skipped_error += 1
                continue
            
            # Get code info
            num_codebooks, num_frames = get_code_info(code_path)
            
            # Filter by duration
            if num_frames < min_frames:
                skipped_too_short += 1
                continue
            if num_frames > max_frames:
                skipped_too_long += 1
                continue
            
            utterance = UtteranceInfo(
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                code_path=str(code_path.absolute()),
                text=transcripts[utterance_id],
                num_frames=num_frames,
                num_codebooks=num_codebooks,
            )
            utterances.append(utterance)
            
        except Exception as e:
            logger.warning(f"Error processing {utterance_id}: {e}")
            skipped_error += 1
            continue
    
    logger.info(f"Split {split_name} results:")
    logger.info(f"  Valid utterances: {len(utterances)}")
    logger.info(f"  Skipped (no transcript): {skipped_no_transcript}")
    logger.info(f"  Skipped (no codes): {skipped_no_codes}")
    logger.info(f"  Skipped (too short): {skipped_too_short}")
    logger.info(f"  Skipped (too long): {skipped_too_long}")
    logger.info(f"  Skipped (error): {skipped_error}")
    
    return utterances


def create_dataframe(utterances: List[UtteranceInfo]) -> pd.DataFrame:
    """
    Create DataFrame from utterance list.
    """
    data = [
        {
            "utterance_id": u.utterance_id,
            "speaker_id": u.speaker_id,
            "chapter_id": u.chapter_id,
            "code_path": u.code_path,
            "text": u.text,
            "num_frames": u.num_frames,
            "num_codebooks": u.num_codebooks,
        }
        for u in utterances
    ]
    return pd.DataFrame(data)


def compute_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute dataset statistics.
    """
    stats = {
        "num_utterances": len(df),
        "num_speakers": df["speaker_id"].nunique(),
        "num_chapters": df["chapter_id"].nunique(),
        "total_frames": df["num_frames"].sum(),
        "mean_frames": df["num_frames"].mean(),
        "std_frames": df["num_frames"].std(),
        "min_frames": df["num_frames"].min(),
        "max_frames": df["num_frames"].max(),
        "utterances_per_speaker": df.groupby("speaker_id").size().describe().to_dict(),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech metadata for Speech LM")
    
    parser.add_argument("--librispeech-root", type=str, required=True,
                       help="Root directory of LibriSpeech dataset")
    parser.add_argument("--codes-root", type=str, required=True,
                       help="Root directory of precomputed audio codes")
    parser.add_argument("--output-dir", type=str, default="data/librispeech_lm",
                       help="Output directory for CSV files")
    parser.add_argument("--splits", type=str, nargs="+",
                       default=["train-clean-100", "train-clean-360", "dev-clean", "test-clean"],
                       help="LibriSpeech splits to process")
    parser.add_argument("--train-splits", type=str, nargs="+",
                       default=["train-clean-100", "train-clean-360"],
                       help="Splits to combine into train.csv")
    parser.add_argument("--dev-splits", type=str, nargs="+",
                       default=["dev-clean"],
                       help="Splits to combine into dev.csv")
    parser.add_argument("--test-splits", type=str, nargs="+",
                       default=["test-clean"],
                       help="Splits to combine into test.csv")
    parser.add_argument("--min-frames", type=int, default=10,
                       help="Minimum number of frames")
    parser.add_argument("--max-frames", type=int, default=7500,
                       help="Maximum number of frames")
    
    args = parser.parse_args()
    
    librispeech_root = Path(args.librispeech_root)
    codes_root = Path(args.codes_root)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all splits
    all_utterances = {}
    for split_name in args.splits:
        utterances = process_split(
            librispeech_root,
            codes_root,
            split_name,
            min_frames=args.min_frames,
            max_frames=args.max_frames,
        )
        all_utterances[split_name] = utterances
        
        # Save individual split CSV
        if utterances:
            df = create_dataframe(utterances)
            split_csv = output_dir / f"{split_name}.csv"
            df.to_csv(split_csv, index=False)
            logger.info(f"Saved {split_csv} with {len(df)} utterances")
    
    # Combine into train/dev/test
    combined = {
        "train": [],
        "dev": [],
        "test": [],
    }
    
    for split_name in args.train_splits:
        if split_name in all_utterances:
            combined["train"].extend(all_utterances[split_name])
    
    for split_name in args.dev_splits:
        if split_name in all_utterances:
            combined["dev"].extend(all_utterances[split_name])
    
    for split_name in args.test_splits:
        if split_name in all_utterances:
            combined["test"].extend(all_utterances[split_name])
    
    # Save combined CSVs and statistics
    import json
    
    for set_name, utterances in combined.items():
        if not utterances:
            continue
        
        df = create_dataframe(utterances)
        csv_path = output_dir / f"{set_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path} with {len(df)} utterances")
        
        # Compute and save statistics
        stats = compute_statistics(df)
        stats_path = output_dir / f"{set_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"Saved statistics to {stats_path}")
        
        # Print summary
        print(f"\n{set_name.upper()} SET SUMMARY:")
        print(f"  Utterances: {stats['num_utterances']}")
        print(f"  Speakers: {stats['num_speakers']}")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Mean frames/utterance: {stats['mean_frames']:.1f}")
    
    logger.info(f"\nData preparation complete! Files saved to: {output_dir}")


if __name__ == "__main__":
    main()