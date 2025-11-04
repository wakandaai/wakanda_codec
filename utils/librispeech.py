"""
LibriSpeech Dataset Utilities

Generate CSV manifests for codec evaluation from LibriSpeech datasets.
Treats the root directory as a specific split (e.g., test-clean, dev-clean).
"""

import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class LibriSpeechProcessor:
    """Utility for processing LibriSpeech split into evaluation CSV manifests"""
    
    def __init__(self, split_root: str, codec_output_root: str):
        """
        Initialize LibriSpeech processor
        
        Args:
            split_root: Path to LibriSpeech split directory (e.g., /path/to/test-clean)
            codec_output_root: Path to codec output root directory
        """
        self.split_root = Path(split_root)
        self.codec_output_root = Path(codec_output_root)
        
        if not self.split_root.exists():
            raise FileNotFoundError(f"Split root not found: {split_root}")
        if not self.codec_output_root.exists():
            raise FileNotFoundError(f"Codec output root not found: {codec_output_root}")
    
    def generate_manifest(self, output_csv: str) -> pd.DataFrame:
        """
        Generate CSV manifest for the LibriSpeech split
        
        Args:
            output_csv: Output CSV file path
            
        Returns:
            DataFrame with columns: reference, decoded, text
        """
        logger.info(f"Processing LibriSpeech split: {self.split_root}")
        
        # Step 1: Parse all transcriptions
        logger.info("Parsing transcriptions...")
        transcriptions = self._parse_transcriptions(self.split_root)
        logger.info(f"Found {len(transcriptions)} transcriptions")
        
        # Step 2: Find all audio files and match with transcriptions
        logger.info("Finding audio files...")
        audio_files = self._find_audio_files(self.split_root)
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Step 3: Generate manifest entries
        manifest_entries = []
        missing_transcriptions = []
        missing_decoded = []
        invalid_files = []
        
        for audio_file in audio_files:
            try:
                # Extract utterance ID from filename
                utterance_id = audio_file.stem  # Remove .flac extension
                
                # Get transcription
                if utterance_id not in transcriptions:
                    missing_transcriptions.append(utterance_id)
                    warnings.warn(f"Missing transcription for {utterance_id}")
                    continue
                
                text = transcriptions[utterance_id]
                
                # Find corresponding decoded file
                relative_path = audio_file.relative_to(self.split_root)
                decoded_file = self.codec_output_root / relative_path
                
                # Validate files exist and are readable
                if not self._validate_file(audio_file):
                    invalid_files.append(str(audio_file))
                    continue
                
                if not self._validate_file(decoded_file):
                    missing_decoded.append(str(decoded_file))
                    warnings.warn(f"Missing or unreadable decoded file: {decoded_file}")
                    continue
                
                # Add to manifest
                manifest_entries.append({
                    'reference': str(audio_file.absolute()),
                    'decoded': str(decoded_file.absolute()),
                    'text': text
                })
                
            except Exception as e:
                logger.warning(f"Error processing {audio_file}: {e}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(manifest_entries)
        
        # Log summary
        logger.info(f"Generated manifest with {len(manifest_entries)} entries")
        if missing_transcriptions:
            logger.warning(f"Missing transcriptions: {len(missing_transcriptions)}")
        if missing_decoded:
            logger.warning(f"Missing decoded files: {len(missing_decoded)}")
        if invalid_files:
            logger.warning(f"Invalid reference files: {len(invalid_files)}")
        
        # Save to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Manifest saved to: {output_path}")
        
        return df
    
    def _parse_transcriptions(self, split_path: Path) -> Dict[str, str]:
        """
        Parse all .trans.txt files in the split
        
        Args:
            split_path: Path to LibriSpeech split directory
            
        Returns:
            Dictionary mapping utterance_id -> transcription
        """
        transcriptions = {}
        
        # Find all .trans.txt files
        trans_files = list(split_path.rglob("*.trans.txt"))
        
        for trans_file in trans_files:
            try:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse line: "utterance_id transcription text"
                        parts = line.split(' ', 1)
                        if len(parts) < 2:
                            logger.warning(f"Invalid transcription line in {trans_file}:{line_num}: {line}")
                            continue
                        
                        utterance_id = parts[0]
                        text = parts[1]
                        
                        if utterance_id in transcriptions:
                            logger.warning(f"Duplicate utterance ID: {utterance_id}")
                        
                        transcriptions[utterance_id] = text
                        
            except Exception as e:
                logger.error(f"Error reading transcription file {trans_file}: {e}")
                continue
        
        return transcriptions
    
    def _find_audio_files(self, split_path: Path) -> List[Path]:
        """
        Find all audio files in the split
        
        Args:
            split_path: Path to LibriSpeech split directory
            
        Returns:
            List of audio file paths
        """
        # LibriSpeech uses .flac format
        audio_files = list(split_path.rglob("*.flac"))
        return sorted(audio_files)
    
    def _validate_file(self, file_path: Path) -> bool:
        """
        Validate that a file exists and is readable
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            return file_path.exists() and file_path.is_file() and os.access(file_path, os.R_OK)
        except Exception:
            return False


def generate_librispeech_manifest(split_root: str, 
                                 codec_output_root: str,
                                 output_csv: str) -> pd.DataFrame:
    """
    Convenience function to generate LibriSpeech manifest
    
    Args:
        split_root: Path to LibriSpeech split directory (e.g., /path/to/test-clean)
        codec_output_root: Path to codec output root directory  
        output_csv: Output CSV file path
        
    Returns:
        DataFrame with evaluation manifest
    """
    processor = LibriSpeechProcessor(split_root, codec_output_root)
    return processor.generate_manifest(output_csv)


# CLI interface for the utility
if __name__ == "__main__":
    import argparse
    import sys
    
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    parser = argparse.ArgumentParser(description="Generate LibriSpeech evaluation manifests")
    parser.add_argument("--split-root", required=True,
                       help="Path to LibriSpeech split directory (e.g., /path/to/test-clean)")
    parser.add_argument("--codec-output-root", required=True,
                       help="Path to codec output root directory")
    parser.add_argument("--output", required=True,
                       help="Output CSV file")
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        df = generate_librispeech_manifest(args.split_root, args.codec_output_root, args.output)
        print(f"Generated manifest with {len(df)} entries: {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to generate manifest: {e}")
        sys.exit(1)