# codec/evaluation/file_discovery.py

"""
File Discovery and Pairing for Dataset Evaluation

Handles finding and pairing reference and decoded audio files from:
1. Directory structures with matching filenames
2. CSV files with explicit path pairs
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class FileDiscovery:
    """Utilities for discovering and pairing audio files for evaluation"""
    
    # Common audio file extensions
    AUDIO_EXTENSIONS = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aiff', '.au'}
    
    @staticmethod
    def from_directories(ref_dir: str, dec_dir: str, 
                        pattern: Optional[str] = None,
                        recursive: bool = True) -> List[Tuple[str, str]]:
        """
        Discover file pairs from reference and decoded directories
        
        Args:
            ref_dir: Directory containing reference audio files
            dec_dir: Directory containing decoded audio files  
            pattern: Optional filename pattern to match (glob style)
            recursive: Whether to search subdirectories
            
        Returns:
            List of (reference_path, decoded_path) tuples
        """
        ref_path = Path(ref_dir)
        dec_path = Path(dec_dir)
        
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference directory not found: {ref_dir}")
        if not dec_path.exists():
            raise FileNotFoundError(f"Decoded directory not found: {dec_dir}")
        
        # Find all audio files in reference directory
        if recursive:
            ref_files = FileDiscovery._find_audio_files_recursive(ref_path, pattern)
        else:
            ref_files = FileDiscovery._find_audio_files_flat(ref_path, pattern)
        
        pairs = []
        not_found = []
        
        for ref_file in ref_files:
            # Calculate relative path from ref_dir
            rel_path = ref_file.relative_to(ref_path)
            
            # Look for corresponding decoded file
            dec_file = dec_path / rel_path
            
            if dec_file.exists():
                pairs.append((str(ref_file), str(dec_file)))
            else:
                not_found.append(str(rel_path))
        
        if not_found:
            logger.warning(f"Could not find decoded files for {len(not_found)} references:")
            for missing in not_found[:10]:  # Log first 10
                logger.warning(f"  Missing: {missing}")
            if len(not_found) > 10:
                logger.warning(f"  ... and {len(not_found) - 10} more")
        
        logger.info(f"Found {len(pairs)} file pairs from directories")
        return pairs
    
    @staticmethod
    def from_csv(csv_path: str, 
                ref_col: str = 'reference',
                dec_col: str = 'decoded') -> List[Tuple[str, str]]:
        """
        Load file pairs from CSV file
        
        Args:
            csv_path: Path to CSV file containing file pairs
            ref_col: Column name for reference file paths
            dec_col: Column name for decoded file paths
            
        Returns:
            List of (reference_path, decoded_path) tuples
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file {csv_path}: {e}")
        
        # Validate required columns exist
        if ref_col not in df.columns:
            raise ValueError(f"Reference column '{ref_col}' not found in CSV. Available: {list(df.columns)}")
        if dec_col not in df.columns:
            raise ValueError(f"Decoded column '{dec_col}' not found in CSV. Available: {list(df.columns)}")
        
        pairs = []
        missing_files = []
        
        for idx, row in df.iterrows():
            ref_path = str(row[ref_col])
            dec_path = str(row[dec_col])
            
            # Check if files exist
            ref_exists = os.path.exists(ref_path)
            dec_exists = os.path.exists(dec_path)
            
            if ref_exists and dec_exists:
                pairs.append((ref_path, dec_path))
            else:
                if not ref_exists:
                    missing_files.append(f"Row {idx}: Reference file not found: {ref_path}")
                if not dec_exists:
                    missing_files.append(f"Row {idx}: Decoded file not found: {dec_path}")
        
        if missing_files:
            logger.warning(f"Found {len(missing_files)} missing files:")
            for missing in missing_files[:10]:  # Log first 10
                logger.warning(f"  {missing}")
            if len(missing_files) > 10:
                logger.warning(f"  ... and {len(missing_files) - 10} more")
        
        logger.info(f"Loaded {len(pairs)} file pairs from CSV")
        return pairs
    
    @staticmethod
    def _find_audio_files_recursive(directory: Path, pattern: Optional[str] = None) -> List[Path]:
        """Find audio files recursively in directory"""
        if pattern:
            files = list(directory.rglob(pattern))
        else:
            files = []
            for ext in FileDiscovery.AUDIO_EXTENSIONS:
                files.extend(directory.rglob(f"*{ext}"))
        
        # Filter to only audio files and sort
        audio_files = [f for f in files if f.suffix.lower() in FileDiscovery.AUDIO_EXTENSIONS]
        return sorted(audio_files)
    
    @staticmethod
    def _find_audio_files_flat(directory: Path, pattern: Optional[str] = None) -> List[Path]:
        """Find audio files in directory (non-recursive)"""
        if pattern:
            files = list(directory.glob(pattern))
        else:
            files = []
            for ext in FileDiscovery.AUDIO_EXTENSIONS:
                files.extend(directory.glob(f"*{ext}"))
        
        # Filter to only audio files and sort
        audio_files = [f for f in files if f.suffix.lower() in FileDiscovery.AUDIO_EXTENSIONS]
        return sorted(audio_files)
    
    @staticmethod
    def validate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Validate that all file pairs exist and are readable
        
        Args:
            pairs: List of (reference_path, decoded_path) tuples
            
        Returns:
            List of valid pairs (subset of input)
        """
        valid_pairs = []
        
        for ref_path, dec_path in pairs:
            try:
                # Check if files exist and are readable
                if os.path.exists(ref_path) and os.access(ref_path, os.R_OK):
                    if os.path.exists(dec_path) and os.access(dec_path, os.R_OK):
                        valid_pairs.append((ref_path, dec_path))
                    else:
                        logger.warning(f"Decoded file not accessible: {dec_path}")
                else:
                    logger.warning(f"Reference file not accessible: {ref_path}")
            except Exception as e:
                logger.warning(f"Error validating pair ({ref_path}, {dec_path}): {e}")
        
        logger.info(f"Validated {len(valid_pairs)}/{len(pairs)} file pairs")
        return valid_pairs