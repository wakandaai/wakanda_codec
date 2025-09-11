# evaluate.py

"""
Audio Codec Evaluation CLI

Usage:
    # Generate LibriSpeech manifest first
    python -m codec.utils.librispeech --librispeech-root /path/to/LibriSpeech \
                                     --codec-output-root /path/to/decoded \
                                     --subset dev-clean \
                                     --output manifest.csv

    # Run evaluation
    python evaluate.py --config config/evaluation_en.yaml \
                       --csv-pairs manifest.csv \
                       --output results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

from codec.evaluation.config import load_config
from codec.evaluation.evaluator import DatasetEvaluator


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio codec performance")
    
    # Required arguments
    parser.add_argument("--config", "-c", required=True,
                       help="Path to evaluation configuration file")
    parser.add_argument("--csv-pairs", required=True,
                       help="CSV file with reference,decoded,text columns")
    parser.add_argument("--output", "-o", required=True,
                       help="Output CSV file for results")
    
    # CSV column configuration
    parser.add_argument("--ref-col", default="reference",
                       help="Reference column name in CSV (default: reference)")
    parser.add_argument("--dec-col", default="decoded", 
                       help="Decoded column name in CSV (default: decoded)")
    parser.add_argument("--text-col", default="text",
                       help="Reference text column name in CSV (default: text)")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Load CSV and create file pairs
        logger.info("Loading CSV file...")
        import pandas as pd
        df = pd.read_csv(args.csv_pairs)
        
        # Validate required columns
        required_cols = [args.ref_col, args.dec_col, args.text_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}. Available: {list(df.columns)}")
        
        file_pairs = [
            (row[args.ref_col], row[args.dec_col], row[args.text_col])
            for _, row in df.iterrows()
        ]
        
        if not file_pairs:
            logger.error("No file pairs found!")
            sys.exit(1)
        
        logger.info(f"Found {len(file_pairs)} file pairs to evaluate")
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = DatasetEvaluator(config)
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results_df = evaluator.evaluate_dataset(file_pairs, args.output)
        
        # Print summary
        summary = evaluator.get_summary_stats(results_df)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for metric, stats in summary.items():
            if stats['count'] > 0:
                print(f"\n{metric.upper()}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean:  {stats['mean']:.4f}")
                print(f"  Std:   {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            else:
                print(f"\n{metric.upper()}: No results")
        
        print(f"\nResults saved to: {args.output}")
        
        # Cleanup
        evaluator.cleanup()
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()