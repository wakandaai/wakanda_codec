"""
Summarize Results

Usage:
    python scripts/summarize_results.py --results-dir results/ --output comparison.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_metric_csv(csv_file: Path) -> Dict:
    """
    Load a single metric CSV and compute statistics
    
    Args:
        csv_file: Path to CSV file
        
    Returns:
        Dictionary with mean, std, count statistics
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Find the metric column (assuming it's the last column or has a specific pattern)
        metric_columns = [col for col in df.columns if col not in ['reference_path', 'decoded_path', 'reference_text']]
        
        if not metric_columns:
            logger.warning(f"No metric columns found in {csv_file}")
            return {'mean': np.nan, 'count': 0}
        
        # Use the first metric column (or the one that matches the filename)
        metric_col = metric_columns[0]
        
        # For files like pesq_wb.csv, pesq_nb.csv, prefer the column that matches
        filename_stem = csv_file.stem.lower()
        for col in metric_columns:
            if filename_stem.replace('_', '').replace('-', '') in col.lower().replace('_', '').replace('-', ''):
                metric_col = col
                break
        
        values = df[metric_col].dropna()
        
        if len(values) == 0:
            return {'mean': np.nan, 'std': np.nan, 'count': 0}
        
        return {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'count': int(len(values)),
            'min': float(values.min()),
            'max': float(values.max())
        }
        
    except Exception as e:
        logger.warning(f"Failed to load {csv_file}: {e}")
        return {'mean': np.nan, 'std': np.nan, 'count': 0}


def load_model_results(results_dir: Path) -> Dict[str, Dict]:
    """
    Load results from all model directories
    
    Args:
        results_dir: Root results directory containing model subdirs
        
    Returns:
        Dictionary mapping model_name -> metric_name -> stats
    """
    model_results = {}
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        model_results[model_name] = {}
        
        # Find all CSV files in the model directory
        csv_files = list(model_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found for model: {model_name}")
            continue
        
        for csv_file in csv_files:
            metric_name = csv_file.stem  # filename without extension
            stats = load_metric_csv(csv_file)
            model_results[model_name][metric_name] = stats
            
            if stats['count'] > 0:
                logger.info(f"Loaded {metric_name} for {model_name}: mean={stats['mean']:.4f}, count={stats['count']}")
            else:
                logger.warning(f"No valid data for {metric_name} in {model_name}")
    
    return model_results


def create_comparison_table(model_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table with models as rows and metrics as columns
    
    Args:
        model_results: Dictionary of model results
        
    Returns:
        DataFrame with comparison results
    """
    if not model_results:
        return pd.DataFrame()
    
    # Get all unique metrics across all models
    all_metrics = set()
    for stats in model_results.values():
        all_metrics.update(stats.keys())
    
    all_metrics = sorted(all_metrics)
    
    # Create comparison data
    comparison_data = []
    
    for model_name, model_stats in model_results.items():
        row = {'model': model_name}
        
        for metric in all_metrics:
            if metric in model_stats:
                stats = model_stats[metric]
                if stats['count'] > 0 and not np.isnan(stats['mean']):
                    # Include mean ± std format
                    mean = stats['mean']
                    std = stats['std']
                    count = stats['count']
                    
                    row[f'{metric}_mean'] = mean
                    row[f'{metric}_std'] = std
                    row[f'{metric}_count'] = count
                    row[f'{metric}_formatted'] = f"{mean:.4f} ± {std:.4f} (n={count})"
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_count'] = 0
                    row[f'{metric}_formatted'] = "No results"
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_count'] = 0
                row[f'{metric}_formatted'] = "Not evaluated"
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def create_summary_table(df_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean summary table with just means for easy reading
    
    Args:
        df_comparison: Full comparison DataFrame
        
    Returns:
        Clean summary DataFrame
    """
    if df_comparison.empty:
        return pd.DataFrame()
    
    # Get metric names (columns ending with '_mean')
    mean_cols = [col for col in df_comparison.columns if col.endswith('_mean')]
    metrics = [col.replace('_mean', '') for col in mean_cols]
    
    # Create summary with just model and mean values
    summary_data = []
    for _, row in df_comparison.iterrows():
        summary_row = {'model': row['model']}
        for metric in metrics:
            mean_col = f'{metric}_mean'
            if mean_col in row and pd.notna(row[mean_col]):
                summary_row[metric] = f"{row[mean_col]:.4f}"
            else:
                summary_row[metric] = "—"
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results across models")
    
    parser.add_argument("--results-dir", "-r", required=True,
                       help="Root directory containing model result subdirectories")
    parser.add_argument("--output", "-o", required=True,
                       help="Output CSV file for comparison results")
    parser.add_argument("--summary-only", action="store_true",
                       help="Generate only clean summary table (means only)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        
        # Load all model results
        logger.info(f"Loading results from: {results_dir}")
        model_results = load_model_results(results_dir)
        
        if not model_results:
            logger.error("No valid model results found!")
            return
        
        logger.info(f"Found results for {len(model_results)} models: {list(model_results.keys())}")
        
        # Print discovered metrics
        all_metrics = set()
        for stats in model_results.values():
            all_metrics.update(stats.keys())
        logger.info(f"Discovered metrics: {sorted(all_metrics)}")
        
        # Create comparison table
        df_comparison = create_comparison_table(model_results)
        
        if args.summary_only:
            # Save only clean summary
            df_summary = create_summary_table(df_comparison)
            df_summary.to_csv(args.output, index=False)
            logger.info(f"Summary comparison saved to: {args.output}")
            
            # Print summary to console
            print("\n" + "="*80)
            print("MODEL COMPARISON SUMMARY")
            print("="*80)
            print(df_summary.to_string(index=False))
            
        else:
            # Save full comparison with detailed stats
            df_comparison.to_csv(args.output, index=False)
            logger.info(f"Detailed comparison saved to: {args.output}")
            
            # Also save summary version
            summary_file = Path(args.output).with_suffix('.summary.csv')
            df_summary = create_summary_table(df_comparison)
            df_summary.to_csv(summary_file, index=False)
            logger.info(f"Summary version saved to: {summary_file}")
            
            # Print summary to console
            print("\n" + "="*80)
            print("MODEL COMPARISON SUMMARY")
            print("="*80)
            print(df_summary.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()