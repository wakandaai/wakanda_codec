# codec/evaluation/evaluator.py

"""
Dataset-level evaluation for neural audio codecs

Sequential metric processing with improved results reporting
"""

import logging
import time
from pathlib import Path
from typing import List, Literal, Tuple, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import torch
import torchaudio

from codec.evaluation.model_manager import ModelManager
from codec.evaluation.config import validate_config, get_enabled_metrics
from codec.evaluation.speaker_sim import compute_speaker_similarity
from codec.evaluation.espnet import compute_stoi
from codec.evaluation.mcd import compute_mcd
from codec.evaluation.torchmetrics import compute_PESQ, compute_NISQA, compute_DNSMOS
from codec.evaluation.wer import compute_wer, transcribe_audio

logger = logging.getLogger(__name__)


class DatasetEvaluator:
    """Main class for evaluating audio codec performance on datasets"""
    
    def __init__(self, config: Dict[str, Any], model_name: Optional[str] = None):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Configuration dictionary specifying metrics and parameters
            model_name: Name of the model being evaluated (for organized results)
        """
        self.config = validate_config(config)
        self.enabled_metrics = get_enabled_metrics(self.config)
        self.model_name = model_name or "unnamed_model"
        
        logger.info(f"Enabled metrics: {self.enabled_metrics}")
        logger.info(f"Model name: {self.model_name}")
        
        # Initialize model manager for on-demand model loading
        self.model_manager = ModelManager(self.config)
        
        # Results storage
        self.results = {}  # Will store results by filename
        self.errors = []
        
        # Setup results directory
        self.results_dir = None
    
    def evaluate_dataset(self, file_pairs: List[Tuple[str, str, str]],
                        output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Evaluate all file pairs and return results DataFrame
        
        Process one metric at a time to avoid GPU memory issues:
        1. Load model for metric
        2. Process all file pairs for that metric  
        3. Save incremental results
        4. Clean up model
        5. Repeat for next metric
        6. Generate final summary
        
        Args:
            file_pairs: List of (reference_path, decoded_path, reference_text) tuples
            output_path: Optional path to save results CSV (defaults to results/{model_name}/)
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Starting evaluation of {len(file_pairs)} pairs")
        logger.info(f"Processing metrics sequentially: {self.enabled_metrics}")
        start_time = time.time()
        
        # Setup results directory
        self._setup_results_directory(output_path)
        
        # Initialize results storage
        self.results = {}
        self.errors = []
        
        # Initialize results with file paths
        for ref_path, dec_path, ref_text in file_pairs:
            file_key = self._get_file_key(ref_path, dec_path)
            self.results[file_key] = {
                'reference_path': ref_path,
                'decoded_path': dec_path,
                'reference_text': ref_text
            }

        n_processes = self.config.get('n_processes', 1)
        
        # Process each metric sequentially
        for metric_name in self.enabled_metrics:
            logger.info(f"Processing metric: {metric_name}")
            self._process_metric_for_all_files(metric_name, file_pairs, n_processes)
            
            # Save incremental results after each metric
            self._save_incremental_results(metric_name)
        
        # Convert results to DataFrame
        if self.results:
            df_results = pd.DataFrame(list(self.results.values()))
        else:
            # Create empty DataFrame with expected columns
            columns = ['reference_path', 'decoded_path', 'reference_text'] + self.enabled_metrics
            df_results = pd.DataFrame(columns=columns)
        
        elapsed_time = time.time() - start_time
        
        # Count successful vs failed evaluations per metric
        success_count = {}
        for metric in self.enabled_metrics:
            if metric in df_results.columns:
                success_count[metric] = df_results[metric].notna().sum()
            else:
                success_count[metric] = 0
        
        logger.info(f"Evaluation completed in {elapsed_time:.1f}s")
        logger.info(f"Results per metric: {success_count}")
        logger.info(f"Total errors: {len(self.errors)}")
        
        return df_results
    
    def _setup_results_directory(self, output_path: Optional[str]):
        """Setup results directory structure"""
        if output_path:
            # Use provided path as base
            self.results_dir = Path(output_path).parent
        else:
            # Create results directory based on model name
            self.results_dir = Path("results") / self.model_name
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def _process_metric_for_all_files(self, metric_name: str, file_pairs: List[Tuple[str, str, str]], n_processes: int):
        """
        Process a single metric for all file pairs
        
        Args:
            metric_name: Name of metric to process
            file_pairs: List of file pairs to process
            n_processes: Number of processes to use (if applicable)
        """
        metric_config = self.config['metrics'][metric_name]
        
        # Load model for this metric (if needed)
        model = self.model_manager.load_model_for_metric(metric_name)
        
        # Process all files for this metric
        for ref_path, dec_path, ref_text in tqdm(file_pairs, desc=f"Computing {metric_name}"):
            file_key = self._get_file_key(ref_path, dec_path)
            
            try:
                score = self._compute_metric(metric_name, ref_path, dec_path, ref_text, metric_config, model, n_processes)
                self.results[file_key][metric_name] = score
                
            except Exception as e:
                error_msg = f"Failed to compute {metric_name} for {Path(ref_path).name}: {e}"
                logger.warning(error_msg)
                
                self.errors.append({
                    'reference_path': ref_path,
                    'decoded_path': dec_path,
                    'metric': metric_name,
                    'error': str(e)
                })
                
                # Set metric value to None for failed computation
                self.results[file_key][metric_name] = None
        
        # Clean up model after processing all files for this metric
        self.model_manager.cleanup_current_model()
    
    def _save_incremental_results(self, metric_name: str):
        """
        Save incremental results for a specific metric
        
        Args:
            metric_name: Name of the metric that was just computed
        """
        # Create DataFrame with current results
        df_current = pd.DataFrame(list(self.results.values()))
        
        # Filter to only include essential columns + current metric
        essential_cols = ['reference_path', 'decoded_path', 'reference_text']
        if metric_name in df_current.columns:
            metric_cols = essential_cols + [metric_name]
            df_metric = df_current[metric_cols].copy()
            
            # Save metric-specific CSV
            metric_file = self.results_dir / f"{metric_name}.csv"
            df_metric.to_csv(metric_file, index=False)
            logger.info(f"Saved {metric_name} results to: {metric_file}")
            
            # Log metric statistics
            metric_values = df_metric[metric_name].dropna()
            if len(metric_values) > 0:
                logger.info(f"{metric_name} stats: mean={metric_values.mean():.4f}, "
                           f"std={metric_values.std():.4f}, count={len(metric_values)}")
    
    def _get_file_key(self, ref_path: str, dec_path: str) -> str:
        """Generate unique key for file pair"""
        return f"{ref_path}|{dec_path}"
    
    def _compute_metric(self, metric_name: str, ref_path: str, dec_path: str, ref_text: str, 
                       metric_config: Dict[str, Any], model: Optional[Any] = None, n_processes: int = 1) -> float:
        """
        Compute a specific metric for a file pair
        
        Args:
            metric_name: Name of the metric to compute
            ref_path: Reference audio file path
            ref_text: Reference transcription text
            dec_path: Decoded audio file path  
            metric_config: Configuration for this metric
            model: Pre-loaded model (if applicable)
            n_processes: Number of processes to use (if applicable)
            
        Returns:
            Metric score
        """
        if metric_name == 'stoi':
            return self._compute_stoi(ref_path, dec_path, metric_config)
        elif metric_name == 'pesq_nb':
            return self._compute_pesq(ref_path, dec_path, metric_config, mode='nb', n_processes=n_processes)
        elif metric_name == 'pesq_wb':
            return self._compute_pesq(ref_path, dec_path, metric_config, mode='wb', n_processes=n_processes)
        elif metric_name == 'mcd':
            return self._compute_mcd(model, ref_path, dec_path)
        elif metric_name == 'speaker_similarity':
            return self._compute_speaker_similarity(ref_path, dec_path, metric_config, model)
        elif metric_name == 'utmos':
            return self._compute_utmos(ref_path, dec_path, metric_config, model)
        elif metric_name == 'nisqa':
            return self._compute_nisqa(ref_path, dec_path, metric_config, model)
        elif metric_name == 'dnsmos':
            return self._compute_dnsmos(ref_path, dec_path, metric_config, model)
        elif metric_name == 'wer':
            return self._compute_wer(dec_path, ref_text, metric_config, model)
        elif metric_name == 'cer':
            return self._compute_cer(ref_path, dec_path, metric_config, model, ref_text)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    def _compute_stoi(self, ref_path: str, dec_path: str, config: Dict[str, Any]) -> float:
        """Compute STOI metric"""
        return compute_stoi(
            ref_path, dec_path,
            extended=config.get('extended', False)
        )
    
    def _compute_pesq(self, ref_path: str, dec_path: str, config: Dict[str, Any], mode: Literal['nb', 'wb'], n_processes: int) -> float:
        """Compute PESQ metric"""
        # load audio to tensors
        ref_audio, _ = torchaudio.load(ref_path)
        dec_audio, _ = torchaudio.load(dec_path)

        # Ensure both audio tensors have the same length
        ref_length = ref_audio.size(1)
        dec_length = dec_audio.size(1)
        
        if dec_length < ref_length:
            # Pad decoded audio if shorter than reference
            padding = ref_length - dec_length
            dec_audio = torch.nn.functional.pad(dec_audio, (0, padding))
        elif dec_length > ref_length:
            # Truncate decoded audio if longer than reference
            dec_audio = dec_audio[:, :ref_length]
        
        # Both tensors now have the same length
        assert ref_audio.size(1) == dec_audio.size(1), f"Audio length mismatch: ref={ref_audio.size(1)}, dec={dec_audio.size(1)}"
        return compute_PESQ(
            ref_audio, dec_audio,
            mode=mode,
            n_processes=n_processes
        )
    
    def _compute_mcd(self, model, ref_path: str, dec_path: str) -> float:
        """Compute MCD metric"""
        return compute_mcd(
            model, ref_path, dec_path
        )
    
    def _compute_speaker_similarity(self, ref_path: str, dec_path: str, 
                                  config: Dict[str, Any], model: Optional[Any] = None) -> float:
        """Compute speaker similarity metric using pre-loaded model"""
        if model is not None:
            # Use pre-loaded model
            return compute_speaker_similarity(
                ref_path, dec_path,
                model=model
            )
        else:
            raise RuntimeError("Speaker similarity model not loaded")
    
    def _compute_utmos(self, ref_path: str, dec_path: str, 
                      config: Dict[str, Any], model: Optional[Any] = None) -> float:

        """Compute UTMOS metric using pre-loaded model"""
        if model is not None:
            # Load audio using soundfile (returns numpy array)
            audio, sr = sf.read(dec_path)
            
            # Ensure audio is 1D (mono)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert stereo to mono by averaging channels
            
            # Convert to tensor and add batch dimension
            # UTMOS expects (batch, time) format, NOT (batch, channels, time)
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # Shape: (1, time)
            
            # Use the predictor's predict method which handles device placement
            score = model.predict(audio_tensor, sr)
            return float(score)
        else:
            raise RuntimeError("UTMOS model not loaded")
    
    def _compute_nisqa(self, ref_path: str, dec_path: str, 
                      config: Dict[str, Any], model: Optional[Any] = None) -> float:
        """Compute NISQA metric using pre-loaded model"""

        # Load audio
        decoded_audio, sr = sf.read(dec_path)
        decoded_tensor = torch.from_numpy(decoded_audio)
        if model is not None:
            # Use pre-loaded model
            if self.model_manager.get_device() != 'cpu':
                decoded_tensor = decoded_tensor.to(self.model_manager.get_device())
            
            score = model(decoded_tensor)
            return float(score[0].item())
        else:
            raise RuntimeError("NISQA model not loaded")
    
    def _compute_dnsmos(self, ref_path: str, dec_path: str, 
                       config: Dict[str, Any], model: Optional[Any] = None) -> float:
        """Compute DNSMOS metric using pre-loaded model"""

        # Load audio
        decoded_audio, sr = sf.read(dec_path)
        decoded_tensor = torch.from_numpy(decoded_audio)
        
        if model is not None:
            # Use pre-loaded model            
            score = model(decoded_tensor)
            if self.model_manager.get_device() != 'cpu':
                score = score.to(self.model_manager.get_device())
            return float(score.item())
        else:
            # Fallback to original function
            return compute_DNSMOS(
                decoded_tensor, 
                fs=sr,
                personalized=config.get('personalized', False),
                device=self.model_manager.get_device()
            )
    
    def _compute_wer(self, dec_path: str, ref_text: str,
                    config: Dict[str, Any], model: Optional[Any] = None) -> float:
        """Compute WER metric using pre-loaded Whisper model"""
        if model is None:
            raise RuntimeError("Whisper model not loaded for WER computation")
        
        # Transcribe decoded audio using pre-loaded model
        language = config.get('language')
        hypothesis_text = transcribe_audio(model, dec_path, language)
        
        # Compute WER
        return compute_wer(model=model, 
                           decoded_path=dec_path, 
                           reference_text=ref_text, 
                           language=language)
    
    def _compute_cer(self, ref_path: str, dec_path: str, 
                    config: Dict[str, Any], model: Optional[Any] = None, ref_text: str = None) -> float:
        """Compute CER metric using pre-loaded Whisper model"""
        if model is None:
            raise RuntimeError("Whisper model not loaded for CER computation")
        
        # Use provided reference text or load from file
        if ref_text is None:
            ref_txt_path = Path(ref_path).with_suffix('.txt')
            if not ref_txt_path.exists():
                raise FileNotFoundError(f"Reference transcription not found: {ref_txt_path}")
            
            with open(ref_txt_path, 'r') as f:
                reference_text = f.read().strip()
        else:
            reference_text = ref_text
        
        # Transcribe decoded audio using pre-loaded model
        language = config.get('language')
        hypothesis_text = transcribe_audio(model, dec_path, language)
        
        # Compute CER
        from jiwer import cer
        return cer(reference_text, hypothesis_text)
    
    def get_summary_stats(self, df_results: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each metric
        
        Args:
            df_results: Results DataFrame
            
        Returns:
            Dictionary of summary statistics per metric
        """
        summary = {}
        
        for metric in self.enabled_metrics:
            if metric in df_results.columns:
                metric_values = df_results[metric].dropna()
                
                if len(metric_values) > 0:
                    summary[metric] = {
                        'count': len(metric_values),
                        'mean': float(metric_values.mean()),
                        'std': float(metric_values.std()),
                        'min': float(metric_values.min()),
                        'max': float(metric_values.max()),
                        'median': float(metric_values.median()),
                        'q25': float(metric_values.quantile(0.25)),
                        'q75': float(metric_values.quantile(0.75))
                    }
                else:
                    summary[metric] = {
                        'count': 0,
                        'mean': None,
                        'std': None,
                        'min': None,
                        'max': None,
                        'median': None,
                        'q25': None,
                        'q75': None
                    }
        
        return summary
    
    def cleanup(self):
        """Clean up resources"""
        self.model_manager.cleanup()