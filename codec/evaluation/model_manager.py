# codec/evaluation/model_manager.py

"""
Model Manager for Model Based metric

Models are loaded on-demand and cleaned up after processing all files for that metric.
"""

import logging
from typing import Dict, Any, Optional
import torch
import warnings
from codec.evaluation.utmos import UTMOSPredictor
from codec.evaluation.speaker_sim import init_model
from codec.evaluation.wer import load_whisper_model
from codec.evaluation.mcd import create_mcd_toolbox
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for loading/unloading individual metric models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        self.current_model = None
        self.current_metric = None
    
    def _get_device(self) -> str:
        """Determine appropriate device based on config"""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_config
    
    def load_model_for_metric(self, metric_name: str) -> Optional[Any]:
        """
        Load model for a specific metric, cleaning up any previously loaded model
        
        Args:
            metric_name: Name of metric to load model for
            
        Returns:
            Loaded model or None if no model needed for this metric
        """
        # Clean up any existing model first
        if self.current_model is not None:
            self.cleanup_current_model()
        
        # Get metric configuration
        metrics_config = self.config.get('metrics', {})
        metric_config = metrics_config.get(metric_name, {})
        
        if not metric_config.get('enabled', False):
            return None
        
        logger.info(f"Loading model for {metric_name}")
        
        try:
            if metric_name == 'utmos':
                model = self._load_utmos_model(metric_config)
            elif metric_name == 'speaker_similarity':
                model = self._load_speaker_model(metric_config)
            elif metric_name == 'wer' or metric_name == 'cer':
                model = self._load_whisper_model(metric_config)
            elif metric_name == 'nisqa':
                model = self._load_nisqa_model(metric_config)
            elif metric_name == 'dnsmos':
                model = self._load_dnsmos_model(metric_config)
            elif metric_name == 'mcd':
                model = create_mcd_toolbox()
            else:
                # No model needed for this metric (e.g., STOI, PESQ)
                return None
            
            self.current_model = model
            self.current_metric = metric_name
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model for {metric_name}: {e}")
            warnings.warn(f"Model loading failed for {metric_name}: {e}")
            return None
    
    def get_current_model(self) -> Optional[Any]:
        """Get the currently loaded model"""
        return self.current_model
    
    def cleanup_current_model(self):
        """Clean up the currently loaded model"""
        if self.current_model is not None:
            logger.info(f"Cleaning up model for {self.current_metric}")
            
            try:
                # Move model to CPU and delete
                if hasattr(self.current_model, 'cpu'):
                    self.current_model.cpu()
                elif hasattr(self.current_model, 'to'):
                    self.current_model.to('cpu')
                
                del self.current_model
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.warning(f"Error during model cleanup: {e}")
            
            finally:
                self.current_model = None
                self.current_metric = None
    
    def _load_utmos_model(self, utmos_config: Dict[str, Any]):
        """Load UTMOS predictor model"""
        
        model_name = utmos_config.get('model_name', 'utmos22_strong')
        predictor = UTMOSPredictor(model_name=model_name, device=self.device)
        logger.info(f"Loaded UTMOS model: {model_name}")
        return predictor
    
    def _load_speaker_model(self, speaker_config: Dict[str, Any]):
        """Load speaker similarity model"""
        
        model_name = speaker_config.get('model_name', 'wavlm_large')
        model = init_model(model_name)
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded speaker similarity model: {model_name}")
        return model
    
    def _load_whisper_model(self, wer_config: Dict[str, Any]):
        """Load Whisper ASR model"""
        
        model_name = wer_config.get('model_name', 'openai/whisper-large-v3')
        pipe = load_whisper_model(model_name=model_name, device=self.device)
        
        logger.info(f"Loaded Whisper model: {model_name}")
        return pipe
    
    def _load_nisqa_model(self, nisqa_config: Dict[str, Any]):
        """Load NISQA model"""
        
        fs = nisqa_config.get('sample_rate', 16000)
        nisqa_metric = NonIntrusiveSpeechQualityAssessment(fs=fs)
        if self.device != 'cpu':
            nisqa_metric = nisqa_metric.to(self.device)
        
        logger.info("Loaded NISQA model")
        return nisqa_metric
    
    def _load_dnsmos_model(self, dnsmos_config: Dict[str, Any]):
        """Load DNSMOS model"""
        
        fs = dnsmos_config.get('sample_rate', 16000)
        personalized = dnsmos_config.get('personalized', False)
        
        dnsmos_metric = DeepNoiseSuppressionMeanOpinionScore(
            fs=fs, 
            personalized=personalized,
            device=self.device,
            cache_sessions=True
        )
        
        logger.info("Loaded DNSMOS model")
        return dnsmos_metric
    
    def get_device(self) -> str:
        """Get device being used for models"""
        return self.device
    
    def cleanup(self):
        """Final cleanup of all resources"""
        self.cleanup_current_model()
        logger.info("Model manager cleanup complete")