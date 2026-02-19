"""
Custom Trainer for TTS with logging of main/sub losses.
"""

import torch
from transformers import Trainer
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TTSTrainer(Trainer):
    """
    Custom trainer that logs main_loss and sub_loss separately.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to extract and log component losses.
        """
        outputs = model(**inputs)
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.total_loss
        
        # Log component losses
        if hasattr(outputs, 'main_loss') and outputs.main_loss is not None:
            self.log({'main_loss': outputs.main_loss.item()})
        
        if hasattr(outputs, 'sub_loss') and outputs.sub_loss is not None:
            self.log({'sub_loss': outputs.sub_loss.item()})
        
        return (loss, outputs) if return_outputs else loss