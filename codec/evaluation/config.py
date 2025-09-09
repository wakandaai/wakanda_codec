# codec/evaluation/config.py

"""
Configuration management for audio codec evaluation
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_file.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and set reasonable defaults
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validated configuration
    """
    # Validate device setting
    valid_devices = {"auto", "cpu", "cuda"}
    if config.get("device", "auto") not in valid_devices:
        raise ValueError(f"Invalid device: {config['device']}. Must be one of {valid_devices}")
    
    # Validate that at least one metric is enabled
    metrics = config.get("metrics", {})
    enabled_metrics = [name for name, cfg in metrics.items() if cfg.get("enabled", False)]
    
    if not enabled_metrics:
        raise ValueError("No metrics are enabled. Please enable at least one metric.")
    
    return config


def get_enabled_metrics(config: Dict[str, Any]) -> List[str]:
    """
    Get list of enabled metric names from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of enabled metric names
    """
    metrics = config.get("metrics", {})
    return [name for name, cfg in metrics.items() if cfg.get("enabled", False)]