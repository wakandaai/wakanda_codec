#!/usr/bin/env python3
"""
Main training script for LlamaTTS.
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codec.tts.tts_model import LlamaTTSForCausalLM, LlamaTTSConfig
from codec.tts.dataset import TTSDataset
from codec.tts.collator import TTSDataCollator
from codec.tts.trainer import TTSTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    codec_config_path: str = field(
        metadata={"help": "Path to codec configuration YAML"}
    )
    train_config_path: str = field(
        metadata={"help": "Path to training configuration YAML"}
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_tokenizer(base_model: str, codec_config: dict, train_config: dict):
    """
    Setup tokenizer with extended vocabulary for speech tokens.
    
    Returns:
        tokenizer with added special tokens and codebook tokens
    """
    logger.info(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    special_tokens = train_config['model']['special_tokens']
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    logger.info(f"Added {num_added} special tokens: {special_tokens}")
    
    # Add codebook tokens
    codebook_vocab_size = codec_config['codebook_vocab_size']
    codebook_tokens = [f'<|c0_{i}|>' for i in range(codebook_vocab_size)]
    num_added = tokenizer.add_tokens(codebook_tokens, special_tokens=False)
    logger.info(f"Added {num_added} codebook tokens (vocab size: {codebook_vocab_size})")
    
    logger.info(f"Final tokenizer vocabulary size: {len(tokenizer)}")
    
    return tokenizer


def create_model(
    base_model: str,
    tokenizer,
    codec_config: dict,
    train_config: dict,
):
    """Create LlamaTTS model."""
    
    logger.info(f"Loading base model config from {base_model}")
    base_config = LlamaConfig.from_pretrained(base_model)
    
    # Create TTS config
    sub_transformer_config = train_config['model']['sub_transformer']
    
    tts_config = LlamaTTSConfig(
        # Base Llama config attributes
        vocab_size=len(tokenizer),
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=base_config.num_hidden_layers,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        max_position_embeddings=base_config.max_position_embeddings,
        rms_norm_eps=base_config.rms_norm_eps,
        rope_theta=base_config.rope_theta,
        attention_bias=base_config.attention_bias,
        attention_dropout=base_config.attention_dropout,
        # TTS-specific
        base_model_name=base_model,
        num_codebooks=codec_config['num_codebooks'],
        codebook_vocab_size=codec_config['codebook_vocab_size'],
        speaker_embedding_dim=codec_config['speaker_embedding']['dim'],
        freeze_speaker_projection=codec_config['speaker_embedding']['freeze_projection'],
        use_sub_transformer=sub_transformer_config['enabled'],
        sub_transformer_config={
            'num_layers': sub_transformer_config['num_layers'],
            'hidden_size': sub_transformer_config['hidden_size'],
            'num_attention_heads': sub_transformer_config['num_attention_heads'],
            'num_key_value_heads': sub_transformer_config['num_key_value_heads'],
            'intermediate_size': sub_transformer_config['intermediate_size'],
        },
        sub_transformer_loss_weight=sub_transformer_config['loss_weight'],
    )
    
    logger.info("Creating LlamaTTS model")
    
    # Load pretrained Llama weights
    logger.info(f"Loading pretrained weights from {base_model}")
    from transformers import LlamaForCausalLM
    base_llama = LlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if train_config['training']['bf16'] else torch.float32,
        cache_dir=train_config['model'].get('cache_dir'),
    )
    
    # Create TTS model
    model = LlamaTTSForCausalLM(tts_config)
    
    # Copy Llama weights
    logger.info("Copying base Llama weights")
    model.model.load_state_dict(base_llama.model.state_dict(), strict=False)
    
    # Resize embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model created with {model.num_parameters():,} parameters")
    
    # Initialize speaker projection with small random values
    nn_init = torch.nn.init
    nn_init.normal_(model.speaker_projection.weight, mean=0.0, std=0.02)
    
    # Freeze speaker projection if requested
    if codec_config['speaker_embedding']['freeze_projection']:
        model.speaker_projection.requires_grad_(False)
        logger.info("Froze speaker projection layer")
    
    return model


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments,))
    
    if len(sys.argv) == 3 and sys.argv[1].endswith('.yaml') and sys.argv[2].endswith('.yaml'):
        # Called with two config files
        model_args, = parser.parse_args_into_dataclasses(args=[
            f"--codec_config_path={sys.argv[1]}",
            f"--train_config_path={sys.argv[2]}"
        ])
    else:
        model_args, = parser.parse_args_into_dataclasses()
    
    # Load configs
    logger.info(f"Loading codec config from {model_args.codec_config_path}")
    codec_config = load_config(model_args.codec_config_path)
    
    logger.info(f"Loading train config from {model_args.train_config_path}")
    train_config = load_config(model_args.train_config_path)
    
    # Set seed
    set_seed(42)
    
    # Setup tokenizer
    base_model = train_config['model']['base_model']
    tokenizer = setup_tokenizer(base_model, codec_config, train_config)
    
    # Create model
    model = create_model(base_model, tokenizer, codec_config, train_config)
    
    # Create datasets
    logger.info("Loading datasets")
    
    dataset_config = {
        'num_codebooks': codec_config['num_codebooks'],
        'codebook_vocab_size': codec_config['codebook_vocab_size'],
        'framerate': codec_config['framerate'],
    }
    
    train_dataset = TTSDataset(
        csv_path=train_config['data']['train_csv'],
        tokenizer=tokenizer,
        config=dataset_config,
        max_seq_length=train_config['training']['max_seq_length'],
    )
    
    eval_dataset = None
    if 'val_csv' in train_config['data']:
        eval_dataset = TTSDataset(
            csv_path=train_config['data']['val_csv'],
            tokenizer=tokenizer,
            config=dataset_config,
            max_seq_length=train_config['training']['max_seq_length'],
        )
    
    # Data collator
    data_collator = TTSDataCollator(
        tokenizer=tokenizer,
        num_codebooks=codec_config['num_codebooks'],
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=train_config['training']['output_dir'],
        num_train_epochs=train_config['training']['num_train_epochs'],
        per_device_train_batch_size=train_config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['training']['gradient_accumulation_steps'],
        learning_rate=train_config['training']['learning_rate'],
        weight_decay=train_config['training']['weight_decay'],
        adam_beta2=train_config['training']['adam_beta2'],
        warmup_ratio=train_config['training']['warmup_ratio'],
        lr_scheduler_type=train_config['training']['lr_scheduler_type'],
        bf16=train_config['training']['bf16'],
        fp16=train_config['training']['fp16'],
        gradient_checkpointing=train_config['training']['gradient_checkpointing'],
        evaluation_strategy=train_config['training']['evaluation_strategy'],
        eval_steps=train_config['training']['eval_steps'],
        save_strategy=train_config['training']['save_strategy'],
        save_steps=train_config['training']['save_steps'],
        save_total_limit=train_config['training']['save_total_limit'],
        logging_steps=train_config['training']['logging_steps'],
        report_to=train_config['training']['report_to'],
        run_name=train_config['training']['run_name'],
        dataloader_num_workers=train_config['training']['dataloader_num_workers'],
        dataloader_prefetch_factor=train_config['training'].get('dataloader_prefetch_factor', 2),
        deepspeed=train_config['training'].get('deepspeed'),
        remove_unused_columns=False,  # We have custom columns
    )
    
    # Create trainer
    trainer = TTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()