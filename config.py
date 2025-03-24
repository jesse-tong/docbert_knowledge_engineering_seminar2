"""
Configuration module for DocBERT
Contains hyperparameter presets for different dataset types
"""

class BaseConfig:
    # Model params
    bert_model = "bert-base-uncased"
    max_seq_length = 512
    dropout = 0.1
    
    # Training params
    batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    epochs = 10
    grad_accum_steps = 1
    
    # Data params
    val_split = 0.1
    test_split = 0.1
    seed = 42

class ShortTextConfig(BaseConfig):
    """Config for short text classification (tweets, comments, etc.)"""
    max_seq_length = 128
    batch_size = 32
    learning_rate = 3e-5
    
class LongDocumentConfig(BaseConfig):
    """Config for long document classification"""
    bert_model = "bert-large-uncased" 
    max_seq_length = 512
    batch_size = 8
    grad_accum_steps = 2
    weight_decay = 0.02
    
class FinetuningConfig(BaseConfig):
    """Config for fine-tuning on a small dataset"""
    learning_rate = 1e-5
    batch_size = 8
    epochs = 15
    weight_decay = 0.03
    dropout = 0.2

CONFIG_PRESETS = {
    "default": BaseConfig,
    "short_text": ShortTextConfig,
    "long_document": LongDocumentConfig,
    "fine_tuning": FinetuningConfig
}

def get_config(preset_name="default"):
    """Get a configuration preset by name"""
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"Config preset '{preset_name}' not found. Available presets: {list(CONFIG_PRESETS.keys())}")
    return CONFIG_PRESETS[preset_name]