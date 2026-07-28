"""
Utility helpers for the Multimodal Medical Report Generation system.

Provides:
    - Reproducibility (seed setting)
    - Device auto-detection (CUDA > MPS > CPU)
    - YAML config loading with dotdict access
    - Logging with TensorBoard integration
    - Checkpoint save/load helpers
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


# ============================================================================
# DotDict — nested dictionary with attribute-style access
# ============================================================================

class DotDict(dict):
    """
    Dictionary subclass enabling dot-notation access.
    
    Example:
        cfg = DotDict({'training': {'lr': 5e-5}})
        cfg.training.lr  # 5e-5
    """

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")


# ============================================================================
# Seed & Device
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """
    Auto-detect the best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"Using CUDA device: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logging.warning("No GPU detected — using CPU. Training will be very slow.")
    return device


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str) -> DotDict:
    """
    Load YAML configuration file and return as DotDict.
    
    Args:
        config_path: Path to the YAML config file.
        
    Returns:
        DotDict with nested attribute access.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    
    config = DotDict(raw)
    logging.info(f"Loaded config from {config_path}")
    return config


def save_config(config: Union[dict, DotDict], save_path: str) -> None:
    """Save config dict back to YAML (useful for experiment logging)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert DotDict back to regular dict for YAML serialization
    def _to_dict(d):
        if isinstance(d, dict):
            return {k: _to_dict(v) for k, v in d.items()}
        return d
    
    with open(save_path, "w") as f:
        yaml.dump(_to_dict(config), f, default_flow_style=False, sort_keys=False)


# ============================================================================
# Logging
# ============================================================================

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    experiment_name: str = "experiment",
) -> logging.Logger:
    """
    Configure logging with console + file handlers.
    
    Args:
        log_dir: Directory for log files. If None, logs to console only.
        log_level: Logging verbosity level.
        experiment_name: Name prefix for log files.
        
    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # File handler (if log_dir provided)
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path / f"{experiment_name}.log")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logging.info(f"Logging to {log_path / f'{experiment_name}.log'}")
    
    # Suppress verbose Hugging Face tokenizer warnings
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass

    return logger


# ============================================================================
# Checkpoint Helpers
# ============================================================================

def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
    is_best: bool = False,
) -> str:
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model_state_dict, optimizer_state_dict,
               epoch, metrics, config, etc.
        checkpoint_dir: Directory to save checkpoints.
        filename: Checkpoint filename.
        is_best: If True, also saves as 'best.pt'.
        
    Returns:
        Path to saved checkpoint.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = ckpt_dir / filename
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved: {filepath}")
    
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(state, best_path)
        logging.info(f"Best model saved: {best_path}")
    
    return str(filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load state dict into.
        optimizer: Optionally load optimizer state.
        device: Device to map tensors to.
        
    Returns:
        Checkpoint dict with metadata (epoch, metrics, etc.).
    """
    if device is None:
        device = get_device()
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info(f"Loaded model from {checkpoint_path}")
    
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logging.info("Loaded optimizer state")
    
    return ckpt


# ============================================================================
# Misc Helpers
# ============================================================================

def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count total (or trainable) parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
