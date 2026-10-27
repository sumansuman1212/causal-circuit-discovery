# src/config.py
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_mlp: int = 512
    n_ctx: int = 32
    vocab_size: int = 100  # Will be dynamically set by dataset generator
    act_fn: str = "gelu"

@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 15
    device: str = "cpu"
    checkpoint_dir: Path = Path("models/checkpoints")
    results_dir: Path = Path("results/baseline")

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
