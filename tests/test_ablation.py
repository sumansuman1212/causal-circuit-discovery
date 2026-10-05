# tests/test_ablation.py
import pytest
import torch
from torch.utils.data import DataLoader
from src.ablation import ComponentAblator
from src.config import ModelConfig
from src.dataset import AssociativeRecallDataset, SyntheticTaskGenerator
from src.model import TinyTransformer


def test_component_ablator():
    seed = 42
    torch.manual_seed(seed)
    task_gen = SyntheticTaskGenerator(num_entities=12, seed=seed)
    cfg = ModelConfig(vocab_size=task_gen.vocab_size, n_layers=2, n_heads=2)
    model = TinyTransformer(cfg)

    clean_data, _ = task_gen.build_dataset(num_samples=32)
    loader = DataLoader(AssociativeRecallDataset(clean_data), batch_size=16)

    ablator = ComponentAblator(model, device="cpu")
    ablator.compute_head_means(loader)

    # Assert means calculated for every layer
    assert len(ablator.mean_head_activations) == 2
    assert ablator.mean_head_activations[0].shape[1] == 2

    # Run evaluation with an ablated head
    acc = ablator.evaluate_with_ablated_heads(loader, ablated_heads={(0, 1)}, mode="mean")
    assert 0.0 <= acc <= 1.0
