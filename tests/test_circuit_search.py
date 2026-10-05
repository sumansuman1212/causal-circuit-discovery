# tests/test_circuit_search.py
import pytest
import torch
from torch.utils.data import DataLoader
from src.ablation import ComponentAblator
from src.circuit_search import MinimalCircuitFinder
from src.config import ModelConfig
from src.dataset import AssociativeRecallDataset, SyntheticTaskGenerator
from src.model import TinyTransformer


def test_minimal_circuit_finder():
    seed = 42
    torch.manual_seed(seed)
    task_gen = SyntheticTaskGenerator(num_entities=12, seed=seed)
    cfg = ModelConfig(vocab_size=task_gen.vocab_size, n_layers=2, n_heads=2)
    model = TinyTransformer(cfg)

    clean_data, _ = task_gen.build_dataset(num_samples=32)
    loader = DataLoader(AssociativeRecallDataset(clean_data), batch_size=16)

    ablator = ComponentAblator(model, device="cpu")
    ablator.compute_head_means(loader)

    finder = MinimalCircuitFinder(model, ablator, device="cpu")
    circuit = {(0, 1), (1, 0)}

    suff_acc = finder.evaluate_circuit_sufficiency(circuit, loader)
    nec_acc = finder.evaluate_circuit_necessity(circuit, loader)

    assert 0.0 <= suff_acc <= 1.0
    assert 0.0 <= nec_acc <= 1.0
