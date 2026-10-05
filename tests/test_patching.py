# tests/test_patching.py
import pytest
import torch
from src.config import ModelConfig
from src.dataset import SyntheticTaskGenerator
from src.model import TinyTransformer
from src.patching import ActivationPatcher


def test_activation_patching_dimensions():
    seed = 42
    torch.manual_seed(seed)
    task_gen = SyntheticTaskGenerator(num_entities=15, seed=seed)
    cfg = ModelConfig(vocab_size=task_gen.vocab_size, n_layers=2, n_heads=2)
    model = TinyTransformer(cfg)

    clean_data, corrupt_data = task_gen.build_dataset(num_samples=16)
    clean_ids = torch.tensor([x[0] for x in clean_data], dtype=torch.long)
    target_ids = torch.tensor([x[1] for x in clean_data], dtype=torch.long)
    corrupt_ids = torch.tensor([x[0] for x in corrupt_data], dtype=torch.long)

    patcher = ActivationPatcher(model, device="cpu")
    df = patcher.patch_all_heads(clean_ids, corrupt_ids, target_ids)

    assert len(df) == 4  # 2 layers * 2 heads
    expected_cols = {"component", "layer", "head", "clean_probability", "corrupted_probability", "patched_probability", "causal_effect"}
    assert expected_cols.issubset(df.columns)
    assert not df["causal_effect"].isna().any()
