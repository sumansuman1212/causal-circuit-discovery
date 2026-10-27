# tests/test_pipeline.py
import torch
from torch.utils.data import DataLoader
from src.config import ModelConfig
from src.dataset import AssociativeRecallDataset, SyntheticTaskGenerator, get_dataloaders
from src.model import TinyTransformer


def test_dataset_generation():
    gen = SyntheticTaskGenerator(num_entities=20, seed=123)
    clean_in, clean_tgt, corrupt_in, corrupt_tgt = gen.generate_pair()

    assert len(clean_in) == len(corrupt_in)
    assert clean_tgt != corrupt_tgt
    assert clean_tgt in clean_in
    assert corrupt_tgt in corrupt_in


def test_model_forward():
    cfg = ModelConfig(vocab_size=35, n_ctx=16)
    model = TinyTransformer(cfg)
    dummy_input = torch.randint(0, 35, (2, 11))
    out = model(dummy_input)
    assert out.shape == (2, 11, 35)


def test_dataloaders_split():
    gen = SyntheticTaskGenerator(num_entities=15, seed=42)
    clean_data, _ = gen.build_dataset(num_samples=100)
    train_loader, val_loader, test_loader = get_dataloaders(clean_data, batch_size=16)

    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 10
    assert len(test_loader.dataset) == 10
