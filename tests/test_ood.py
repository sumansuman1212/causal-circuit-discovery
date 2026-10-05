# tests/test_ood.py
import pytest
from src.dataset import SyntheticTaskGenerator
from src.ood import OODTaskGenerator


def test_ood_task_generator_formats():
    seed = 42
    base_gen = SyntheticTaskGenerator(num_entities=30, seed=seed)
    ood_gen = OODTaskGenerator(base_gen, seed=seed)

    # 1. Test OOD-1 (Unseen Entities)
    tokens_1, tgt_1 = ood_gen.generate_ood1_unseen_entities()
    assert len(tokens_1) == 11
    assert tgt_1 in tokens_1
    assert tgt_1 in [base_gen.vocab[e] for e in ood_gen.unseen_entities]

    # 2. Test OOD-2 (Distractors)
    tokens_2, tgt_2 = ood_gen.generate_ood2_distractors()
    assert len(tokens_2) == 15  # Includes 4 distractor tokens (s, likes, o, sep)
    assert tgt_2 in tokens_2

    # 3. Test OOD-3 (Long Context)
    tokens_3, tgt_3 = ood_gen.generate_ood3_longer_context()
    assert len(tokens_3) == 15  # 3 pairs of 4 tokens + 3 query tokens
    assert tgt_3 in tokens_3
