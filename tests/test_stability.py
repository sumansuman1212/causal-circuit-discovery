# tests/test_stability.py
import numpy as np
import pytest
from src.stability import (
    compute_functional_stability_matrix,
    compute_jaccard_similarity,
    compute_structural_stability_matrix,
)


def test_jaccard_similarity():
    s1 = {"L1H0", "L2H1"}
    s2 = {"L1H0", "L2H1"}
    assert compute_jaccard_similarity(s1, s2) == 1.0

    s3 = {"L1H0", "L3H2"}
    assert compute_jaccard_similarity(s1, s3) == 1 / 3

    s4 = {"L0H0"}
    assert compute_jaccard_similarity(s1, s4) == 0.0


def test_stability_matrices():
    circuits = {
        1: {"L1H0", "L2H1"},
        2: {"L1H0", "L2H1"},
        3: {"L0H0", "L2H1"},
    }
    seeds = [1, 2, 3]
    mat, mean_sim, std_sim = compute_structural_stability_matrix(circuits, seeds)
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 1.0)
    assert 0.0 <= mean_sim <= 1.0

    causal_vecs = {
        1: np.array([0.1, 0.8, 0.2, 0.9]),
        2: np.array([0.1, 0.75, 0.25, 0.85]),
        3: np.array([0.9, 0.2, 0.8, 0.1]),
    }
    f_mat, f_mean, _ = compute_functional_stability_matrix(causal_vecs, seeds)
    assert f_mat.shape == (3, 3)
    assert np.allclose(np.diag(f_mat), 1.0)
    assert -1.0 <= f_mean <= 1.0
