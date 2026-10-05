# src/stability.py
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch

from src.circuit_metrics import plot_causal_heatmap
from src.circuit_search import run_circuit_discovery
from src.config import ModelConfig, TrainConfig
from src.patching import run_activation_patching_experiment
from src.train import train_baseline


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard index: |A & B| / |A | B|."""
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 1.0
    return intersection / union


def compute_structural_stability_matrix(
    circuits: Dict[int, Set[str]], seeds: List[int]
) -> Tuple[np.ndarray, float, float]:
    """
    Computes pairwise Jaccard similarity matrix across all seeds.
    Returns:
        matrix: (N, N) array
        mean_pairwise: scalar mean off-diagonal similarity
        std_pairwise: scalar std off-diagonal similarity
    """
    n = len(seeds)
    matrix = np.zeros((n, n))
    off_diagonals = []

    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            sim = compute_jaccard_similarity(circuits[s1], circuits[s2])
            matrix[i, j] = sim
            if i != j:
                off_diagonals.append(sim)

    mean_pairwise = float(np.mean(off_diagonals)) if off_diagonals else 1.0
    std_pairwise = float(np.std(off_diagonals)) if off_diagonals else 0.0
    return matrix, mean_pairwise, std_pairwise


def compute_functional_stability_matrix(
    causal_vectors: Dict[int, np.ndarray], seeds: List[int]
) -> Tuple[np.ndarray, float, float]:
    """
    Computes pairwise Pearson correlation of causal effect vectors across all 16 heads.
    """
    n = len(seeds)
    matrix = np.zeros((n, n))
    off_diagonals = []

    for i, s1 in enumerate(seeds):
        for j, s2 in enumerate(seeds):
            r, _ = pearsonr(causal_vectors[s1], causal_vectors[s2])
            matrix[i, j] = r
            if i != j:
                off_diagonals.append(r)

    mean_pairwise = float(np.mean(off_diagonals)) if off_diagonals else 1.0
    std_pairwise = float(np.std(off_diagonals)) if off_diagonals else 0.0
    return matrix, mean_pairwise, std_pairwise


def plot_dual_stability_matrices(
    jaccard_mat: np.ndarray,
    functional_mat: np.ndarray,
    seeds: List[int],
    output_dir: Path = Path("results/figures"),
) -> Path:
    """
    Renders side-by-side heatmaps: Structural (Jaccard) vs Functional (Causal Correlation).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"S_{s}" for s in seeds]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Structural Stability (Jaccard)
    im0 = axes[0].imshow(jaccard_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0].set_title("Structural Stability (Jaccard Similarity)", fontsize=11, fontweight="bold")
    axes[0].set_xticks(range(len(seeds)))
    axes[0].set_yticks(range(len(seeds)))
    axes[0].set_xticklabels(labels)
    axes[0].set_yticklabels(labels)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    for i in range(len(seeds)):
        for j in range(len(seeds)):
            val = jaccard_mat[i, j]
            color = "white" if val > 0.6 else "black"
            axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    # 2. Functional Stability (Pearson r)
    im1 = axes[1].imshow(functional_mat, cmap="YlGnBu", vmin=-0.2, vmax=1.0)
    axes[1].set_title("Functional Stability (Causal Vector r)", fontsize=11, fontweight="bold")
    axes[1].set_xticks(range(len(seeds)))
    axes[1].set_yticks(range(len(seeds)))
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for i in range(len(seeds)):
        for j in range(len(seeds)):
            val = functional_mat[i, j]
            color = "white" if val > 0.6 else "black"
            axes[1].text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    out_file = output_dir / "circuit_stability_matrix.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[SAVED] Dual stability matrix saved to {out_file}")
    return out_file


def run_cross_seed_analysis(
    seeds: List[int] = [42, 123, 456, 789, 1000],
    epochs: int = 15,
    retention_threshold: float = 0.90,
    results_dir: Path = Path("results/stability"),
) -> Dict:
    """
    Executes Phase 16, 17, and 18 across multiple random seeds.
    Trains models, patches heads, discovers minimal circuits, and analyzes cross-seed stability.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    circuits: Dict[int, Set[str]] = {}
    causal_vectors: Dict[int, np.ndarray] = {}
    circuit_summaries: List[Dict] = []

    print(f"=== Starting Multi-Seed Analysis for Seeds: {seeds} ===")

    for seed in seeds:
        print(f"\n==========================================")
        print(f"--> Processing Random Seed: {seed}")
        print(f"==========================================")

        t_cfg = TrainConfig(seed=seed, epochs=epochs)
        m_cfg = ModelConfig()
        ckpt_path = t_cfg.checkpoint_dir / f"model_seed{seed}.pt"
        patch_csv = Path(f"results/patching/seed{seed}.csv")
        circ_json = Path(f"results/circuits/seed{seed}.json")

        # 1. Train if checkpoint does not exist
        if not ckpt_path.exists():
            print(f"Training baseline model for seed {seed}...")
            train_baseline(t_cfg, m_cfg)

        # 2. Patch activations if CSV does not exist
        if not patch_csv.exists():
            print(f"Running activation patching for seed {seed}...")
            df_patch = run_activation_patching_experiment(
                checkpoint_path=ckpt_path,
                seed=seed,
                output_dir=Path("results/patching"),
            )
        else:
            df_patch = pd.read_csv(patch_csv)

        # Render individual causal heatmap
        plot_causal_heatmap(df_patch, seed=seed)

        # Cache causal vector (ordered by component name: L0H0, L0H1, ...)
        df_sorted = df_patch.sort_values(by=["layer", "head"])
        causal_vectors[seed] = df_sorted["causal_effect"].to_numpy()

        # 3. Discover minimal circuit if JSON does not exist
        if not circ_json.exists():
            print(f"Discovering minimal circuit for seed {seed}...")
            circ_data = run_circuit_discovery(
                checkpoint_path=ckpt_path,
                patching_csv=patch_csv,
                seed=seed,
                retention_threshold=retention_threshold,
                output_dir=Path("results/circuits"),
            )
        else:
            with open(circ_json, "r") as f:
                circ_data = json.load(f)

        circuits[seed] = set(circ_data["circuit_components"])
        circuit_summaries.append({
            "seed": seed,
            "circuit_size": circ_data["circuit_size"],
            "circuit_components": ", ".join(circ_data["circuit_components"]),
            "baseline_accuracy": circ_data["baseline_accuracy"],
            "sufficiency_accuracy": circ_data["metrics"]["sufficiency_accuracy"],
        })

    # Phase 17: Structural Stability (Jaccard)
    jaccard_mat, mean_jaccard, std_jaccard = compute_structural_stability_matrix(circuits, seeds)
    
    # Phase 18: Functional Stability (Correlation)
    functional_mat, mean_corr, std_corr = compute_functional_stability_matrix(causal_vectors, seeds)

    # Plot figure
    plot_dual_stability_matrices(jaccard_mat, functional_mat, seeds)

    # Save summary tables
    df_table = pd.DataFrame(circuit_summaries)
    df_table.to_csv(Path("results/tables/cross_seed_circuits.csv"), index=False)

    stability_report = {
        "seeds": seeds,
        "mean_structural_jaccard": mean_jaccard,
        "std_structural_jaccard": std_jaccard,
        "mean_functional_correlation": mean_corr,
        "std_functional_correlation": std_corr,
        "circuits": {s: list(circuits[s]) for s in seeds},
    }
    with open(results_dir / "cross_seed_stability.json", "w") as f:
        json.dump(stability_report, f, indent=2)

    print("\n" + "=" * 50)
    print("           CROSS-SEED STABILITY RESULTS           ")
    print("=" * 50)
    print(df_table.to_string(index=False))
    print("-" * 50)
    print(f"Structural Stability (Mean Jaccard): {mean_jaccard:.3f} ± {std_jaccard:.3f}")
    print(f"Functional Stability (Mean Pearson r): {mean_corr:.3f} ± {std_corr:.3f}")
    print("=" * 50)

    return stability_report


if __name__ == "__main__":
    run_cross_seed_analysis()
