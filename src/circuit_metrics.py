# src/circuit_metrics.py
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_causal_heatmap(
    df_patching: pd.DataFrame,
    seed: int = 42,
    output_dir: Path = Path("results/figures"),
    title: str = "Attention Head Causal Effect on Target Restoration",
) -> Path:
    """
    Renders an L x H heatmap of causal effects and saves it as high-resolution PNG.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = df_patching["layer"].max() + 1
    n_heads = df_patching["head"].max() + 1
    grid = np.zeros((n_layers, n_heads))

    for _, row in df_patching.iterrows():
        grid[int(row["layer"]), int(row["head"])] = row["causal_effect"]

    plt.figure(figsize=(7, 5), dpi=300)
    im = plt.imshow(grid, cmap="coolwarm", aspect="auto", vmin=-0.2, vmax=max(1.0, grid.max()))
    cbar = plt.colorbar(im)
    cbar.set_label("Causal Effect (CE)", rotation=270, labelpad=15)

    plt.title(f"{title} (Seed {seed})", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Attention Head Index", fontsize=10)
    plt.ylabel("Transformer Layer Index", fontsize=10)
    plt.xticks(np.arange(n_heads), [f"H{i}" for i in range(n_heads)])
    plt.yticks(np.arange(n_layers), [f"L{i}" for i in range(n_layers)])

    # Annotate numeric values in cells
    for i in range(n_layers):
        for j in range(n_heads):
            val = grid[i, j]
            color = "white" if abs(val) > 0.5 * grid.max() else "black"
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    output_path = output_dir / f"causal_heatmap_seed{seed}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] Causal heatmap saved to {output_path}")
    return output_path


def select_candidate_components(
    df_patching: pd.DataFrame,
    threshold: float = 0.20,
    top_k: int = None,
) -> List[Tuple[int, int]]:
    """
    Selects heads with causal effect exceeding threshold or by top-k rank.
    Returns list of (layer, head) tuples.
    """
    sorted_df = df_patching.sort_values(by="causal_effect", ascending=False)
    if top_k is not None:
        candidates = sorted_df.head(top_k)
    else:
        candidates = sorted_df[sorted_df["causal_effect"] >= threshold]

    selected = [(int(row["layer"]), int(row["head"])) for _, row in candidates.iterrows()]
    return selected
