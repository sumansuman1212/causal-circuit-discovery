# src/metrics.py
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_bootstrap_ci(data: List[float], n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    if len(data) < 2:
        return (float(data[0]), float(data[0])) if data else (0.0, 0.0)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1.0 - alpha) * 100))


def compile_final_tables(
    tables_dir: Path = Path("results/tables"),
    circuits_dir: Path = Path("results/circuits"),
    ablation_csv: Path = Path("results/ablation/seed42.csv"),
    patching_csv: Path = Path("results/patching/seed42.csv"),
):
    tables_dir.mkdir(parents=True, exist_ok=True)

    table1_data = [
        {"Configuration": "Context Window (n_ctx)", "Value": "32 tokens"},
        {"Configuration": "Layers (n_layers)", "Value": "4"},
        {"Configuration": "Attention Heads", "Value": "4 per layer (16 total)"},
        {"Configuration": "Model Dimension (d_model)", "Value": "128"},
        {"Configuration": "MLP Dimension (d_mlp)", "Value": "512"},
        {"Configuration": "Activation Function", "Value": "GELU"},
        {"Configuration": "Optimizer", "Value": "AdamW (lr=3e-4, weight_decay=0.01)"},
        {"Configuration": "Batch Size", "Value": "64"},
        {"Configuration": "Dataset Size", "Value": "10,000 synthetic pairs"},
    ]
    pd.DataFrame(table1_data).to_csv(tables_dir / "table1_architecture.csv", index=False)

    if patching_csv.exists() and ablation_csv.exists():
        df_patch = pd.read_csv(patching_csv)
        df_abl = pd.read_csv(ablation_csv)
        merged = []
        for _, p_row in df_patch.iterrows():
            comp = p_row["component"]
            abl_match = df_abl[df_abl["component"] == comp]
            abl_drop = abl_match["accuracy_drop"].values[0] if len(abl_match) > 0 else np.nan
            merged.append({
                "Component": comp,
                "Layer": int(p_row["layer"]),
                "Head": int(p_row["head"]),
                "Causal Effect (CE)": p_row["causal_effect"],
                "Patched Probability": p_row["patched_probability"],
                "Ablation Drop": abl_drop,
            })
        pd.DataFrame(merged).sort_values(by="Causal Effect (CE)", ascending=False).to_csv(
            tables_dir / "table2_causal_components.csv", index=False
        )

    circuit_files = list(circuits_dir.glob("seed*.json"))
    if circuit_files:
        sizes, suff_accs, gaps = [], [], []
        for fpath in circuit_files:
            with open(fpath, "r") as f:
                data = json.load(f)
            sizes.append(data["circuit_size"])
            suff_accs.append(data["metrics"]["sufficiency_accuracy"])
            gaps.append(data["baseline_accuracy"] - data["metrics"]["sufficiency_accuracy"])

        stats_summary = {
            "num_seeds": len(circuit_files),
            "circuit_size_mean": float(np.mean(sizes)),
            "circuit_size_median": float(np.median(sizes)),
            "circuit_size_95ci": list(compute_bootstrap_ci(sizes)),
            "sufficiency_acc_mean": float(np.mean(suff_accs)),
            "sufficiency_acc_95ci": list(compute_bootstrap_ci(suff_accs)),
            "performance_gap_mean": float(np.mean(gaps)),
        }
        with open(tables_dir / "multi_seed_aggregate_stats.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

    print("[SUCCESS] src/metrics.py created and tables ready.")

if __name__ == "__main__":
    compile_final_tables()
