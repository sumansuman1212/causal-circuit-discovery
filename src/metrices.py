# src/metrics.py
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_bootstrap_ci(data: List[float], n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float]:
    """Computes non-parametric two-sided bootstrap confidence intervals."""
    if len(data) < 2:
        return (float(data[0]), float(data[0])) if data else (0.0, 0.0)
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return low, high


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """Computes Cohen's d effect size between two independent groups."""
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compile_final_tables(
    tables_dir: Path = Path("results/tables"),
    circuits_dir: Path = Path("results/circuits"),
    ablation_csv: Path = Path("results/ablation/seed42.csv"),
    patching_csv: Path = Path("results/patching/seed42.csv"),
):
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. TABLE 1: Model Architecture & Training Hyperparameters
    table1_data = [
        {"Hyperparameter / Configuration": "Context Window (n_ctx)", "Value": "32 tokens"},
        {"Hyperparameter / Configuration": "Layers (n_layers)", "Value": "4"},
        {"Hyperparameter / Configuration": "Attention Heads (n_heads)", "Value": "4 per layer (16 total)"},
        {"Hyperparameter / Configuration": "Model Dimension (d_model)", "Value": "128"},
        {"Hyperparameter / Configuration": "MLP Dimension (d_mlp)", "Value": "512"},
        {"Hyperparameter / Configuration": "Activation Function", "Value": "GELU"},
        {"Hyperparameter / Configuration": "Optimizer", "Value": "AdamW (lr=3e-4, weight_decay=0.01)"},
        {"Hyperparameter / Configuration": "Batch Size", "Value": "64"},
        {"Hyperparameter / Configuration": "Training Dataset Size", "Value": "10,000 synthetic pairs"},
        {"Hyperparameter / Configuration": "Evaluation Splits", "Value": "80% Train, 10% Val, 10% Test"},
    ]
    df_t1 = pd.DataFrame(table1_data)
    df_t1.to_csv(tables_dir / "table1_architecture.csv", index=False)
    print(f"[SAVED] Table 1 saved to {tables_dir / 'table1_architecture.csv'}")

    # 2. TABLE 2: Top Causal Components & Multi-Intervention Validation
    if patching_csv.exists() and ablation_csv.exists():
        df_patch = pd.read_csv(patching_csv)
        df_abl = pd.read_csv(ablation_csv)
        
        # Merge patching and ablation
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
                "Patched Accuracy P(Clean)": p_row["patched_probability"],
                "Ablation Accuracy Drop": abl_drop,
            })
        
        df_t2 = pd.DataFrame(merged).sort_values(by="Causal Effect (CE)", ascending=False)
        df_t2.to_csv(tables_dir / "table2_causal_components.csv", index=False)
        print(f"[SAVED] Table 2 saved to {tables_dir / 'table2_causal_components.csv'}")

    # 3. Compile Multi-Seed Statistical Summary
    circuit_files = list(circuits_dir.glob("seed*.json"))
    if circuit_files:
        sizes, suff_accs, gaps = [], [], []
        for fpath in circuit_files:
            with open(fpath, "r") as f:
                data = json.load(f)
            sizes.append(data["circuit_size"])
            suff_accs.append(data["metrics"]["sufficiency_accuracy"])
            gaps.append(data["baseline_accuracy"] - data["metrics"]["sufficiency_accuracy"])

        ci_size = compute_bootstrap_ci(sizes)
        ci_suff = compute_bootstrap_ci(suff_accs)

        stats_summary = {
            "num_seeds": len(circuit_files),
            "circuit_size_mean": float(np.mean(sizes)),
            "circuit_size_median": float(np.median(sizes)),
            "circuit_size_95ci": list(ci_size),
            "sufficiency_acc_mean": float(np.mean(suff_accs)),
            "sufficiency_acc_95ci": list(ci_suff),
            "performance_gap_mean": float(np.mean(gaps)),
        }
        with open(tables_dir / "multi_seed_aggregate_stats.json", "w") as f:
            json.dump(stats_summary, f, indent=2)
        print(f"[SAVED] Aggregate multi-seed statistics to {tables_dir / 'multi_seed_aggregate_stats.json'}")


if __name__ == "__main__":
    compile_final_tables()
