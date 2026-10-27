# src/circuit_search.py
import json
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ablation import ComponentAblator
from src.circuit_metrics import select_candidate_components
from src.config import ModelConfig, TrainConfig
from src.dataset import SyntheticTaskGenerator, get_dataloaders
from src.evaluate import evaluate_model
from src.model import TinyTransformer


class MinimalCircuitFinder:
    def __init__(self, model: TinyTransformer, ablator: ComponentAblator, device: str = "cpu"):
        self.model = model.to(device)
        self.ablator = ablator
        self.device = device
        self.all_heads: Set[Tuple[int, int]] = {
            (l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
        }

    def evaluate_circuit_sufficiency(
        self,
        circuit: Set[Tuple[int, int]],
        loader: DataLoader,
        mode: str = "mean",
    ) -> float:
        """
        Sufficiency Test:
        Keep ONLY the circuit active. Everything outside the circuit is ablated.
        Complement ablated = All Heads - Circuit.
        """
        complement = self.all_heads - circuit
        acc = self.ablator.evaluate_with_ablated_heads(loader, ablated_heads=complement, mode=mode)
        return acc

    def evaluate_circuit_necessity(
        self,
        circuit: Set[Tuple[int, int]],
        loader: DataLoader,
        mode: str = "mean",
    ) -> float:
        """
        Necessity Test:
        Remove ONLY the circuit. Everything else remains active.
        """
        acc = self.ablator.evaluate_with_ablated_heads(loader, ablated_heads=circuit, mode=mode)
        return acc

    def find_minimal_circuit(
        self,
        candidate_heads: List[Tuple[int, int]],
        loader: DataLoader,
        baseline_acc: float,
        retention_ratio: float = 0.90,
    ) -> Tuple[Set[Tuple[int, int]], Dict[str, float]]:
        """
        Greedy backward search: Starts with candidate set, removes heads one by one
        as long as sufficiency accuracy >= retention_ratio * baseline_acc.
        """
        target_acc = baseline_acc * retention_ratio
        current_circuit = set(candidate_heads)

        # Baseline check with full candidate set
        initial_acc = self.evaluate_circuit_sufficiency(current_circuit, loader)
        print(f"Initial Candidate Circuit (size {len(current_circuit)}) Sufficiency Acc: {initial_acc*100:.2f}% (Target: {target_acc*100:.2f}%)")

        if initial_acc < target_acc:
            print("[WARN] Full candidate set cannot reach sufficiency threshold. Returning all candidates.")
            return current_circuit, {"sufficiency_acc": initial_acc}

        # Greedy pruning
        pruning_order = list(current_circuit)
        # Sort candidates ascending by single causal effect if available, or try each
        for head in pruning_order:
            trial_circuit = current_circuit - {head}
            trial_acc = self.evaluate_circuit_sufficiency(trial_circuit, loader)

            if trial_acc >= target_acc:
                current_circuit = trial_circuit
                print(f"Pruned {f'L{head[0]}H{head[1]}'}. Circuit size: {len(current_circuit)} | Sufficiency Acc: {trial_acc*100:.2f}%")
            else:
                print(f"Retained {f'L{head[0]}H{head[1]}'}. Removing it drops accuracy to {trial_acc*100:.2f}% (< {target_acc*100:.2f}%)")

        final_suff_acc = self.evaluate_circuit_sufficiency(current_circuit, loader)
        return current_circuit, {"sufficiency_acc": final_suff_acc}


def run_circuit_discovery(
    checkpoint_path: Path = Path("models/checkpoints/model_seed42.pt"),
    patching_csv: Path = Path("results/patching/seed42.csv"),
    seed: int = 42,
    retention_threshold: float = 0.90,
    num_random_controls: int = 10,
    device: str = "cpu",
    output_dir: Path = Path("results/circuits"),
) -> Dict:
    """
    Discovers minimal circuit, verifies sufficiency, necessity, and random controls, then dumps JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Dataset & Model
    task_gen = SyntheticTaskGenerator(seed=seed)
    model_cfg = ModelConfig(vocab_size=task_gen.vocab_size)
    clean_data, _ = task_gen.build_dataset(num_samples=5000)
    _, _, test_loader = get_dataloaders(clean_data, batch_size=64)

    model = TinyTransformer(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    ablator = ComponentAblator(model, device=device)
    ablator.compute_head_means(test_loader)
    baseline_acc, _ = evaluate_model(model, test_loader, device=device)
    print(f"Full Model Baseline Accuracy: {baseline_acc * 100:.2f}%")

    # 2. Candidate Selection (Patching CE >= 0.15)
    df_patching = pd.read_csv(patching_csv)
    candidates = select_candidate_components(df_patching, threshold=0.15)
    if not candidates:
        candidates = select_candidate_components(df_patching, top_k=4)

    # 3. Minimal Circuit Search
    finder = MinimalCircuitFinder(model, ablator, device=device)
    minimal_circuit, search_meta = finder.find_minimal_circuit(
        candidate_heads=candidates,
        loader=test_loader,
        baseline_acc=baseline_acc,
        retention_ratio=retention_threshold,
    )

    k = len(minimal_circuit)
    circuit_labels = sorted([f"L{l}H{h}" for l, h in minimal_circuit])
    print(f"\n[FOUND] Minimal Circuit C* (size {k}): {circuit_labels}")

    # 4. Rigorous Hypothesis Testing: Sufficiency, Necessity, Controls
    sufficiency_acc = finder.evaluate_circuit_sufficiency(minimal_circuit, test_loader)
    necessity_acc = finder.evaluate_circuit_necessity(minimal_circuit, test_loader)
    necessity_drop = baseline_acc - necessity_acc

    # Random Size-Matched Sufficiency Controls
    random_sufficiencies = []
    for _ in range(num_random_controls):
        rand_circuit = set(random.sample(list(finder.all_heads), k))
        rand_suff = finder.evaluate_circuit_sufficiency(rand_circuit, test_loader)
        random_sufficiencies.append(rand_suff)

    rand_mean_suff = float(np.mean(random_sufficiencies))
    rand_std_suff = float(np.std(random_sufficiencies))

    # 5. Compile and Export Specification
    circuit_summary = {
        "seed": seed,
        "circuit_size": k,
        "circuit_components": circuit_labels,
        "baseline_accuracy": baseline_acc,
        "target_retention_ratio": retention_threshold,
        "metrics": {
            "sufficiency_accuracy": sufficiency_acc,
            "sufficiency_preservation_ratio": sufficiency_acc / (baseline_acc + 1e-12),
            "necessity_ablated_accuracy": necessity_acc,
            "necessity_accuracy_drop": necessity_drop,
            "random_control_sufficiency_mean": rand_mean_suff,
            "random_control_sufficiency_std": rand_std_suff,
        },
        "search_parameters": {
            "initial_candidates_count": len(candidates),
            "algorithm": "greedy_backward_pruning",
        }
    }

    out_file = output_dir / f"seed{seed}.json"
    with open(out_file, "w") as f:
        json.dump(circuit_summary, f, indent=2)

    print(f"[SAVED] Minimal circuit definition exported to {out_file}")
    print("\n--- Summary Evaluation ---")
    print(f"Circuit Preserved (Sufficiency): {sufficiency_acc*100:.2f}%")
    print(f"Random Control Preserved:       {rand_mean_suff*100:.2f}% ± {rand_std_suff*100:.2f}%")
    print(f"Circuit Removed (Necessity):     {necessity_acc*100:.2f}% (Drop: {necessity_drop*100:.2f}%)")

    return circuit_summary


if __name__ == "__main__":
    t_cfg = TrainConfig()
    ckpt = t_cfg.checkpoint_dir / f"model_seed{t_cfg.seed}.pt"
    run_circuit_discovery(checkpoint_path=ckpt, seed=t_cfg.seed)
