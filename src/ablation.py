# src/ablation.py
import json
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.circuit_metrics import select_candidate_components
from src.config import ModelConfig, TrainConfig
from src.dataset import SyntheticTaskGenerator, get_dataloaders
from src.evaluate import evaluate_model
from src.model import TinyTransformer


class ComponentAblator:
    def __init__(self, model: TinyTransformer, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.mean_head_activations: Dict[int, torch.Tensor] = {}
# In src/ablation.py

    def compute_head_means(self, loader: DataLoader):
        """
        Computes the dataset-wide mean activation for each layer and head.
        Averages across both batches and sequence positions to allow arbitrary sequence lengths:
        Shape: (1, n_heads, 1, d_head)
        """
        layer_sums = {l: 0.0 for l in range(self.model.cfg.n_layers)}
        total_tokens = 0

        def accumulate_hook(head_out: torch.Tensor, layer_idx: int) -> torch.Tensor:
            # head_out: (B, n_heads, T, d_head)
            # Sum over batch (dim 0) and sequence length (dim 2)
            layer_sums[layer_idx] = layer_sums[layer_idx] + head_out.detach().sum(dim=(0, 2))
            return head_out

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device)
                B, T = x.shape
                total_tokens += B * T
                self.model(x, hook_fn=accumulate_hook)

        # Resulting shape: (1, n_heads, 1, d_head), which broadcasts across any (B, n_heads, T, d_head)
        self.mean_head_activations = {
            l: (layer_sums[l] / total_tokens).unsqueeze(0).unsqueeze(2)
            for l in range(self.model.cfg.n_layers)
        }

    def evaluate_with_ablated_heads(
        self,
        loader: DataLoader,
        ablated_heads: Set[Tuple[int, int]],
        mode: str = "mean",
    ) -> float:
        """
        Evaluates test performance while suppressing the specified heads.
        ablated_heads: Set of (layer, head) tuples to neutralize.
        mode: "mean" or "zero"
        """
        def ablation_hook(head_out: torch.Tensor, layer_idx: int) -> torch.Tensor:
            # head_out: (batch_size, n_heads, seq_len, d_head)
            heads_in_layer = [h for (l, h) in ablated_heads if l == layer_idx]
            if not heads_in_layer:
                return head_out

            ablated = head_out.clone()
            for h in heads_in_layer:
                if mode == "mean" and layer_idx in self.mean_head_activations:
                    # mean_val shape: (1, 1, d_head), broadcastable across any (B, T, d_head)
                    mean_val = self.mean_head_activations[layer_idx][:, h, :, :]
                    ablated[:, h, :, :] = mean_val
                else:
                    ablated[:, h, :, :] = 0.0
            return ablated

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, hook_fn=ablation_hook)[:, -1, :]
                correct += (logits.argmax(dim=-1) == y).sum().item()
                total += len(y)

        accuracy = correct / total
        return accuracy
    


def run_ablation_experiment(
    checkpoint_path: Path = Path("models/checkpoints/model_seed42.pt"),
    patching_csv: Path = Path("results/patching/seed42.csv"),
    seed: int = 42,
    num_random_repeats: int = 10,
    output_dir: Path = Path("results/ablation"),
) -> pd.DataFrame:
    """
    Executes single-candidate ablations, full-candidate set ablation,
    and rigorous random ablation controls.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Dataset & Model
    task_gen = SyntheticTaskGenerator(seed=seed)
    model_cfg = ModelConfig(vocab_size=task_gen.vocab_size)
    clean_data, _ = task_gen.build_dataset(num_samples=5000)
    _, _, test_loader = get_dataloaders(clean_data, batch_size=64)

    model = TinyTransformer(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")

    # 2. Baseline Model Performance
    ablator = ComponentAblator(model, device=device)
    ablator.compute_head_means(test_loader)
    baseline_acc, _ = evaluate_model(model, test_loader, device=device)
    print(f"Clean Baseline Accuracy: {baseline_acc * 100:.2f}%")

    # 3. Load Causal Candidates
    df_patching = pd.read_csv(patching_csv)
    candidates = select_candidate_components(df_patching, threshold=0.20)
    all_heads = [(l, h) for l in range(model_cfg.n_layers) for h in range(model_cfg.n_heads)]
    non_candidates = [head for head in all_heads if head not in candidates]

    records = []

    # 4. Individual Candidate Ablation
    print("\n--- Evaluating Individual Candidate Ablations ---")
    for layer, head in candidates:
        comp_name = f"L{layer}H{head}"
        acc = ablator.evaluate_with_ablated_heads(test_loader, ablated_heads={(layer, head)})
        drop = baseline_acc - acc
        records.append({
            "experiment": "candidate_ablation",
            "component": comp_name,
            "ablated_accuracy": acc,
            "accuracy_drop": drop,
            "drop_percentage": (drop / baseline_acc) * 100,
        })
        print(f"Ablated {comp_name:<6} -> Accuracy: {acc*100:.2f}% (Drop: {drop*100:.2f}%)")

    # 5. Full Candidate Circuit Ablation
    candidate_set = set(candidates)
    joint_acc = ablator.evaluate_with_ablated_heads(test_loader, ablated_heads=candidate_set)
    joint_drop = baseline_acc - joint_acc
    records.append({
        "experiment": "full_circuit_ablation",
        "component": f"ALL_CANDIDATES_{len(candidates)}",
        "ablated_accuracy": joint_acc,
        "accuracy_drop": joint_drop,
        "drop_percentage": (joint_drop / baseline_acc) * 100,
    })
    print(f"\nAblated ALL {len(candidates)} Candidates -> Accuracy: {joint_acc*100:.2f}% (Drop: {joint_drop*100:.2f}%)")

    # 6. Random Control Ablation
    print(f"\n--- Running Random Controls ({num_random_repeats} samples of size k={len(candidates)}) ---")
    random_drops = []
    k = len(candidates)

    for r in range(num_random_repeats):
        rand_heads = set(random.sample(all_heads, k))
        rand_acc = ablator.evaluate_with_ablated_heads(test_loader, ablated_heads=rand_heads)
        rand_drop = baseline_acc - rand_acc
        random_drops.append(rand_drop)

    rand_mean_drop = float(np.mean(random_drops))
    rand_std_drop = float(np.std(random_drops))

    records.append({
        "experiment": "random_control_summary",
        "component": f"RANDOM_{k}_HEADS",
        "ablated_accuracy": baseline_acc - rand_mean_drop,
        "accuracy_drop": rand_mean_drop,
        "drop_percentage": (rand_mean_drop / baseline_acc) * 100,
    })
    print(f"Random Control Mean Drop: {rand_mean_drop*100:.2f}% ± {rand_std_drop*100:.2f}%")

    # 7. Save Results
    df_results = pd.DataFrame(records)
    out_file = output_dir / f"seed{seed}.csv"
    df_results.to_csv(out_file, index=False)
    print(f"[DONE] Saved ablation experiment metrics to {out_file}")

    return df_results
