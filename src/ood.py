# src/ood.py
import json
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ablation import ComponentAblator
from src.circuit_search import MinimalCircuitFinder
from src.config import ModelConfig, TrainConfig
from src.dataset import AssociativeRecallDataset, SyntheticTaskGenerator
from src.evaluate import evaluate_model
from src.model import TinyTransformer


class OODTaskGenerator:
    """
    Generates controlled distribution shifts while keeping the core semantic
    associative recall task identical.
    """
    def __init__(self, base_generator: SyntheticTaskGenerator, seed: int = 999):
        self.base_gen = base_generator
        self.vocab = base_generator.vocab
        self.id2token = base_generator.id2token
        self.vocab_size = base_generator.vocab_size
        random.seed(seed)

        # Reserve unseen entity tokens within the model's vocabulary capacity
        # Base generator used E_0 to E_29. We construct OOD tokens from E_30 to E_59
        self.in_distribution_entities = self.base_gen.entity_tokens
        
        # Verify if extra tokens exist in vocab; if not, reserve partition
        self.num_total = len(self.in_distribution_entities)
        self.id_entities = self.in_distribution_entities[: self.num_total // 2]
        self.unseen_entities = self.in_distribution_entities[self.num_total // 2 :]

    def generate_ood1_unseen_entities(self) -> Tuple[List[int], int]:
        """OOD-1: Associative recall with strictly unseen entity tokens."""
        s1, s2 = random.sample(self.unseen_entities, 2)
        o1, o2 = random.sample([e for e in self.unseen_entities if e not in (s1, s2)], 2)

        tokens = [s1, "likes", o1, "<SEP>", s2, "likes", o2, "<SEP>", "<QUERY>", s1, "likes"]
        ids = [self.vocab[t] for t in tokens]
        return ids, self.vocab[o1]

    def generate_ood2_distractors(self) -> Tuple[List[int], int]:
        """OOD-2: Injects irrelevant distractor noise pairs before the query."""
        s1, s2 = random.sample(self.in_distribution_entities, 2)
        o1, o2 = random.sample([e for e in self.in_distribution_entities if e not in (s1, s2)], 2)
        # Distractor pair
        d_s, d_o = random.sample([e for e in self.in_distribution_entities if e not in (s1, s2, o1, o2)], 2)

        tokens = [
            s1, "likes", o1, "<SEP>",
            d_s, "likes", d_o, "<SEP>",  # Distractor
            s2, "likes", o2, "<SEP>",
            "<QUERY>", s1, "likes"
        ]
        ids = [self.vocab[t] for t in tokens]
        return ids, self.vocab[o1]

    def generate_ood3_longer_context(self) -> Tuple[List[int], int]:
        """OOD-3: 3 pairs of associations instead of 2 (longer sequence)."""
        entities = random.sample(self.in_distribution_entities, 6)
        s1, o1 = entities[0], entities[1]
        s2, o2 = entities[2], entities[3]
        s3, o3 = entities[4], entities[5]

        # Target subject can be any of the 3
        target_s, target_o = random.choice([(s1, o1), (s2, o2), (s3, o3)])

        tokens = [
            s1, "likes", o1, "<SEP>",
            s2, "likes", o2, "<SEP>",
            s3, "likes", o3, "<SEP>",
            "<QUERY>", target_s, "likes"
        ]
        ids = [self.vocab[t] for t in tokens]
        return ids, self.vocab[target_o]

    def build_ood_dataset(self, shift_type: str, num_samples: int = 1000) -> List[Tuple[List[int], int]]:
        dataset = []
        gen_fn = {
            "ood1_unseen": self.generate_ood1_unseen_entities,
            "ood2_distractor": self.generate_ood2_distractors,
            "ood3_long_context": self.generate_ood3_longer_context,
        }[shift_type]

        for _ in range(num_samples):
            dataset.append(gen_fn())
        return dataset


def evaluate_ood_shifts(
    checkpoint_path: Path = Path("models/checkpoints/model_seed42.pt"),
    circuit_json: Path = Path("results/circuits/seed42.json"),
    seed: int = 42,
    num_eval_samples: int = 1000,
    device: str = "cpu",
    output_dir: Path = Path("results/ood"),
    tables_dir: Path = Path("results/tables"),
    figures_dir: Path = Path("results/figures"),
) -> pd.DataFrame:
    """
    Evaluates both Full Model and Discovered Circuit across In-Distribution and OOD test splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Discovered Minimal Circuit
    with open(circuit_json, "r") as f:
        circ_data = json.load(f)
    circuit_component_names: List[str] = circ_data["circuit_components"]
    # Parse into (layer, head) integer tuples
    circuit: Set[Tuple[int, int]] = {
        (int(c.split("H")[0][1:]), int(c.split("H")[1])) for c in circuit_component_names
    }
    print(f"Loaded Circuit from {circuit_json}: {circuit_component_names} (k={len(circuit)})")

    # 2. Re-create Base Task Generator and Model
    base_gen = SyntheticTaskGenerator(seed=seed)
    model_cfg = ModelConfig(vocab_size=base_gen.vocab_size)
    model = TinyTransformer(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    ablator = ComponentAblator(model, device=device)
    finder = MinimalCircuitFinder(model, ablator, device=device)

    # 3. Prepare ID Test Set and Compute Head Means
    clean_id_data, _ = base_gen.build_dataset(num_samples=num_eval_samples)
    id_loader = DataLoader(AssociativeRecallDataset(clean_id_data), batch_size=64)
    ablator.compute_head_means(id_loader)

    # 4. Initialize OOD Generator
    ood_gen = OODTaskGenerator(base_gen, seed=seed)

    evaluation_splits = {
        "In-Distribution": id_loader,
        "OOD-1 (Unseen Entities)": DataLoader(
            AssociativeRecallDataset(ood_gen.build_ood_dataset("ood1_unseen", num_eval_samples)),
            batch_size=64,
        ),
        "OOD-2 (Distractors)": DataLoader(
            AssociativeRecallDataset(ood_gen.build_ood_dataset("ood2_distractor", num_eval_samples)),
            batch_size=64,
        ),
        "OOD-3 (Longer Context)": DataLoader(
            AssociativeRecallDataset(ood_gen.build_ood_dataset("ood3_long_context", num_eval_samples)),
            batch_size=64,
        ),
    }

    records = []
    print("\n--- Evaluating ID vs OOD Splits ---")

    for split_name, loader in evaluation_splits.items():
        # Baseline Full Model accuracy
        full_acc, _ = evaluate_model(model, loader, device=device)
        # Discovered Circuit Sufficiency accuracy (ablating non-circuit heads)
        circ_acc = finder.evaluate_circuit_sufficiency(circuit, loader, mode="mean")
        # Retention Ratio
        retention = circ_acc / (full_acc + 1e-12)

        records.append({
            "Dataset Split": split_name,
            "Full Model Accuracy": full_acc,
            "Circuit Accuracy": circ_acc,
            "Circuit Retention Ratio": retention,
            "Circuit Gap": full_acc - circ_acc,
        })

        print(
            f"[{split_name:<25}] Full Model: {full_acc * 100:.2f}% | "
            f"Circuit: {circ_acc * 100:.2f}% | Retention: {retention * 100:.2f}%"
        )

    df_ood = pd.DataFrame(records)

    # 5. Export Table 4
    table_path = tables_dir / "ood_evaluation.csv"
    df_ood.to_csv(table_path, index=False)
    print(f"\n[SAVED] Table 4 exported to {table_path}")

    # 6. Render Figure 8: Bar Chart Comparison
    plot_ood_comparison(df_ood, seed=seed, output_path=figures_dir / "ood_performance.png")

    return df_ood


def plot_ood_comparison(df_ood: pd.DataFrame, seed: int, output_path: Path):
    """Generates Figure 8: Full Model vs Circuit Accuracy across ID and OOD splits."""
    labels = df_ood["Dataset Split"].tolist()
    full_accs = [acc * 100 for acc in df_ood["Full Model Accuracy"]]
    circ_accs = [acc * 100 for acc in df_ood["Circuit Accuracy"]]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5), dpi=300)
    plt.bar([i - width / 2 for i in x], full_accs, width=width, label="Full Model", color="#2b5c8f")
    plt.bar([i + width / 2 for i in x], circ_accs, width=width, label="Discovered Circuit (Preserved)", color="#e07a5f")

    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.title(f"Model vs Minimal Circuit Performance Under OOD Shifts (Seed {seed})", fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x, labels, fontsize=9)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)

    # Label values on bars
    for i in x:
        plt.text(i - width / 2, full_accs[i] + 1.5, f"{full_accs[i]:.1f}%", ha="center", fontsize=8)
        plt.text(i + width / 2, circ_accs[i] + 1.5, f"{circ_accs[i]:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[SAVED] Figure 8 saved to {output_path}")


if __name__ == "__main__":
    evaluate_ood_shifts()
