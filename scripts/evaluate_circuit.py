#!/usr/bin/env python3
"""
CLI entry point to evaluate a discovered circuit against in-distribution
and out-of-distribution (OOD) test sets.
Usage:
    python scripts/evaluate_circuit.py --seed 42 --device cpu
"""
import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ood import evaluate_ood_shifts


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Discovered Minimal Circuit & OOD Generalization")
    parser.add_argument("--seed", type=int, default=42, help="Seed index of model and circuit to evaluate")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of test evaluation samples per split")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Path to checkpoints")
    parser.add_argument("--circuits-dir", type=str, default="results/circuits", help="Path to circuits JSON")
    parser.add_argument("--output-dir", type=str, default="results/ood", help="Directory for OOD results")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Directory for output tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory for output figures")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    device = args.device
    ckpt_path = Path(args.checkpoint_dir) / f"model_seed{seed}.pt"
    circ_json = Path(args.circuits_dir) / f"seed{seed}.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")
    if not circ_json.exists():
        raise FileNotFoundError(f"Circuit JSON not found at {circ_json}. Discover circuit first.")

    print(f"==================================================")
    print(f"Evaluating Circuit & Distribution Shift Robustness")
    print(f"Seed: {seed} | Device: {device} | Samples: {args.num_samples}")
    print(f"==================================================")

    df_ood = evaluate_ood_shifts(
        checkpoint_path=ckpt_path,
        circuit_json=circ_json,
        seed=seed,
        num_eval_samples=args.num_samples,
        device=device,
        output_dir=Path(args.output_dir),
        tables_dir=Path(args.tables_dir),
        figures_dir=Path(args.figures_dir),
    )

    print("\n--- Summary Evaluation Table ---")
    print(df_ood.to_string(index=False))


if __name__ == "__main__":
    main()
