#!/usr/bin/env python3
"""
CLI entry point to run circuit discovery experiments.
Usage:
    python scripts/run_experiment.py --seed 42 --stage all --device cpu
"""
import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainConfig
from src.train import train_baseline
from src.patching import run_activation_patching_experiment
from src.circuit_search import run_circuit_discovery


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Circuit Discovery Pipeline Runner")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data, model, and sampling")
    parser.add_argument(
        "--stage",
        type=str,
        default="discover",
        choices=["train", "patch", "discover", "all"],
        help="Experiment stage to execute",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--retention", type=float, default=0.90, help="Performance retention threshold (alpha)")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Path to checkpoints")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output results")
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    res_dir = Path(args.results_dir)
    ckpt_path = ckpt_dir / f"model_seed{seed}.pt"
    patch_csv = res_dir / "patching" / f"seed{seed}.csv"

    print(f"==================================================")
    print(f"Causal Circuit Discovery Pipeline")
    print(f"Seed: {seed} | Device: {device} | Stage: {args.stage}")
    print(f"==================================================")

    if args.stage in ["train", "all"]:
        print(f"\n[Stage 1/3] Training baseline model (seed {seed})...")
        t_cfg = TrainConfig(seed=seed, epochs=args.epochs, device=device, checkpoint_dir=ckpt_dir)
        m_cfg = ModelConfig()
        train_baseline(t_cfg, m_cfg)

    if args.stage in ["patch", "all"]:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run with --stage train first.")
        print(f"\n[Stage 2/3] Running activation patching...")
        run_activation_patching_experiment(
            checkpoint_path=ckpt_path,
            seed=seed,
            device=device,
            output_dir=res_dir / "patching",
        )

    if args.stage in ["discover", "all"]:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run with --stage train first.")
        if not patch_csv.exists():
            raise FileNotFoundError(f"Patching CSV not found at {patch_csv}. Run with --stage patch first.")
        print(f"\n[Stage 3/3] Executing minimal circuit search & causal validation...")
        circuit_info = run_circuit_discovery(
            checkpoint_path=ckpt_path,
            patching_csv=patch_csv,
            seed=seed,
            retention_threshold=args.retention,
            device=device,
            output_dir=res_dir / "circuits",
        )
        print("\nExtracted Circuit:", circuit_info["circuit_components"])


if __name__ == "__main__":
    main()
