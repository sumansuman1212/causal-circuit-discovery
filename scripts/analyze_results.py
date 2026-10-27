#!/usr/bin/env python3
"""
CLI entry point to aggregate multi-seed stability metrics and compile results tables.
Usage:
    python scripts/analyze_results.py --seeds 42 123 456 789 1000 --device cpu
"""
import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import compile_final_tables
from src.stability import run_cross_seed_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Seed Analysis and Results Compilation")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1000],
        help="List of random seeds to analyze",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--retention", type=float, default=0.90, help="Retention threshold")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Directory for tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory for figures")
    parser.add_argument("--stability-dir", type=str, default="results/stability", help="Directory for stability JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("Analyzing Multi-Seed Stability & Results")
    print(f"Seeds: {args.seeds} | Device: {args.device}")
    print("==================================================")

    # 1. Run cross-seed stability
    stability_report = run_cross_seed_analysis(
        seeds=args.seeds,
        retention_threshold=args.retention,
        device=args.device,
        results_dir=Path(args.stability_dir),
    )

    # 2. Compile aggregated tables
    compile_final_tables(
        tables_dir=Path(args.tables_dir),
        circuits_dir=Path("results/circuits"),
        ablation_csv=Path("results/ablation/seed42.csv"),
        patching_csv=Path("results/patching/seed42.csv"),
    )

    print("\n[DONE] Multi-seed stability report and summary tables updated successfully.")


if __name__ == "__main__":
    main()
