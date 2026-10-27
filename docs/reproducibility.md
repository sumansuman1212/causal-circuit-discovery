# Reproducibility Guide

This guide provides step-by-step instructions for reproducing all experimental results, figures, and summary tables on standard CPU hardware or cloud environments.

---

## 1. Supported Environments & Hardware

- **Python:** 3.10, 3.11, 3.12, 3.13 tested.
- **Hardware Requirements:**
  - **CPU Only:** Fully supported. All models are compact ($\approx 420,\!000$ parameters) and evaluate in seconds on a modern multi-core CPU.
  - **RAM:** Minimum 4 GB.
  - **Disk:** $\approx 100$ MB for checkpoints and saved figures.
  - **GPU (Optional):** Supported via PyTorch CUDA, but strictly optional. (For local execution without GPU, keep `device="cpu"`).

---

## 2. Installation Instructions

### Local Setup (CPU)
```bash
# 1. Clone repository
git clone https://github.com/sumansuman1212/causal-circuit-discovery.git
cd causal-circuit-discovery

# 2. Create isolated virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Verify environment
python verify_env.py
```

### Google Colab Setup (Optional GPU)
If you wish to execute large-scale multi-seed runs on a cloud GPU:
```python
# In a Colab notebook cell:
!git clone https://github.com/sumansuman1212/causal-circuit-discovery.git
%cd causal-circuit-discovery
!pip install -r requirements.txt
!python scripts/run_experiment.py --seed 42 --stage all --device cuda
```

---

## 3. Verifying Invariants via Test Suite

Execute the unit test suite to verify module behavior, dataloaders, and hook mechanics:

```bash
pytest -v
```

Expected output:
```
tests/test_ablation.py::test_component_ablator PASSED
tests/test_circuit_search.py::test_minimal_circuit_finder PASSED
tests/test_ood.py::test_ood_task_generator_formats PASSED
tests/test_patching.py::test_activation_patching_dimensions PASSED
tests/test_pipeline.py::test_dataset_generation PASSED
tests/test_pipeline.py::test_model_forward PASSED
tests/test_pipeline.py::test_dataloaders_split PASSED
tests/test_stability.py::test_jaccard_similarity PASSED
tests/test_stability.py::test_stability_matrices PASSED
============================== 9 passed ==============================
```

---

## 4. End-to-End Reproduction Workflow

### Step 1: Discover Minimal Circuit (Seed 42)
```bash
python scripts/run_experiment.py --seed 42 --stage discover --device cpu
```
This loads `models/checkpoints/model_seed42.pt` and `results/patching/seed42.csv`, executes greedy backward pruning, and validates sufficiency and necessity.

### Step 2: Evaluate OOD Generalization
```bash
python scripts/evaluate_circuit.py --seed 42 --num-samples 1000 --device cpu
```
This generates the ID, OOD-1, OOD-2, and OOD-3 test splits, evaluates retention, and updates `results/tables/ood_evaluation.csv` and `results/figures/ood_performance.png`.

### Step 3: Multi-Seed Stability & Analysis
```bash
python scripts/analyze_results.py --seeds 42 123 456 789 1000 --device cpu
```
This computes structural Jaccard and functional Pearson stability matrices across all 5 seeds, updating `results/stability/cross_seed_stability.json` and `results/figures/circuit_stability_matrix.png`.

---

## 5. Random Seeds and Determinism

The pipeline uses fixed random seeds throughout:
- PyTorch: `torch.manual_seed(seed)`
- Python: `random.seed(seed)`
- NumPy: `np.random.seed(seed)`

Evaluated seeds: `[42, 123, 456, 789, 1000]`.
All checkpoint files in `models/checkpoints/` are versioned and directly loadable with `torch.load(..., map_location="cpu")`.
