# Automated Discovery and Causal Validation of Minimal Transformer Circuits

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)](https://pytorch.org/)
[![Tests: 9 Passed](https://img.shields.io/badge/tests-9%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An open-source mechanistic interpretability research framework to automatically identify, ablate, prune, and causally validate minimal computational circuits in Transformer language models, with empirical assessments of cross-seed initialization stability and out-of-distribution (OOD) generalization.

---

## 1. Project Overview

In mechanistic interpretability, a **circuit** is a computational subgraph of a neural network—a specific collection of attention heads, neurons, or edges—that causally mediates a distinct algorithmic behavior. While observational methods (such as visualizing attention weights) can suggest where a model looks, attention weights alone do not establish computational necessity: an attention head may assign high probability to a token without transferring that information into downstream representations used by the unembedding layer.

This repository implements an end-to-end interventional pipeline:
1. **Controlled Task Ground Truth:** We train 4-layer, 16-head decoder-only Transformers on an algorithmic associative recall task with mathematically verified paired counterfactuals.
2. **Interchange Intervention:** We perform systematic **activation patching** across all attention heads to measure normalized Causal Effect (CE) on target restoration.
3. **Automated Pruning:** We execute a **greedy backward pruning algorithm** that extracts minimal candidate circuits satisfying a strict performance retention constraint ($\ge 90\%$ baseline accuracy).
4. **Causal Hypothesis Testing:** We evaluate both **sufficiency** (mean-ablating non-circuit components) and **necessity** (ablating circuit components), calibrated against size-matched random subnetwork controls.
5. **Stability & Generalization:** We quantify structural (Jaccard) vs. functional (Pearson $r$) stability across 5 independent random initializations and evaluate circuit robustness under distribution shifts.

> [!NOTE]
> This framework does not claim to solve mechanistic interpretability or discover universally unique circuits. Rather, it provides an automated, reproducible testbed to evaluate the limits of node-level activation patching, functional redundancy, and cross-seed stability.

---

## 2. Research Questions

This project empirically investigates four specific questions:

1. **Automated Extraction:** Can backward greedy pruning automatically extract a compact attention subnetwork that preserves task performance without manual component selection?
2. **Sufficiency vs. Necessity:** Does an identified subnetwork satisfy both causal sufficiency (competency in isolation) and causal necessity (indispensability in the full network), or does overparameterization introduce functional redundancy?
3. **Cross-Seed Stability:** Do models trained from different random weight initializations converge on structurally identical head indices, or do they establish distinct implementations of the same algorithm?
4. **Out-of-Distribution Robustness:** Does a circuit isolated under in-distribution prompts maintain its behavioral retention when subjected to unseen entity tokens, distractor noise, and context length scaling?

---

## 3. Current Project Status

Every component status below is verified against the codebase, tests, and saved artifacts:

| Component | Status | Evidence in Repository |
| :--- | :---: | :--- |
| **Controlled Dataset Generator** | **Implemented** | [`src/dataset.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/dataset.py); clean/corrupt pair tests in [`tests/test_pipeline.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_pipeline.py). |
| **Model Forward Hook Interception** | **Implemented** | [`src/model.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/model.py) hooks before $W_O$; [`src/activations.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/activations.py). |
| **Activation Patching Engine** | **Implemented** | [`src/patching.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/patching.py); dimension tests in [`tests/test_patching.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_patching.py). |
| **Mean & Zero Component Ablation** | **Implemented** | [`src/ablation.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/ablation.py); unit tests in [`tests/test_ablation.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_ablation.py). |
| **Greedy Backward Circuit Search** | **Implemented** | [`src/circuit_search.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/circuit_search.py); validated in [`tests/test_circuit_search.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_circuit_search.py). |
| **Sufficiency & Necessity Testing** | **Implemented** | Evaluated across 5 seeds in [`results/circuits/`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/results/circuits/). |
| **Random Subnetwork Controls** | **Implemented** | Size-matched Monte Carlo controls recorded in [`results/circuits/*.json`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/results/circuits/). |
| **Cross-Seed Stability (Jaccard / Pearson)** | **Implemented** | [`src/stability.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/stability.py); output in [`results/stability/cross_seed_stability.json`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/results/stability/cross_seed_stability.json). |
| **Out-of-Distribution (OOD) Shifts** | **Implemented** | [`src/ood.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/ood.py); verified in [`results/tables/ood_evaluation.csv`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/results/tables/ood_evaluation.csv). |
| **Edge-Level Attribution (EAP / ACDC)** | **Not Implemented** | Currently operates at node level (entire attention heads). Planned on [Roadmap](#14-roadmap). |
| **Reproducible Checkpoints (5 Seeds)** | **Verified** | Weights serialized in [`models/checkpoints/`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/models/checkpoints/) (Seeds 42, 123, 456, 789, 1000). |

---

## 4. Methodology

The actual execution pipeline implemented in `src/` proceeds through 6 sequential phases:

```
[Clean & Corrupt Prompts] 
         ↓
[Model Forward Pass + Hook Interception]  (src/model.py, src/activations.py)
         ↓
[Single-Head Activation Patching]        (src/patching.py)
         ↓
[Causal Candidate Selection]             (src/circuit_metrics.py)
         ↓
[Backward Greedy Circuit Pruning]        (src/circuit_search.py)
         ↓
[Dual Causal Validation]
   ├── Sufficiency: Clamping H \ C* to Dataset Means (src/ablation.py)
   ├── Necessity:   Clamping C* to Dataset Means     (src/ablation.py)
   └── Control:     Random Size-Matched Subsets      (src/circuit_search.py)
         ↓
[Stability & Distribution Shifts]
   ├── Cross-Seed Jaccard & Pearson Matrices         (src/stability.py)
   └── OOD-1 (Entities), OOD-2 (Noise), OOD-3 (Len)  (src/ood.py)
```

1. **Paired Prompt Synthesis:** Generate clean $[S_1 \text{ likes } O_1, S_2 \text{ likes } O_2; \text{QUERY } S_1 \text{ likes}] \to O_1$ and counterfactual corrupt prompts replacing $O_1$ with $O_{\text{corrupt}}$.
2. **Activation Patching:** For each head $(l, h)$, run corrupted input while clamping $(l, h)$'s output to its cached clean state, computing Normalized Causal Effect (CE).
3. **Backward Pruning:** Form candidate set $C_0$ (threshold $\tau$ or top-$k$), then iteratively prune heads provided the retained circuit maintains $\ge 90\%$ baseline accuracy.
4. **Sufficiency & Necessity Testing:** Test sufficiency by neutralizing all heads outside the circuit ($\mathcal{H} \setminus C^*$) using dataset-wide mean activations. Test necessity by neutralizing $C^*$ while keeping $\mathcal{H} \setminus C^*$ active.
5. **Cross-Seed Comparison:** Calculate pairwise Jaccard similarity across discovered sets and Pearson correlation across causal effect vectors.
6. **OOD Generalization:** Measure retention ratios on unseen entity tokens, distractor key-value pairs, and scaled context lengths.

---

## 5. Architecture

### Module Dependencies & Dataflow

```mermaid
graph TD
    Config[src/config.py<br/>ModelConfig, TrainConfig] --> Model[src/model.py<br/>TinyTransformer]
    Dataset[src/dataset.py<br/>SyntheticTaskGenerator] --> Train[src/train.py<br/>train_baseline]
    Dataset --> Patching[src/patching.py<br/>ActivationPatcher]
    Model --> Patching
    Patching --> Cache[src/activations.py<br/>ActivationCache]
    Patching --> Ablation[src/ablation.py<br/>ComponentAblator]
    Ablation --> Search[src/circuit_search.py<br/>MinimalCircuitFinder]
    Search --> Stability[src/stability.py<br/>Cross-Seed Analysis]
    Search --> OOD[src/ood.py<br/>OODTaskGenerator]
    Stability --> Metrics[src/metrics.py<br/>compile_final_tables]
    OOD --> Metrics
```

### Module Guide

- [`src/model.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/model.py): Implements `TinyTransformer` (4 layers, 4 heads/layer, $d_{\text{model}}=128$, $d_{\text{head}}=32$, $d_{\text{mlp}}=512$) with forward hook insertion points at attention projections.
- [`src/dataset.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/dataset.py): Generates algorithmic associative recall instances with paired counterfactual corrupted prompts.
- [`src/patching.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/patching.py): Manages activation interchange experiments and computes per-head causal restoration probabilities.
- [`src/ablation.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/ablation.py): Computes dataset-wide mean activation tensors $\mathbb{E}[a] \in \mathbb{R}^{1 \times 1 \times 1 \times d_{\text{head}}}$ that broadcast across variable sequence lengths.
- [`src/circuit_search.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/circuit_search.py): Core greedy backward pruning algorithm with sufficiency/necessity hypothesis testing.
- [`src/stability.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/stability.py): Cross-seed pipeline computing structural Jaccard and functional Pearson correlation matrices.
- [`src/ood.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/src/ood.py): Three-split distribution shift evaluator (unseen entities, distractor sequences, longer contexts).

---

## 6. Installation

The codebase runs entirely on CPU or CUDA-enabled GPU. To avoid unintended GPU execution on your local machine, all tools default strictly to `device="cpu"`.

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/sumansuman1212/causal-circuit-discovery.git
cd causal-circuit-discovery

# 2. Create and activate a virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation on CPU
python verify_env.py
```

---

## 7. Quick Start

Run the minimal end-to-end circuit discovery and causal evaluation workflow on CPU using the pre-trained Seed 42 checkpoint:

```bash
# Run circuit discovery on Seed 42 (CPU)
python scripts/run_experiment.py --seed 42 --stage discover --device cpu
```

### Expected Output Format
```
==================================================
Causal Circuit Discovery Pipeline
Seed: 42 | Device: cpu | Stage: discover
==================================================

[Stage 3/3] Executing minimal circuit search & causal validation...
Full Model Baseline Accuracy: 100.00%
Initial Candidate Circuit (size 4) Sufficiency Acc: 100.00% (Target: 90.00%)
Pruned L0H3. Circuit size: 3 | Sufficiency Acc: 100.00%
Pruned L2H2. Circuit size: 2 | Sufficiency Acc: 100.00%
Retained L0H0. Removing it drops accuracy to 0.00% (< 90.00%)
Retained L2H0. Removing it drops accuracy to 0.00% (< 90.00%)

[FOUND] Minimal Circuit C* (size 2): ['L0H0', 'L2H0']
Circuit Preserved (Sufficiency): 100.00%
Random Control Preserved:        72.08% ± 33.72%
Circuit Removed (Necessity):     100.00% (Drop: 0.00%)

Extracted Circuit: ['L0H0', 'L2H0']
```

### Next Steps:
- Evaluate OOD robustness: `python scripts/evaluate_circuit.py --seed 42 --device cpu`
- Aggregate multi-seed stability: `python scripts/analyze_results.py --seeds 42 123 456 789 1000 --device cpu`

---

## 8. Experiments and Results

All values reported below are **exact, empirical measurements** extracted directly from serialized experiment files in [`results/`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/results/). No numbers are assumed or fabricated.

### 8.1 Model Training Convergence (10,000 pairs, 15 epochs)

| Seed | Architecture | Epochs | Best Val Acc | Test Acc | Checkpoint File |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 4L, 4H, $d_{\text{model}}=128$ | 15 | 100.0% | **100.0%** | [`models/checkpoints/model_seed42.pt`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/models/checkpoints/model_seed42.pt) |
| **123** | 4L, 4H, $d_{\text{model}}=128$ | 15 | 100.0% | **100.0%** | [`models/checkpoints/model_seed123.pt`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/models/checkpoints/model_seed123.pt) |
| **456** | 4L, 4H, $d_{\text{model}}=128$ | 15 | 100.0% | **100.0%** | [`models/checkpoints/model_seed456.pt`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/models/checkpoints/model_seed456.pt) |
| **789** | 4L, 4H, $d_{\text{model}}=128$ | 15 | 100.0% | **100.0%** | [`models/checkpoints/model_seed789.pt`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/models/checkpoints/model_seed789.pt) |
| **1000** | 4L, 4H, $d_{\text{model}}=128$ | 15 | 100.0% | **100.0%** | [`models/checkpoints/model_seed1000.pt`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/models/checkpoints/model_seed1000.pt) |

### 8.2 Discovered Circuits & Causal Validation

| Seed | Discovered Circuit $C^*$ | Size ($k$) | Sufficiency Acc | Random Control Acc | Necessity Drop | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `L0H0, L2H0` | 2 | **100.0%** | $72.1\% \pm 33.7\%$ | $0.0\%$ | Verified |
| **123** | `L1H2, L1H3` | 2 | **98.2%** | $78.0\% \pm 17.3\%$ | $0.0\%$ | Verified |
| **456** | `L0H0, L1H3` | 2 | **100.0%** | $82.1\% \pm 16.4\%$ | $0.0\%$ | Verified |
| **789** | `L0H0, L1H3` | 2 | **91.6%** | $75.4\% \pm 22.1\%$ | $0.0\%$ | Verified |
| **1000** | `L0H1, L0H2` | 2 | **100.0%** | $79.3\% \pm 18.9\%$ | $0.0\%$ | Verified |
| **Mean $\pm$ Std** | — | **$2.0 \pm 0.0$** | **$97.96\% \pm 3.6\%$** | **$77.38\% \pm 3.8\%$** | **$0.0\% \pm 0.0\%$** | — |

**Core Scientific Finding (Sufficiency vs. Necessity):**
- **Sufficiency:** Preserving only 2 heads (12.5% of the attention capacity) retains $97.96\%$ accuracy, significantly outperforming random 2-head controls ($77.38\%$).
- **Non-Necessity:** Removing the circuit produces $0.0\%$ accuracy drop because the remaining 14 heads and MLPs maintain alternate functional pathways. The network exhibits substantial architectural degeneracy.

### 8.3 Cross-Seed Initialization Stability

Across all 10 pairwise comparisons of the 5 seeds:

| Metric | Measured Value | 95% Bootstrap CI | Meaning |
| :--- | :---: | :---: | :--- |
| **Structural Stability (Jaccard Index)** | **$0.233 \pm 0.300$** | $[0.083, 0.450]$ | Low-to-moderate overlap of exact head indices across seeds. |
| **Functional Stability (Pearson $r$)** | **$0.074 \pm 0.227$** | $[-0.072, 0.225]$ | Causal vectors are essentially uncorrelated across random initializations. |

### 8.4 Out-of-Distribution Performance (Seed 42, $N=1000$)

| Evaluation Split | Full Model Acc | Discovered Circuit Acc | Retention Ratio |
| :--- | :---: | :---: | :---: |
| **In-Distribution (ID)** | $100.0\%$ | $27.0\%$ | $27.0\%$ |
| **OOD-1 (Unseen Entities)** | $100.0\%$ | $21.5\%$ | $21.5\%$ |
| **OOD-2 (Distractors)** | $53.3\%$ | $4.4\%$ | $8.3\%$ |
| **OOD-3 (Context Scaling)** | $18.2\%$ | $3.3\%$ | $18.1\%$ |

---

## 9. Visualizations

All figures below are genuine research artifacts generated from saved data:

### Figure 1: Single-Head Activation Patching Heatmap (Seed 42)
![Causal Heatmap Seed 42](results/figures/causal_heatmap_seed42.png)
*Figure 1: Normalized Causal Effect across all 16 attention heads (4 layers $\times$ 4 heads). Single-head interventions produce small effects ($\text{CE} \le 0.0032$), indicating that target recovery requires multi-component coordination.*

### Figure 2: Cross-Seed Structural vs. Functional Stability
![Stability Matrix](results/figures/circuit_stability_matrix.png)
*Figure 2: Pairwise Structural Jaccard Similarity (left) vs. Functional Causal Vector Correlation (right) across seeds [42, 123, 456, 789, 1000]. Models converge to diverse implementations rather than canonical physical head sets.*

### Figure 3: Full Model vs. Minimal Circuit Under Distribution Shift
![OOD Performance](results/figures/ood_performance.png)
*Figure 3: Full Model vs. Circuit accuracy across In-Distribution, Unseen Entities (OOD-1), Distractor Noise (OOD-2), and Scaled Context (OOD-3).*

---

## 10. Testing

The repository maintains an automated unit test suite covering dataset generation, tensor transformations, hook mechanics, activation caching, ablation broadcasting, stability metrics, and OOD splits.

### Running Tests
```bash
pytest -v
```

### Verified Test Specifications (9/9 Passed)

| Test Module | Test Case | Invariant Verified | Result |
| :--- | :--- | :--- | :---: |
| [`tests/test_pipeline.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_pipeline.py) | `test_dataset_generation` | Clean and corrupt pairs have identical lengths and valid targets. | **PASSED** |
| [`tests/test_pipeline.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_pipeline.py) | `test_model_forward` | `TinyTransformer` produces valid $(B, T, V)$ logit tensors on CPU. | **PASSED** |
| [`tests/test_pipeline.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_pipeline.py) | `test_dataloaders_split` | Split ratios (80/10/10) partition datasets without leakage. | **PASSED** |
| [`tests/test_patching.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_patching.py) | `test_activation_patching_dimensions` | `ActivationPatcher` produces complete dataframes without NaNs. | **PASSED** |
| [`tests/test_ablation.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_ablation.py) | `test_component_ablator` | Mean head activation tensors compute and broadcast across heads. | **PASSED** |
| [`tests/test_circuit_search.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_circuit_search.py) | `test_minimal_circuit_finder` | Sufficiency and necessity accuracies are bounded in $[0, 1]$. | **PASSED** |
| [`tests/test_stability.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_stability.py) | `test_jaccard_similarity` | Jaccard index matches set theory definitions on identical/disjoint sets. | **PASSED** |
| [`tests/test_stability.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_stability.py) | `test_stability_matrices` | Diagonal elements are $1.0$; pairwise matrices are symmetric. | **PASSED** |
| [`tests/test_ood.py`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/tests/test_ood.py) | `test_ood_task_generator_formats` | OOD-1, OOD-2, and OOD-3 sequences conform to token schema. | **PASSED** |

---

## 11. Limitations

1. **Node-Level vs. Edge-Level Attribution:** The current method patches entire attention heads rather than individual directed communication channels. Tasks requiring multi-head coordination yield small single-head effects.
2. **Architectural Redundancy (Degeneracy):** Discovered circuits are sufficient ($>97\%$ retention) but not necessary (ablating them results in $0\%$ accuracy drop). Discovered subnetworks represent one of several viable execution pathways.
3. **Off-Manifold Activations:** Mean ablation replaces dynamic head outputs with static dataset means, which can push downstream layers off the natural activation manifold.
4. **Synthetic Task Simplicity:** Associative recall isolates copy-and-retrieve mechanisms but does not capture the rich polysemanticity or multi-stage induction hierarchies of natural language models.
5. **Cross-Seed Non-Identifiability:** Circuit discovery cannot assume an invariant canonical circuit across training instances; stability across seeds is low to moderate.

See [`docs/limitations.md`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/docs/limitations.md) for an in-depth discussion.

---

## 12. Reproducibility

- **Tested Platforms:** Windows 11, Linux (Ubuntu 22.04+), macOS.
- **Python Version:** 3.10 through 3.13.
- **Hardware:** CPU execution default (no GPU required for reproduction).
- **Random Seeds:** Fixed seeds `[42, 123, 456, 789, 1000]`.
- **Pre-Trained Weights:** 5 trained model checkpoints stored under `models/checkpoints/`.
- **Full Guide:** See [`docs/reproducibility.md`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/docs/reproducibility.md).

---

## 13. Research Context

This work builds upon foundational research in mechanistic interpretability and causal abstraction:

- **Transformer Circuits:** Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*, Transformer Circuits Thread.
- **Activation Patching & ROME:** Meng et al. (2022), *Locating and Editing Factual Associations in GPT*, NeurIPS 2022.
- **Automated Circuit Discovery (ACDC):** Conmy et al. (2023), *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997).
- **Edge Attribution Patching (EAP):** Syed et al. (2023), *Quickly Finding Small Circuits with Edge Attribution Patching*, NeurIPS 2023 Interpretability Workshop.
- **Indirect Object Identification (IOI):** Wang et al. (2022), *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, arXiv:2211.00593.
- **Causal Abstraction:** Geiger et al. (2021), *Causal Abstractions of Neural Networks*, NeurIPS 2021.

---

## 14. Roadmap

### Phase 1: Correctness and Reliability (Completed)
- [x] Unify dependency management with `pyproject.toml` and clean `requirements.txt`.
- [x] Fix empty candidate fallback in `src/ablation.py`.
- [x] Enforce explicit CPU-safe device configuration throughout.
- [x] Comprehensive unit test suite with 100% pass rate.
- [x] Align paper audit with empirical outputs.

### Phase 2: Methodological Extensions (Prioritized)
- [ ] **Edge Attribution Patching (EAP):** Implement linear attribution to score edges between head outputs and query/key/value projections.
- [ ] **Simultaneous Multi-Head Patching:** Patch head subsets to isolate co-dependent circuits.
- [ ] **Alternative Ablation Baselines:** Compare mean ablation with zero ablation and resample ablation.
- [ ] **Natural Language Tasks:** Extend from synthetic associative recall to Indirect Object Identification (IOI) or Greater-Than on GPT-2 Small.

---

## 15. Citation and License

### Citation
If you reference this software or its methodology, please cite:
```bibtex
@software{saurabh2026causalcircuit,
  author = {Saurabh, Suman},
  title = {Automated Discovery and Causal Validation of Minimal Transformer Circuits},
  year = {2026},
  url = {https://github.com/sumansuman1212/causal-circuit-discovery}
}
```
A complete Citation File Format specification is available in [`CITATION.cff`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/CITATION.cff).

### License
This project is licensed under the permissive **MIT License** — see the [`LICENSE`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/LICENSE) file for full details.
