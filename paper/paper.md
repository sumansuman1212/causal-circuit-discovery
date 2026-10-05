# Automated Discovery and Causal Validation of Minimal Transformer Circuits: Stability Across Initialization and Distribution Shift

**Author:** Independent AI Safety & Mechanistic Interpretability Laboratory  
**Date:** October 2026  
**Repository:** `causal-circuit-discovery`

---

## Abstract

Mechanistic interpretability aims to reverse-engineer deep neural networks into human-understandable computational subgraphs, or *circuits*. However, existing circuit identification often relies on manual inspection or lacks validation across initialization seeds and distributional perturbations. In this work, we introduce an end-to-end framework for the automated discovery, causal validation, and stability assessment of minimal Transformer circuits. Using a controlled associative recall task with an algorithmic ground truth, we train small decoder-only Transformers (4 layers, 16 attention heads) to $>95\%$ behavioral accuracy. We apply systematic activation patching across all attention heads to compute direct causal effects on target token recovery, select causal candidates, and execute a backward greedy pruning algorithm to isolate the minimal circuit $C^* = \arg\min \vert{}C\vert{}$ satisfying a $\ge 90\%$ behavioral retention constraint. 

Across 5 independently trained seeds, we reveal two key empirical findings: (1) **Structural vs. Functional Divergence:** While structural circuit similarity across seeds is moderate (Mean Jaccard Index $J = 0.47 \pm 0.12$), functional stability measured via causal intervention-response correlation is high ($r = 0.89 \pm 0.05$), demonstrating that independently initialized models converge on equivalent functional pathways implemented by non-identical heads. (2) **Out-of-Distribution Vulnerability:** Under distribution shifts (unseen entity tokens, sequence distractors, and context length scaling), isolated circuits retain performance on unseen tokens ($>90\%$), but experience steep degradation under distractor noise, isolating the specific sub-network components responsible for contextual retrieval failure.

---

## 1. Introduction

A central challenge in AI safety and mechanistic interpretability is moving beyond post-hoc correlational observations toward verifiable causal mechanisms. While attention pattern visualizations frequently suggest which tokens a model attends to, correlation does not imply computational necessity: an attention head may display sharp attention weights toward an informative token without transmitting that information into downstream representations used by the final unembedding layer.

To rigorously understand model internals, we must adopt an interventional approach: intervening directly on internal activation manifolds and measuring the causal effect on behavioral logits. This paper answers three foundational questions:
1. *Can minimal computational circuits be automatically extracted without manual heuristic identification?*
2. *Do discovered minimal circuits remain structurally identical across independent random weight initializations, or do models discover alternate algorithmic implementations?*
3. *Does an identified in-distribution circuit remain sufficient when the model faces out-of-distribution (OOD) contexts?*

---

## 2. Experimental Task and Architecture

### 2.1 Controlled Associative Recall
To eliminate ambiguity about the ground-truth computational mechanism, we design an algorithmic associative recall task. Each prompt presents key-value associations followed by a query prompt:
$$\mathcal{X}_{\text{clean}} = [S_1 \text{ likes } O_1 \ , \ S_2 \text{ likes } O_2 \ ; \ \text{QUERY } S_1 \text{ likes}] \implies \mathcal{Y}^* = O_1$$

To enable interchange intervention, we construct paired counterfactual prompts $\mathcal{X}_{\text{corrupt}}$ where the target object is replaced:
$$\mathcal{X}_{\text{corrupt}} = [S_1 \text{ likes } O_{\text{corrupt}} \ , \ S_2 \text{ likes } O_2 \ ; \ \text{QUERY } S_1 \text{ likes}] \implies \mathcal{Y}^*_{\text{corrupt}} = O_{\text{corrupt}}$$

### 2.2 Model Specifications
We utilize a small, clean decoder-only Transformer implemented from first principles to guarantee precise forward hook injection at each attention head projection ($W_O$ output).
- **Depth:** 4 Layers ($L \in \{0, 1, 2, 3\}$)
- **Heads per layer:** 4 Heads ($H \in \{0, 1, 2, 3\}$), total of 16 attention heads
- **Hidden dimension:** $d_{\text{model}} = 128$, $d_{\text{head}} = 32$, $d_{\text{mlp}} = 512$
- **Total Parameters:** $\approx 420,\!000$

All models are trained with AdamW ($\text{lr}=3\times 10^{-4}$, weight decay $0.01$) for 15 epochs until convergence ($>90\%$ test accuracy).

---

## 3. Methodology


### 3.1 Activation Patching and Causal Effect
For each attention head $(l, h)$ in layer $l \in [0, 3]$ and head $h \in [0, 3]$, we run the clean prompt $\mathcal{X}_{\text{clean}}$ and cache head activations $a_{l, h}^{\text{clean}} \in \mathbb{R}^{B \times T \times d_{\text{head}}}$. We then evaluate the corrupted input $\mathcal{X}_{\text{corrupt}}$ while replacing head $(l, h)$'s output with $a_{l, h}^{\text{clean}}$.

The normalized **Causal Effect (CE)** is defined as:
$$\text{CE}(l, h) = \frac{P_{\text{patched}}(\mathcal{Y}^*_{\text{clean}}) - P_{\text{corrupt}}(\mathcal{Y}^*_{\text{clean}})}{P_{\text{clean}}(\mathcal{Y}^*_{\text{clean}}) - P_{\text{corrupt}}(\mathcal{Y}^*_{\text{clean}})}$$
A score of $\text{CE} \approx 1.0$ indicates that restoring head $(l, h)$'s activation single-handedly restores clean behavioral retrieval on the corrupted background.

### 3.2 Candidate Selection and Minimal Circuit Search
We define candidates as heads with $\text{CE}(l, h) \ge \tau$ (default $\tau = 0.15$). We formulate minimal circuit identification as a constrained optimization problem:
$$C^* = \arg\min_{C \subseteq \mathcal{H}} \vert{}C\vert{} \quad \text{subject to} \quad \text{Sufficiency}(C) \ge \alpha \cdot \text{BaselineAccuracy}$$
where $\mathcal{H}$ is the set of all 16 attention heads, and $\alpha = 0.90$ represents a $90\%$ performance retention threshold. 

We apply backward greedy pruning: starting from candidate set $C_0$, we sequentially prune heads whose removal preserves the sufficiency condition. 

### 3.3 Sufficiency and Necessity Interventions
- **Sufficiency Test:** All heads $\mathcal{H} \setminus C^*$ outside the circuit are mean-ablated (replaced by their dataset-wide expected activation $\mathbb{E}[a]$). If the model retains accuracy, $C^*$ is sufficient.
- **Necessity Test:** The circuit $C^*$ is mean-ablated while leaving $\mathcal{H} \setminus C^*$ active. A sharp accuracy collapse demonstrates necessity.
- **Random Control:** Size-matched subsets $C_{\text{rand}} \subset \mathcal{H}$ ($\vert{}C_{\text{rand}}\vert{} = \vert{}C^*\vert{}$) are evaluated over 10 trials to rule out non-specific preservation.

### 3.4 Cross-Seed Stability Formulations
We evaluate cross-seed stability across 5 independent seeds:
1. **Structural Stability (Jaccard Index):**
   $$J(C_i, C_j) = \frac{\vert{}C_i \cap C_j\vert{}}{\vert{}C_i \cup C_j\vert{}}$$
2. **Functional Stability (Pearson Correlation):**
   $$r(\mathbf{v}_i, \mathbf{v}_j) = \frac{\sum_{k=1}^{16} (v_{i, k} - \bar{v}_i)(v_{j, k} - \bar{v}_j)}{\sqrt{\sum_{k=1}^{16} (v_{i, k} - \bar{v}_i)^2 \sum_{k=1}^{16} (v_{j, k} - \bar{v}_j)^2}}$$
   where $\mathbf{v}_i \in \mathbb{R}^{16}$ is the full vector of head causal effects.

---

## 4. Empirical Results

### 4.1 Causal Effect Localization
Activation patching reveals that causal effect is highly concentrated in late-intermediate layers.

![Causal Heatmap](../results/figures/causal_heatmap_seed42.png)
*Figure 1: Attention Head Causal Effects (Seed 42). Intervention reveals that associative retrieval is localized primarily in Layer 2 (L2H1) and Layer 3 (L3H0), while early layers exhibit near-zero direct causal effect.*

### 4.2 Candidate Ablation and Controls
Mean ablation validates that candidate components are causally necessary for target output:
- **Baseline Accuracy:** $96.40\%$
- **Ablating Minimal Circuit:** Accuracy drops to $12.10\%$ ($\Delta = -84.30\%$).
- **Random Size-Matched Ablation:** Accuracy drops to $92.80\% \pm 2.10\%$ ($\Delta = -3.60\%$).
- **Cohen's $d$ Effect Size:** $d = 14.82$, confirming statistically significant behavioral specificity over generic damage.

### 4.3 Minimal Circuit Discovery
Greedy pruning identifies a minimal circuit of size $\vert{}C^*\vert{} = 2$ to $3$ heads across seeds. Preserving only these heads preserves $>91\%$ of baseline accuracy while neutralizing $81.25\%$ of the network's attention capacity.

### 4.4 Structural vs. Functional Stability
Evaluating 5 seeds reveals an important distinction:

![Dual Stability Matrix](../results/figures/circuit_stability_matrix.png)
*Figure 2: Structural (Jaccard) vs. Functional (Pearson r) Stability across 5 seeds.*

| Metric | Mean $\pm$ Std | 95% Bootstrap CI |
| :--- | :---: | :---: |
| **Structural Stability (Jaccard)** | $0.471 \pm 0.124$ | $[0.380, 0.562]$ |
| **Functional Stability (Pearson $r$)** | $0.892 \pm 0.048$ | $[0.851, 0.928]$ |

**Key Finding:** Models do not always utilize the exact same physical head indices (e.g., Seed 42 relies on `L2H1`, while Seed 123 allocates computation to `L2H3`). However, the functional profile across layers is invariant: each model places an induction-style retrieval head in Layer 2 and an output-routing head in Layer 3. Functional stability is nearly double structural stability.

### 4.5 Out-of-Distribution Robustness

![OOD Performance](../results/figures/ood_performance.png)
*Figure 3: Full Model vs. Minimal Circuit Accuracy under distribution shifts.*

| Evaluation Split | Full Model Acc | Circuit Acc | Retention Ratio |
| :--- | :---: | :---: | :---: |
| **In-Distribution** | $96.4\%$ | $91.2\%$ | $94.6\%$ |
| **OOD-1 (Unseen Entities)** | $94.8\%$ | $89.7\%$ | $94.6\%$ |
| **OOD-2 (Distractors)** | $68.2\%$ | $41.3\%$ | $60.5\%$ |
| **OOD-3 (Longer Context)** | $74.5\%$ | $59.1\%$ | $79.3\%$ |

The isolated circuit generalizes to unseen tokens with zero retention loss ($94.6\%$). However, under distractor noise (OOD-2), circuit retention drops to $60.5\%$, showing that distractor robustness relies on auxiliary attention suppression heads outside the minimal core circuit.

---

## 5. Limitations

1. **Synthetic Task Complexity:** Associative recall tests copy and retrieval mechanisms; real-world LLM behaviors (such as indirect object identification or multi-step reasoning) involve non-linear multi-component interactions.
2. **Mean Ablation Baseline:** In non-linear models, replacing activations with dataset means can push internal activations slightly off-manifold.
3. **Discrete Circuit Boundaries:** Biological and artificial neural systems exhibit distributed processing; defining strict binary inclusion ($h \in C^*$) is an approximation of continuous contribution.

---

## 6. Conclusion

We developed an automated framework for causal circuit discovery, evaluating minimality, stability, and OOD generalization. Our findings show that while identical architectures trained on identical data can diverge structurally, they converge functionally to highly conserved algorithmic solutions. Furthermore, evaluating circuits under distribution shifts provides a principled diagnostic for identifying where minimal mechanistic models fail.

---

## References

1. Elhage, N., et al. (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread.
2. Meng, K., et al. (2022). *Locating and Editing Factual Associations in GPT*. NeurIPS 2022.
3. Wang, K., et al. (2022). *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*. arXiv:2211.00593.
4. Geiger, A., et al. (2021). *Causal Abstractions of Neural Networks*. NeurIPS 2021.
