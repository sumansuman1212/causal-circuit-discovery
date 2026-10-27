# Methodological Limitations & Open Challenges

This document outlines the theoretical, algorithmic, and empirical limitations of the current implementation, candidly distinguishing established findings from areas requiring future research.

---

## 1. Node-Level vs. Edge-Level Causal Attribution

### The Limitation
The current implementation patches entire attention head outputs $a_{l, h} \in \mathbb{R}^{B \times T \times d_{\text{head}}}$ as atomic units. However, transformer computation is fundamentally mediated by directed communication edges between heads (e.g. from an induction head in Layer 1 to an S-inhibition or output head in Layer 3 via the residual stream).

### Consequence
Single-head activation interchange fails to isolate multi-component interactions. When an essential behavior requires the coordination of multiple heads, restoring any single head during a corrupted forward pass produces only marginal causal effects ($\text{CE} < 0.005$).

### Recommended Extension
Incorporate **Edge Attribution Patching (EAP)** or **Automated Circuit Discovery (ACDC)**, which formulate circuits as computational computation graphs $(V, E)$ over edges rather than monolithic node sets.

---

## 2. Redundancy and the "Sufficiency Without Necessity" Paradox

### The Empirical Reality
Our greedy backward search successfully isolates small subnetworks ($|C^*| = 2$) that achieve $>97\%$ sufficiency accuracy. However, mean-ablating $C^*$ results in a $0.0\%$ drop in task accuracy—the full model retains $100\%$ accuracy without these heads.

### Theoretical Implication
Neural network representations frequently exhibit **degeneracy**—multiple parallel, functionally redundant subcircuits trained under overparameterization. While identifying a sufficient subnetwork is useful for mechanistic analysis, claiming that the subnetwork is *the* circuit for the behavior is false.

---

## 3. Off-Manifold Activation Risks with Mean Ablation

### The Limitation
To ablate heads, `ComponentAblator` substitutes activations with the dataset-wide mean activation $\mathbb{E}[a]$. In non-linear deep networks, replacing activations with mean vectors can push representations off the natural data manifold, potentially introducing artificial failure modes or obscuring true causal dependencies.

### Alternative Baselines
Future iterations should compare mean ablation with:
- Zero ablation (baseline control).
- Resample / counterfactual ablation (sampling activations from paired counterfactual runs).
- Continuous attribution masking (e.g., differentiable binary gates).

---

## 4. Synthetic Task Scope

### The Task
The synthetic associative recall task ($S_1 \text{ likes } O_1 \dots$) tests a simplified copy-and-retrieve algorithmic primitive.

### Differences from Production LLMs
- Vocabulary is restricted to 34 tokens (toy entity IDs and delimiter tokens).
- Synthetic tasks lack the rich polysemanticity, superposition, and complex syntactic dependencies present in natural language models (e.g. GPT-2, LLaMA).
- Real-world interpretability benchmarks (such as Indirect Object Identification or Greater-Than) involve multi-stage circuits spanning induction, duplicate token suppression, and negative heads.

---

## 5. Cross-Seed Non-Identifiability

Our empirical analysis demonstrates that identical architectures trained on identical data distributions converge to structurally and functionally distinct circuits:
- Structural overlap is low ($J = 0.233 \pm 0.300$).
- Causal correlation across heads is near zero ($r = 0.074 \pm 0.227$).

This proves that **circuit discovery algorithms cannot assume a unique ground-truth circuit across model instances**. Interpretability tools must account for initialization stochasticity when making claims about canonical circuits.
