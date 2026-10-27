# Methodology & Formal Definitions

This document details the mathematical framework, causal intervention definitions, circuit discovery algorithms, and statistical validation metrics implemented in this repository.

---

## 1. Algorithmic Associative Recall Task

To establish an unambiguous ground truth, we formulate a synthetic associative recall task over discrete entity tokens $\mathcal{E} = \{E_0, E_1, \dots, E_{N-1}\}$:

$$\mathcal{X}_{\text{clean}} = [S_1 \text{ likes } O_1 \ , \ S_2 \text{ likes } O_2 \ ; \ \text{QUERY } S_1 \text{ likes}]$$

The model is trained via cross-entropy loss at the final sequence position to predict target token:
$$\mathcal{Y}^* = O_1$$

### Counterfactual Corrupted Prompt
To isolate causal components via interchange intervention, we construct paired corrupted prompts where only the target object is modified:
$$\mathcal{X}_{\text{corrupt}} = [S_1 \text{ likes } O_{\text{corrupt}} \ , \ S_2 \text{ likes } O_2 \ ; \ \text{QUERY } S_1 \text{ likes}] \implies \mathcal{Y}^*_{\text{corrupt}} = O_{\text{corrupt}}$$

---

## 2. Causal Effect via Activation Patching

For each attention head $(l, h)$ with layer $l \in \{0, \dots, L-1\}$ and head $h \in \{0, \dots, H-1\}$:
1. Run $\mathcal{X}_{\text{clean}}$ and cache the head activation $a_{l, h}^{\text{clean}}$.
2. Run $\mathcal{X}_{\text{corrupt}}$ while substituting head $(l, h)$'s output with $a_{l, h}^{\text{clean}}$.
3. Measure the change in clean target probability $P(\mathcal{Y}^*_{\text{clean}})$.

The **Normalized Causal Effect (CE)** is defined as:
$$\text{CE}(l, h) = \frac{P_{\text{patched}}(\mathcal{Y}^*_{\text{clean}}) - P_{\text{corrupt}}(\mathcal{Y}^*_{\text{clean}})}{P_{\text{clean}}(\mathcal{Y}^*_{\text{clean}}) - P_{\text{corrupt}}(\mathcal{Y}^*_{\text{clean}})}$$

- $\text{CE}(l, h) \approx 1.0$: Restoring head $(l, h)$ alone completely restores clean task performance.
- $\text{CE}(l, h) \approx 0.0$: Restoring head $(l, h)$ has negligible direct causal impact on target retrieval.

---

## 3. Minimal Circuit Search

We formulate circuit discovery as finding the minimal head subset $C^* \subseteq \mathcal{H}$ subject to a behavioral retention constraint:

$$C^* = \arg\min_{C \subseteq \mathcal{H}} |C| \quad \text{subject to} \quad \text{Sufficiency}(C) \ge \alpha \cdot \text{BaselineAccuracy}$$

where $\alpha \in (0, 1]$ (default $\alpha = 0.90$).

### Greedy Backward Pruning Algorithm
1. **Candidate Set Initialization:** Select initial candidate set $C_0 = \{(l, h) \mid \text{CE}(l, h) \ge \tau\}$. If no head exceeds $\tau$, fallback to top-$k$ heads ($k=4$).
2. **Sufficiency Check:** Verify that preserving $C_0$ while mean-ablating $\mathcal{H} \setminus C_0$ achieves accuracy $\ge \alpha \cdot \text{BaselineAccuracy}$.
3. **Sequential Elimination:** For each head $h \in C$:
   - Temporarily remove $h$: $C' = C \setminus \{h\}$.
   - Evaluate $\text{Sufficiency}(C')$.
   - If $\text{Sufficiency}(C') \ge \alpha \cdot \text{BaselineAccuracy}$, update $C \leftarrow C'$.
   - Otherwise, retain $h$.
4. Return minimal sufficient circuit $C^*$.

---

## 4. Causal Validation: Sufficiency vs. Necessity

A rigorous mechanistic claim requires establishing both sufficiency and necessity:

| Test | Operational Definition | Interventional State |
| :--- | :--- | :--- |
| **Sufficiency** | Does $C^*$ alone suffice to execute the behavior? | Keep $C^*$ active; clamp all $\mathcal{H} \setminus C^*$ to dataset mean activations $\mathbb{E}[a]$. |
| **Necessity** | Is $C^*$ indispensable for executing the behavior? | Clamp $C^*$ to dataset mean activations $\mathbb{E}[a]$; leave $\mathcal{H} \setminus C^*$ active. |
| **Random Control** | Is performance retention specific to $C^*$ or generic? | Evaluate 10 random subsets $C_{\text{rand}} \subset \mathcal{H}$ with $|C_{\text{rand}}| = |C^*|$. |

---

## 5. Cross-Seed Stability Formulations

We assess whether discovered circuits are architectural constants or seed-dependent implementations:

1. **Structural Stability (Pairwise Jaccard Index):**
   $$J(C_i, C_j) = \frac{|C_i \cap C_j|}{|C_i \cup C_j|}$$
   Measures exact component overlap across distinct training initializations.

2. **Functional Stability (Pearson Correlation of Causal Vectors):**
   $$r(\mathbf{v}_i, \mathbf{v}_j) = \frac{\sum_{k=1}^{16} (v_{i, k} - \bar{v}_i)(v_{j, k} - \bar{v}_j)}{\sqrt{\sum_{k=1}^{16} (v_{i, k} - \bar{v}_i)^2 \sum_{k=1}^{16} (v_{j, k} - \bar{v}_j)^2}}$$
   where $\mathbf{v}_i \in \mathbb{R}^{16}$ is the vector of head causal effects for seed $i$.

---

## 6. Distribution Shift Taxonomy (OOD)

To determine whether minimal circuits represent robust algorithms or in-distribution heuristics, we evaluate:

1. **OOD-1 (Unseen Entities):** Evaluates prompts composed of entity tokens held out during evaluation. Tests vocabulary independence of the retrieval mechanism.
2. **OOD-2 (Distractors):** Injects irrelevant key-value pairs (`D_S likes D_O <SEP>`) between the context and query. Tests resistance to attention distraction.
3. **OOD-3 (Context Scaling):** Scales context from 2 key-value pairs (seq length 11) to 3 key-value pairs (seq length 15). Tests length generalization.
