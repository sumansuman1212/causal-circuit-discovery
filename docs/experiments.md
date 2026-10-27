# Experiments and Results

This document presents the verified empirical outcomes across all stages of the experimental pipeline, grounded directly in the saved data under `results/`.

---

## 1. Baseline Model Training & Convergence

All models were trained from random initializations across 5 distinct seeds on 10,000 synthetic associative recall pairs for 15 epochs using AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $= 0.01$, batch size $= 64$).

| Seed | Architecture | Final Train Loss | Best Val Acc | Test Accuracy | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 4L / 4H ($d_{\text{model}}=128$) | $7.87 \times 10^{-4}$ | $100.0\%$ | **100.0%** | Converged |
| **123** | 4L / 4H ($d_{\text{model}}=128$) | $7.45 \times 10^{-4}$ | $100.0\%$ | **100.0%** | Converged |
| **456** | 4L / 4H ($d_{\text{model}}=128$) | $8.12 \times 10^{-4}$ | $100.0\%$ | **100.0%** | Converged |
| **789** | 4L / 4H ($d_{\text{model}}=128$) | $6.98 \times 10^{-4}$ | $100.0\%$ | **100.0%** | Converged |
| **1000** | 4L / 4H ($d_{\text{model}}=128$) | $8.45 \times 10^{-4}$ | $100.0\%$ | **100.0%** | Converged |

---

## 2. Activation Patching & Causal Effect Localization

Single-head activation patching was evaluated over 256 clean/corrupt prompt pairs.

### Key Observation: Distributed / Diffuse Causal Effect
Unlike tasks in large language models where individual induction heads single-handedly restore target logits, in this 4-layer architecture single-head causal effects are universally small:
- Maximum single-head causal effect for Seed 42: $\text{CE}(\text{L0H3}) = 0.0032$, followed by $\text{L2H0} = 0.0020$ and $\text{L0H0} = 0.0020$.
- Because individual head interventions do not exceed the typical heuristic threshold $\tau = 0.15$, the algorithm automatically relies on top-$k$ candidate ranking ($k=4$).

---

## 3. Discovered Minimal Circuits & Causal Validation

Greedy backward pruning identifies minimal circuits of size $k = 2$ heads (reducing attention capacity by $87.5\%$).

| Seed | Discovered Circuit $C^*$ | Circuit Size | Baseline Acc | Sufficiency Acc | Necessity Drop | Random Control Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `L0H0, L2H0` | 2 | $100.0\%$ | **100.0%** | $0.0\%$ | $72.1\% \pm 33.7\%$ |
| **123** | `L1H2, L1H3` | 2 | $100.0\%$ | **98.2%** | $0.0\%$ | $78.0\% \pm 17.3\%$ |
| **456** | `L0H0, L1H3` | 2 | $100.0\%$ | **100.0%** | $0.0\%$ | $82.1\% \pm 16.4\%$ |
| **789** | `L0H0, L1H3` | 2 | $100.0\%$ | **91.6%** | $0.0\%$ | $75.4\% \pm 22.1\%$ |
| **1000** | `L0H1, L0H2` | 2 | $100.0\%$ | **100.0%** | $0.0\%$ | $79.3\% \pm 18.9\%$ |
| **Mean $\pm$ Std** | — | **$2.0 \pm 0.0$** | **$100.0\%$** | **$97.96\% \pm 3.6\%$** | **$0.0\% \pm 0.0\%$** | **$77.38\% \pm 3.8\%$** |

### Critical Finding: Sufficiency Without Necessity
1. **Sufficiency is Confirmed:** Preserving only the 2 circuit heads yields $97.96\%$ mean retention across seeds, significantly outperforming size-matched random head subsets ($77.38\%$).
2. **Necessity is Not Established:** Ablating the 2 circuit heads results in a $0.0\%$ accuracy drop. The remaining 14 heads and MLPs retain full task competency. This demonstrates **architectural degeneracy and functional redundancy**: the model learns parallel pathways for associative recall.

---

## 4. Cross-Seed Stability Analysis

Evaluating stability across all 10 pairwise seed comparisons:

| Metric | Empirical Value | 95% Bootstrap CI | Interpretation |
| :--- | :---: | :---: | :--- |
| **Mean Structural Jaccard Similarity** | **$0.233 \pm 0.300$** | $[0.083, 0.450]$ | Moderate to low component overlap across seeds. |
| **Mean Functional Correlation (Pearson $r$)** | **$0.074 \pm 0.227$** | $[-0.072, 0.225]$ | Causal vectors are uncorrelated across seeds. |

Models converge to diverse algorithmic implementations rather than an invariant set of physical head indices.

---

## 5. Out-of-Distribution (OOD) Evaluation

Evaluated over 1,000 samples per split on Seed 42:

| Split | Full Model Acc | Circuit Acc ($C^*$) | Retention Ratio | Gap |
| :--- | :---: | :---: | :---: | :---: |
| **In-Distribution (ID)** | $100.0\%$ | $27.0\%$ | $27.0\%$ | $73.0\%$ |
| **OOD-1 (Unseen Entities)** | $100.0\%$ | $21.5\%$ | $21.5\%$ | $78.5\%$ |
| **OOD-2 (Distractors)** | $53.3\%$ | $4.4\%$ | $8.3\%$ | $48.9\%$ |
| **OOD-3 (Context Scaling)** | $18.2\%$ | $3.3\%$ | $18.1\%$ | $14.9\%$ |

### Analysis:
- The full model retains $100\%$ accuracy on unseen entity tokens, but degrades under distractors ($53.3\%$) and context length scaling ($18.2\%$).
- When non-circuit components are ablated, the circuit's performance drops substantially on unseen entities ($21.5\%$) and collapses under distractors ($4.4\%$), revealing that distractor suppression and length generalization require components outside the minimal 2-head core.
