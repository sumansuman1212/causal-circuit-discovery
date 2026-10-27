# Research Paper Draft & Empirical Audit Notes

This directory contains the working paper draft ([`paper.md`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/paper/paper.md)) and the compiled publication PDF ([`paper.pdf`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/paper/paper.pdf)) documenting the experimental investigation of automated Transformer circuit discovery.

### Compiling the Paper PDF
To recompile the markdown paper into PDF format at any time:
```bash
python scripts/compile_paper.py
```
This generates `paper/paper.html` with vector MathJax equations and prints to `paper/paper.pdf`.

---

## Status: Pre-Submission Working Draft (Empirically Aligned)

> [!NOTE]
> **Scientific Integrity Update (October 2026)**  
> The markdown draft ([`paper.md`](file:///c:/Users/saura/Coding/SSProjects/causal-circuit-discovery-main/paper/paper.md)) has been fully updated and aligned with the verified empirical data under `results/`.
>
> All tables, figures, causal effect values, cross-seed stability metrics ($J = 0.233 \pm 0.300$, $r = 0.074 \pm 0.227$), and OOD retention rates strictly report genuine experimental measurements. The paper's core thesis now formally investigates **architectural degeneracy and functional redundancy** in algorithmic Transformers.

---

## Empirical Reality vs. Initial Draft Claims

| Metric / Finding | Initial Draft Claim (`paper.md`) | Verified Experimental Ground Truth (`results/`) | Methodological Explanation |
| :--- | :--- | :--- | :--- |
| **Causal Effect Magnitude** | Implies high direct effect ($\approx 1.0$) localized in late layers | Single-head $\text{CE} \in [0.0001, 0.0032]$ | Single-head activation interchange fails to flip prediction alone; associative recall requires multi-head coordination. |
| **Discovered Circuit (Seed 42)** | Claimed: `L2H1`, `L3H0` | Actual: `L0H0`, `L2H0` (`circuits/seed42.json`) | Candidate selection fell back to top-4 heads; greedy pruning preserved `L0H0` and `L2H0`. |
| **Circuit Necessity Drop** | Claimed: $-84.3\%$ drop ($96.4\% \to 12.1\%$, Cohen's $d = 14.82$) | Actual: $0.0\%$ drop (Accuracy remains $100.0\%$) | **Functional Redundancy**: When the 2 circuit heads are ablated, the remaining 14 heads and MLPs fully retain task performance. The circuit is *sufficient*, but *not necessary*. |
| **Structural Stability (Jaccard)** | Claimed: $J = 0.471 \pm 0.124$ | Actual: $J = 0.233 \pm 0.300$ (`stability/cross_seed_stability.json`) | Random initializations find divergent head sets (`L0H0/L2H0`, `L1H2/L1H3`, `L0H1/L0H2`). |
| **Functional Stability (Pearson $r$)** | Claimed: $r = 0.892 \pm 0.048$ | Actual: $r = 0.074 \pm 0.227$ (`stability/cross_seed_stability.json`) | Single-head causal intervention vectors are effectively uncorrelated ($r \approx 0$) across independent seeds. |
| **OOD Retention (Unseen Tokens)** | Claimed: $94.6\%$ retention ($94.8\% \to 89.7\%$) | Actual: $21.5\%$ retention ($100.0\% \to 21.5\%$) (`tables/ood_evaluation.csv`) | The isolated 2-head circuit degrades significantly under entity shifts. |
| **OOD Retention (Distractors)** | Claimed: $60.5\%$ retention ($68.2\% \to 41.3\%$) | Actual: $8.3\%$ retention ($53.3\% \to 4.4\%$) | Both full model and circuit suffer severe contextual interference under distractors. |
| **OOD Retention (Long Context)** | Claimed: $79.3\%$ retention ($74.5\% \to 59.1\%$) | Actual: $18.1\%$ retention ($18.2\% \to 3.3\%$) | Sequence length scaling causes degradation in both model and isolated circuit. |

---

## Action Plan for Research Publication

To elevate this project to a peer-reviewed publication or top-tier AI safety fellowship submission:
1. **Transition from Node-Level to Edge-Level Attribution:** Adopt Edge Attribution Patching (EAP) or Automated Circuit Discovery (ACDC) to capture interactions between head pairs rather than isolated heads.
2. **Multi-Head Simultaneous Interventions:** Evaluate joint ablation rather than isolated single-head patching to account for distributed representation.
3. **Formal Redundancy Metrics:** Reframe the paper around *degeneracy and architectural redundancy in algorithmic transformers*—the fact that circuits are sufficient but not necessary is itself an interesting interpretability result when properly documented.
