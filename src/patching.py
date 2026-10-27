# src/patching.py
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import torch
import torch.nn.functional as F

from src.activations import ActivationCache
from src.config import ModelConfig, TrainConfig
from src.dataset import SyntheticTaskGenerator
from src.model import TinyTransformer


class ActivationPatcher:
    def __init__(self, model: TinyTransformer, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cache_collector = ActivationCache()

    def run_clean_and_corrupt_baselines(
        self, clean_tokens: torch.Tensor, corrupt_tokens: torch.Tensor, target_ids: torch.Tensor
    ) -> Tuple[float, float, Dict[str, torch.Tensor]]:
        """
        Calculates baseline clean and corrupt probabilities on clean target tokens,
        and saves the clean activations for all heads.
        """
        clean_tokens = clean_tokens.to(self.device)
        corrupt_tokens = corrupt_tokens.to(self.device)
        target_ids = target_ids.to(self.device)

        # 1. Clean run: capture logits and cache
        clean_logits, clean_cache = self.cache_collector.extract_activations(self.model, clean_tokens)
        clean_probs = F.softmax(clean_logits[:, -1, :], dim=-1)
        p_clean = clean_probs.gather(1, target_ids.unsqueeze(1)).mean().item()

        # 2. Corrupt run: unpatched baseline
        with torch.no_grad():
            corrupt_logits = self.model(corrupt_tokens)
        corrupt_probs = F.softmax(corrupt_logits[:, -1, :], dim=-1)
        p_corrupt = corrupt_probs.gather(1, target_ids.unsqueeze(1)).mean().item()

        return p_clean, p_corrupt, clean_cache

    def patch_single_head(
        self,
        corrupt_tokens: torch.Tensor,
        target_ids: torch.Tensor,
        target_layer: int,
        target_head: int,
        clean_head_activation: torch.Tensor,
    ) -> float:
        """
        Runs corrupted input through the model while replacing target_head's activation
        in target_layer with the clean activation.
        """
        corrupt_tokens = corrupt_tokens.to(self.device)
        target_ids = target_ids.to(self.device)

        def patch_hook(head_out: torch.Tensor, layer_idx: int) -> torch.Tensor:
            if layer_idx == target_layer:
                # head_out: (batch_size, n_heads, seq_len, d_head)
                patched = head_out.clone()
                patched[:, target_head, :, :] = clean_head_activation[:, target_head, :, :]
                return patched
            return head_out

        with torch.no_grad():
            patched_logits = self.model(corrupt_tokens, hook_fn=patch_hook)

        patched_probs = F.softmax(patched_logits[:, -1, :], dim=-1)
        p_patched = patched_probs.gather(1, target_ids.unsqueeze(1)).mean().item()
        return p_patched

    def patch_all_heads(
        self,
        clean_tokens: torch.Tensor,
        corrupt_tokens: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> pd.DataFrame:
        """
        Systematically patches each head (L0H0 through L3H3) and calculates Causal Effect (CE).
        """
        p_clean, p_corrupt, clean_cache = self.run_clean_and_corrupt_baselines(
            clean_tokens, corrupt_tokens, target_ids
        )
        norm_factor = p_clean - p_corrupt

        results = []
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads

        print(f"Baseline P(Clean Target): Clean = {p_clean:.4f}, Corrupt = {p_corrupt:.4f}, Delta = {norm_factor:.4f}")

        for layer in range(n_layers):
            clean_layer_acts = clean_cache[f"L{layer}_head_out"].to(self.device)
            for head in range(n_heads):
                p_patched = self.patch_single_head(
                    corrupt_tokens=corrupt_tokens,
                    target_ids=target_ids,
                    target_layer=layer,
                    target_head=head,
                    clean_head_activation=clean_layer_acts,
                )

                # Avoid division by zero
                causal_effect = (p_patched - p_corrupt) / (norm_factor + 1e-12)

                results.append({
                    "component": f"L{layer}H{head}",
                    "layer": layer,
                    "head": head,
                    "clean_probability": p_clean,
                    "corrupted_probability": p_corrupt,
                    "patched_probability": p_patched,
                    "causal_effect": causal_effect,
                })

        df = pd.DataFrame(results)
        return df


def run_activation_patching_experiment(
    checkpoint_path: Path = Path("models/checkpoints/model_seed42.pt"),
    num_eval_samples: int = 256,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Path = Path("results/patching"),
) -> pd.DataFrame:
    """
    Driver for activation patching: loads checkpoint, builds paired data, evaluates, and writes CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Vocabulary
    task_gen = SyntheticTaskGenerator(seed=seed)
    model_cfg = ModelConfig(vocab_size=task_gen.vocab_size)
    clean_data, corrupt_data = task_gen.build_dataset(num_samples=num_eval_samples)

    clean_ids = torch.tensor([x[0] for x in clean_data], dtype=torch.long)
    target_ids = torch.tensor([x[1] for x in clean_data], dtype=torch.long)
    corrupt_ids = torch.tensor([x[0] for x in corrupt_data], dtype=torch.long)

    # 2. Load Trained Model
    model = TinyTransformer(model_cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")

    # 3. Patching Execution
    patcher = ActivationPatcher(model, device=device)
    df_results = patcher.patch_all_heads(clean_ids, corrupt_ids, target_ids)

    # 4. Save results
    output_file = output_dir / f"seed{seed}.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n[DONE] Saved patching results to {output_file}")

    # Print ranked causal heads
    ranked = df_results.sort_values(by="causal_effect", ascending=False)[
        ["component", "patched_probability", "causal_effect"]
    ]
    print("\n--- Top Ranked Components ---")
    print(ranked.head(6).to_string(index=False))
    return df_results


if __name__ == "__main__":
    t_cfg = TrainConfig()
    ckpt = t_cfg.checkpoint_dir / f"model_seed{t_cfg.seed}.pt"
    run_activation_patching_experiment(checkpoint_path=ckpt, seed=t_cfg.seed)
