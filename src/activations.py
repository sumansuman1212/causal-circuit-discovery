# src/activations.py
from typing import Dict, Tuple
import torch
import torch.nn as nn
from src.model import TinyTransformer


class ActivationCache:
    """
    Manages extraction and storage of intermediate model activations
    without modifying model parameters.
    """
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}

    def clear(self):
        self.cache.clear()

    def get_caching_hook(self):
        def hook_fn(head_out: torch.Tensor, layer_idx: int) -> torch.Tensor:
            # head_out shape: (batch_size, n_heads, seq_len, d_head)
            self.cache[f"L{layer_idx}_head_out"] = head_out.detach().clone()
            return head_out
        return hook_fn

    def extract_activations(
        self, model: TinyTransformer, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Runs a forward pass and captures all attention head outputs.
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            activations: dictionary of detached activations
        """
        self.clear()
        hook_fn = self.get_caching_hook()
        with torch.no_grad():
            logits = model(input_ids, hook_fn=hook_fn)
        return logits, {k: v.clone() for k, v in self.cache.items()}
