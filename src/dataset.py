# src/dataset.py
import random
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

class AssociativeRecallDataset(Dataset):
    def __init__(self, data: List[Tuple[List[int], int]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_id = self.data[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

class SyntheticTaskGenerator:
    def __init__(self, num_entities: int = 30, seed: int = 42):
        random.seed(seed)
        self.special_tokens = {"<PAD>": 0, "<SEP>": 1, "<QUERY>": 2, "likes": 3}
        self.entity_tokens = [f"E_{i}" for i in range(num_entities)]

        # Build vocabulary
        self.vocab: Dict[str, int] = {**self.special_tokens}
        for token in self.entity_tokens:
            self.vocab[token] = len(self.vocab)
        self.id2token = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def generate_pair(self) -> Tuple[List[int], int, List[int], int]:
        """
        Creates a clean prompt and a counterfactual corrupted prompt.
        Format: "S1 likes O1 <SEP> S2 likes O2 <SEP> <QUERY> S1 likes" -> Target: O1
        """
        s1, s2 = random.sample(self.entity_tokens, 2)
        o1, o2, o_corrupt = random.sample([e for e in self.entity_tokens if e not in (s1, s2)], 3)

        clean_tokens = [s1, "likes", o1, "<SEP>", s2, "likes", o2, "<SEP>", "<QUERY>", s1, "likes"]
        corrupt_tokens = [s1, "likes", o_corrupt, "<SEP>", s2, "likes", o2, "<SEP>", "<QUERY>", s1, "likes"]

        clean_ids = [self.vocab[t] for t in clean_tokens]
        corrupt_ids = [self.vocab[t] for t in corrupt_tokens]

        return clean_ids, self.vocab[o1], corrupt_ids, self.vocab[o_corrupt]

    def build_dataset(self, num_samples: int) -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]]]:
        clean_data, corrupt_data = [], []
        for _ in range(num_samples):
            c_in, c_tgt, k_in, k_tgt = self.generate_pair()
            clean_data.append((c_in, c_tgt))
            corrupt_data.append((k_in, k_tgt))
        return clean_data, corrupt_data

def get_dataloaders(clean_data: List[Tuple[List[int], int]], batch_size: int = 64, split_ratios=(0.8, 0.1, 0.1)):
    n = len(clean_data)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train_ds = AssociativeRecallDataset(clean_data[:n_train])
    val_ds = AssociativeRecallDataset(clean_data[n_train:n_train + n_val])
    test_ds = AssociativeRecallDataset(clean_data[n_train + n_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
