# src/train.py
import json
import torch
import torch.nn as nn
from src.config import ModelConfig, TrainConfig
from src.model import TinyTransformer
from src.dataset import SyntheticTaskGenerator, get_dataloaders
from src.evaluate import evaluate_model

def train_baseline(train_cfg: TrainConfig, model_cfg: ModelConfig):
    torch.manual_seed(train_cfg.seed)

    # Generate data
    task_gen = SyntheticTaskGenerator(seed=train_cfg.seed)
    model_cfg.vocab_size = task_gen.vocab_size
    clean_data, _ = task_gen.build_dataset(num_samples=10000)
    train_loader, val_loader, test_loader = get_dataloaders(clean_data, batch_size=train_cfg.batch_size)

    # Initialize model
    model = TinyTransformer(model_cfg).to(train_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(train_cfg.epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(train_cfg.device), y.to(train_cfg.device)
            optimizer.zero_grad()
            logits = model(x)[:, -1, :]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += len(y)

        train_acc = correct / total
        val_acc, val_loss = evaluate_model(model, val_loader, train_cfg.device)
        print(f"Epoch {epoch+1:02d}/{train_cfg.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), train_cfg.checkpoint_dir / f"model_seed{train_cfg.seed}.pt")

    # Evaluate test split
    test_acc, test_loss = evaluate_model(model, test_loader, train_cfg.device)
    metrics = {
        "seed": train_cfg.seed,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "best_val_acc": best_val_acc
    }
    with open(train_cfg.results_dir / f"seed{train_cfg.seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Final Test Accuracy: {test_acc * 100:.2f}%. Checkpoint saved.")

if __name__ == "__main__":
    t_cfg = TrainConfig()
    m_cfg = ModelConfig()
    train_baseline(t_cfg, m_cfg)


