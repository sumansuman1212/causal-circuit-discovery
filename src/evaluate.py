# src/evaluate.py
import torch
import torch.nn as nn

def evaluate_model(model: nn.Module, loader, device: str):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)[:, -1, :]  # Final token position
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += len(y)
    return correct / total, total_loss / total
