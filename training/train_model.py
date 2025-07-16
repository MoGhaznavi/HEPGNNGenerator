import numpy as np
import torch
from torch.amp import autocast, GradScaler
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    all_preds, all_scores, all_labels = [], [], []
    scaler = GradScaler('cuda')

    for x_list, edge_idx_list, edge_idx_out_list, y_batch in loader:
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
        edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type = 'cuda'):
            scores = model(x_list, edge_idx_list, edge_idx_out_list)
            loss = criterion(scores, y_batch.squeeze(1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y_batch)
        preds = scores.argmax(dim=1)
        correct += (preds == y_batch.squeeze(1)).sum().item()
        total += len(y_batch)

        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(scores.detach().cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    return {
        'loss': total_loss / total,
        'acc': correct / total,
        'preds': np.array(all_preds),
        'scores': np.array(all_scores),
        'labels': np.array(all_labels)
    }