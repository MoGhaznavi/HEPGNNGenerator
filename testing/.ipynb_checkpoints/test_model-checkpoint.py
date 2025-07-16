import numpy as np
import torch
from torch.amp import autocast, GradScaler
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def test_model(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_scores, all_labels = [], [], []
    with torch.no_grad():
        for x_list, edge_idx_list, edge_idx_out_list, y_batch in loader:
            x_list = [x.to(device, non_blocking=True) for x in x_list]
            edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
            edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
            y_batch = y_batch.to(device, non_blocking=True)
            with autocast(device_type = 'cuda'):
                scores = model(x_list, edge_idx_list, edge_idx_out_list)
                loss = criterion(scores, y_batch.squeeze(1))
            total_loss += loss.item() * len(y_batch)
            preds = scores.argmax(dim=1)
            correct += (preds == y_batch.squeeze(1)).sum().item()
            total += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return {
        'loss': total_loss / total,
        'acc': correct / total,
        'preds': np.array(all_preds),
        'scores': np.array(all_scores),
        'labels': np.array(all_labels)
    }