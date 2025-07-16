import numpy as np
import torch

def compute_sqrt_inverse_weights(labels, num_classes, device):
    counts = np.bincount(labels.flatten())
    weights = 1 / np.sqrt(counts + 1e-5)
    weights /= (weights.sum() * num_classes)
    return torch.tensor(weights, dtype=torch.float32, device=device) # Computing logic: 1 / sqrt(count)

def run_inference(model, loader, device):
    model.eval()
    all_preds, all_scores, all_labels = [], [], []

    with torch.no_grad():
        for x_list, edge_idx_list, edge_idx_out_list, y_batch in loader:
            x_list = [x.to(device, non_blocking=True) for x in x_list]
            edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
            edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
            y_batch = y_batch.to(device, non_blocking=True)

            # Assuming forward pass returns 'scores'
            scores = model(x_list, edge_idx_list, edge_idx_out_list)
            all_preds.append(scores.argmax(dim=1).cpu())
            all_scores.append(scores.cpu())
            all_labels.append(y_batch.cpu())

    return {
        'preds': torch.cat(all_preds).numpy(),
        'scores': torch.cat(all_scores).numpy(),
        'labels': torch.cat(all_labels).numpy()
    }