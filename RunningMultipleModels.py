# Standard Library
import argparse
import lzma
import os
import pickle
import re
import signal
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, IterableDataset

# PyTorch Geometric
from torch_geometric.nn import GCNConv
from torch.nn import BatchNorm1d

# --- Global setup -------------------------------------------------------------

# Set multiprocessing start method to 'spawn' (required for CUDA on some systems)
mp.set_start_method('spawn', force=True)

# Enable cuDNN auto-tuning for faster GPU operations (may reduce determinism)
torch.backends.cudnn.benchmark = True

# Limit the number of CPU threads each process can use
torch.set_num_threads(2)


# --- Utility functions --------------------------------------------------------

def load_pickle(filename: str) -> Any:
    """Load an object from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_shared_data(load_path: str) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Load all data files that will be shared across GPU processes.

    Returns:
        scaled_data:    Dict of pre-scaled datasets
        unscaled_data:  Dict of original datasets
        neighbor_pairs_list:    Array of neighbor index pairs
        labels_for_neighbor_pairs: Array of labels matching neighbor pairs
    """
    print("Loading shared data...")

    # Read multiple pre-saved pickle files
    scaled_data = load_pickle(os.path.join(load_path, "scaled_data_1000events.pkl"))
    unscaled_data = load_pickle(os.path.join(load_path, "data_1000events.pkl"))
    neighbor_pairs_list = load_pickle(os.path.join(load_path, "neighbor_pairs_list.pkl"))
    labels_for_neighbor_pairs = load_pickle(os.path.join(load_path, "labels_for_neighbor_pairs_1000events.pkl"))

    # Display the shapes of loaded arrays for verification
    print("Loaded data shapes:")
    print(f"  scaled_data['data_0']: {scaled_data['data_0'].shape}")
    print(f"  unscaled_data['data_0']: {unscaled_data['data_0'].shape}")
    print(f"  neighbor_pairs_list: {neighbor_pairs_list.shape}")
    print(f"  labels_for_neighbor_pairs: {labels_for_neighbor_pairs.shape}")

    return scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs


def compute_weight_tensor(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Calculate per-class weights to handle class imbalance.

    Args:
        labels:       Array of class labels
        num_classes:  Total number of classes
        device:       Torch device (CPU/GPU) to place the tensor on
    """
    counts = np.bincount(labels.flatten())          # Count samples per class
    weights = 1 / np.sqrt(counts + 1e-5)            # Inverse-sqrt weighting
    weights /= (weights.sum() * num_classes)        # Normalize weights
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_data_pickle(filename: str, directory: str, data: Any) -> None:
    """
    Save a Python object as a pickle file.

    Creates the directory if it does not exist.
    """
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def find_latest_checkpoint(save_dir: str, base_name: str) -> Optional[Tuple[int, str]]:
    """
    Find the newest model checkpoint in a directory.

    Args:
        save_dir:  Directory containing checkpoint files
        base_name: Base filename pattern (e.g., 'model')

    Returns:
        Tuple of (latest_epoch_number, checkpoint_path) or None if not found.
    """
    pattern = re.compile(f"{re.escape(os.path.splitext(base_name)[0])}_epoch(\\d+).pt")
    checkpoints = []

    # Search directory for files matching the pattern and extract epoch numbers
    for fname in os.listdir(save_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, os.path.join(save_dir, fname)))

    # Return the checkpoint with the highest epoch number, or None if empty
    return max(checkpoints, key=lambda x: x[0]) if checkpoints else None

# Dataset: Generates balanced graph-edge batches for multi-class classification
class MultiClassBatchGenerator(IterableDataset):
    """
    Iterable dataset that:
      • Precomputes balanced neighbor-pair samples for each event
      • Supports optional padding, unscaled features, and debug logging
    """

    def __init__(self,
        features_dict: Dict,
        neighbor_pairs: np.ndarray,
        labels: np.ndarray,
        class_counts: Dict,
        mode: str = "train",
        is_bi_directional: bool = True,
        batch_size: int = 1,
        train_ratio: float = 0.7,
        padding: bool = False,
        with_labels: bool = False,
        padding_class: int = 0,
        debug: bool = False,
        unscaled_data_dict: Optional[Dict] = None,
    ):
        # Configuration flags
        self.debug = debug
        self.is_bi_directional = is_bi_directional
        self.batch_size = batch_size
        self.padding = padding
        self.with_labels = with_labels
        self.padding_class = padding_class

        # Store scaled features as float32 tensors
        self.features_dict = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in features_dict.items()
        }

        # Optionally store unscaled features for analysis
        self.unscaled_features_dict = (
            self._precompute_unscaled_features(unscaled_data_dict)
            if unscaled_data_dict is not None else None
        )

        # Neighbor pairs and class labels
        self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.class_counts = class_counts

        # Split events into train/test sets
        num_events = len(features_dict)
        train_size = int(num_events * train_ratio)
        self.event_indices = (
            list(range(train_size)) if mode == "train" else list(range(train_size, num_events))
        )

        # Precompute all per-event samples for fast iteration
        self.precomputed_samples = self._precompute_all_samples()

        if self.debug:
            print(f"Initialized with {len(self.precomputed_samples)} samples")
            print(f"Mode: {mode}, Events: {len(self.event_indices)}")

    # ----- Helpers -------------------------------------------------------------

    def _precompute_unscaled_features(self, unscaled_data_dict: Dict) -> Dict:
        """
        Convert raw detector data to [SNR, η, φ] format for logging/analysis.
        Handles input tensors of shape (N,4) or (N,3).
        """
        unscaled_features = {}
        for k, v in unscaled_data_dict.items():
            t = torch.as_tensor(v, dtype=torch.float32)
            if t.shape[1] == 4:  # convert (E,px,py,pz) to (E,η,φ)
                eta = t[:, 1]
                phi = torch.atan2(t[:, 3], t[:, 2])
                t = torch.cat([t[:, 0:1], eta[:, None], phi[:, None]], dim=1)
            elif t.shape[1] != 3:
                raise ValueError(f"Unexpected feature dimension {t.shape[1]} for key {k}")
            unscaled_features[k] = t
        return unscaled_features

    def _build_class_indices(self, event_id: int) -> Dict[int, torch.Tensor]:
        """Return indices of neighbor pairs for each class within one event."""
        return {
            cls: torch.where(self.labels[event_id] == cls)[0]
            for cls in self.class_counts.keys()
        }

    def _sample_edges(self, event_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Balanced sampling of neighbor pairs for a single event.
        Returns:
            sampled indices,
            mask of true (vs. padded) samples,
            output labels (optionally padded)
        """
        class_indices = self._build_class_indices(event_id)
        sampled_indices, padded_mask = [], []

        # Randomly sample up to class_counts per class
        for cls, count in self.class_counts.items():
            idx = class_indices.get(cls, torch.tensor([], dtype=torch.long))
            if len(idx) > 0:
                selected = idx[torch.randperm(len(idx))[: min(count, len(idx))]]
                sampled_indices.append(selected)
                padded_mask.extend([True] * len(selected))

        sampled = torch.cat(sampled_indices) if sampled_indices else torch.tensor([], dtype=torch.long)

        # Optional padding with a specified class
        if self.padding and len(sampled) < sum(self.class_counts.values()):
            pad_size = sum(self.class_counts.values()) - len(sampled)
            pad_idx = class_indices.get(self.padding_class, torch.tensor([], dtype=torch.long))
            if len(pad_idx) > 0:
                sampled = torch.cat([sampled, pad_idx[:pad_size]])
                padded_mask.extend([False] * min(pad_size, len(pad_idx)))

        padded_mask = torch.tensor(padded_mask, dtype=torch.bool)

        # Adjust output labels (mark padded nodes if not using real labels)
        true_labels = self.labels[event_id][sampled]
        out_labels = true_labels if self.with_labels else true_labels.clone()
        if not self.with_labels:
            out_labels[~padded_mask] = 4  # special label for padded samples

        return sampled, padded_mask, out_labels

    def _precompute_all_samples(self) -> List[Tuple]:
        """Create and cache all event samples for quick DataLoader iteration."""
        samples: List[Tuple] = []
        for event_id in self.event_indices:
            edge_idx, mask, out_labels = self._sample_edges(event_id)
            if len(edge_idx) == 0:
                continue
            pairs = self.neighbor_pairs[edge_idx].T
            x_scaled = self.features_dict[f"data_{event_id}"]
            x_unscaled = (
                self.unscaled_features_dict.get(f"data_{event_id}")
                if self.unscaled_features_dict else None
            )
            samples.append((x_scaled, pairs, pairs.clone(), out_labels.unsqueeze(1), x_unscaled))
        return samples

    # ----- IterableDataset interface ------------------------------------------

    def __iter__(self):
        """Yield precomputed samples one by one."""
        for s in self.precomputed_samples:
            yield s

    def __len__(self):
        """Number of samples available."""
        return len(self.precomputed_samples)

    @staticmethod
    def collate_data(batch: List[Tuple]) -> Tuple:
        """
        Combine a list of samples into batch format for DataLoader.
        Returns:
            (node_features_list, edge_index_list,
             original_edge_index_list, label_tensor, unscaled_list)
        """
        x_list = [b[0] for b in batch]
        edge_index_list = [b[1] for b in batch]
        edge_index_out_list = [b[2] for b in batch]
        y_batch = torch.cat([b[3] for b in batch], dim=0)
        unscaled_list = None if batch[0][4] is None else [b[4] for b in batch]
        return x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list


# Model: Graph Neural Network for edge classification
class MultiEdgeClassifier(nn.Module):
    """
    Graph Convolutional Network (GCN) that classifies edges between nodes.
    Supports optional layer weighting, softmax scaling, and debug timing.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 device: torch.device,
                 num_layers: int = 6,
                 layer_weights: bool = False,
                 softmax: bool = False,
                 debug: bool = False):
        super().__init__()
        self.device = device
        self.debug = debug
        self.layer_weights_enabled = layer_weights
        self.softmax = softmax
        self.num_layers = num_layers

        # Initial node embedding layer
        self.node_embedding = nn.Linear(input_dim, hidden_dim)

        # Stack of GCN + BatchNorm layers
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Final edge classification layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

        # Optional learnable layer weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers)) if layer_weights else None

    def forward(self,
                x_list: List[torch.Tensor],
                edge_index_list: List[torch.Tensor],
                edge_index_out_list: List[torch.Tensor],
                y_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass:
          • Embed nodes
          • Apply multiple GCN layers with residual connections
          • Concatenate representations of edge endpoints
          • Predict edge classes
        """
        if self.debug:
            total_start = time.perf_counter()
            timings = {"weight_prep": 0.0, "move_to_device": 0.0,
                       "node_embedding": 0.0,
                       "layers": [0.0] * self.num_layers,
                       "edge_repr": 0.0, "final_fc": 0.0}

        all_edge_reprs = []

        # Optional layer-weight normalization
        if self.layer_weights_enabled:
            weights = (torch.softmax(self.layer_weights, dim=0)
                       if self.softmax else self.layer_weights)
        else:
            weights = None

        # Process each graph in the batch
        for x, proc_edges, orig_edges in zip(x_list, edge_index_list, edge_index_out_list):
            x = x.to(self.device, non_blocking=True)
            proc_edges = proc_edges.to(self.device, non_blocking=True)

            # Initial embedding
            x_embed = self.node_embedding(x)

            # Apply GCN layers with residual connections
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                h = torch.relu(bn(conv(x_embed, proc_edges)))
                if weights is not None:
                    h = weights[i] * h
                x_embed = x_embed + h

            # Build edge-level representations
            src, dst = orig_edges[0], orig_edges[1]
            edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
            all_edge_reprs.append(edge_repr)

        # Final classification across all edges in the batch
        out = self.fc(torch.cat(all_edge_reprs, dim=0))
        return out

def train_model(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: nn.Module, scaler: GradScaler, device: torch.device,
                debug: bool = False) -> Dict[str, float]:
    """
    Train the model for one epoch.
    Uses mixed-precision and gradient scaling for speed and stability.
    Returns a dict with average loss and accuracy.
    """
    model.train()
    total_loss = correct = total = 0
    start_train = time.perf_counter()

    for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list) in enumerate(loader):
        if debug:
            start_batch = time.perf_counter()

        # Move all tensors in the batch to GPU/CPU device
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
        edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
        y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

        optimizer.zero_grad(set_to_none=True)  # Faster gradient reset

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type="cuda"):
            scores = model(x_list, edge_idx_list, edge_idx_out_list)
            loss = criterion(scores, y_batch)

        # Backpropagation with gradient scaling to avoid underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track loss and accuracy
        total_loss += loss.item() * len(y_batch)
        preds = scores.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

        # Optional debug info every 10 batches
        if debug and batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}: loss={loss.item():.4f}, "
                  f"time={time.perf_counter() - start_batch:.3f}s")

    if debug:
        print(f"Training epoch completed in {time.perf_counter() - start_train:.2f}s")

    return {
        "loss": total_loss / total if total else 0,
        "acc": correct / total if total else 0,
    }


def test_model(model: nn.Module, loader: DataLoader, criterion: nn.Module,
               device: torch.device, debug: bool = False) -> Dict[str, float]:
    """
    Evaluate the model on a validation/test set.
    No gradient computation, returns average loss and accuracy.
    """
    model.eval()
    total_loss = correct = total = 0
    start_test = time.perf_counter()

    with torch.no_grad():
        for x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list in loader:
            # Move tensors to device
            x_list = [x.to(device, non_blocking=True) for x in x_list]
            edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
            edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
            y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda"):
                scores = model(x_list, edge_idx_list, edge_idx_out_list)
                loss = criterion(scores, y_batch)

            # Track metrics
            total_loss += loss.item() * len(y_batch)
            preds = scores.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    if debug:
        print(f"Testing completed in {time.perf_counter() - start_test:.2f}s")

    return {
        "loss": total_loss / total if total else 0,
        "acc": correct / total if total else 0,
    }


@torch.no_grad()
def run_inference_with_generator(model, generator, device, debug=False):
    """
    Run inference event-by-event using a data generator.
    Collects predictions, softmax scores, neighbor pairs,
    and both scaled and unscaled node features for each event.
    """
    model.eval()
    all_results = []

    for i, (x_scaled, edge_index, edge_index_out, y, x_unscaled) in enumerate(generator):
        if debug and i % 10 == 0:
            print(f"Processing event {i}")

        # Move inputs to device
        x_scaled = x_scaled.to(device)
        edge_index = edge_index.to(device)
        edge_index_out = edge_index_out.to(device)

        # Forward pass and predictions
        out = model([x_scaled], [edge_index], [edge_index_out])
        preds = out.argmax(dim=1).cpu().numpy()
        scores = torch.softmax(out, dim=1).cpu().numpy()

        # Extract endpoints of each predicted edge
        src_nodes = edge_index_out[0].cpu()
        dst_nodes = edge_index_out[1].cpu()

        # Gather scaled and unscaled node features for those endpoints
        feats_i_scaled = x_scaled[src_nodes].cpu().numpy()
        feats_j_scaled = x_scaled[dst_nodes].cpu().numpy()
        feats_i_unscaled = x_unscaled[src_nodes].numpy()
        feats_j_unscaled = x_unscaled[dst_nodes].numpy()

        # Grab η (pseudorapidity) from unscaled features
        eta_i = feats_i_unscaled[:, 1]
        eta_j = feats_j_unscaled[:, 1]

        # Store all inference results for this event
        all_results.append({
            "event_id": i,
            "preds": preds,
            "scores": scores,
            "labels": y.squeeze().numpy() if y is not None else None,
            "neighbor_pairs": edge_index_out.cpu().numpy().T,
            "features_i_scaled": feats_i_scaled,
            "features_j_scaled": feats_j_scaled,
            "features_i_unscaled": feats_i_unscaled,
            "features_j_unscaled": feats_j_unscaled,
            "eta_i": eta_i,
            "eta_j": eta_j,
        })

        # For debugging, limit to first 5 events
        if debug and i >= 4:
            break

    return all_results

def run_model(model: nn.Module, batch_size: int, save_dir: str, best_model_name: str,
              train_generator_class: type, test_generator_class: type,
              train_generator_kwargs: Dict, test_generator_kwargs: Dict,
              epochs: int, device: torch.device, optimizer: optim.Optimizer,
              criterion: nn.Module, unscaled_data_dict: Optional[Dict] = None,
              lr: float = 1e-3, resume: bool = True, patience: int = 10,
              delta: float = 0.0001, debug: bool = False) -> Tuple[Dict, nn.Module, str]:
    """
    Main training loop:
      • Trains for the given number of epochs
      • Saves checkpoints and the best model
      • Supports resuming, early stopping, and final inference
    """

    # --- Setup paths and initial metrics ---
    if not debug:
        os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, best_model_name)
    best_model_path = os.path.join(save_dir, f"best_{best_model_name}")
    metrics_path = os.path.splitext(model_path)[0] + ".pkl"

    start_epoch = 1
    best_test_acc, best_test_loss = 0.0, float("inf")
    best_epoch, total_time_trained = 0, 0.0
    scaler = torch.amp.GradScaler(device="cuda")

    # --- Resume from latest checkpoint if enabled ---
    if resume:
        checkpoint_result = find_latest_checkpoint(save_dir, best_model_name)
        if checkpoint_result:
            resumed_epoch, latest_checkpoint_path = checkpoint_result
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = resumed_epoch + 1
            if not debug:
                print(f"Resuming from checkpoint: {latest_checkpoint_path} (epoch {resumed_epoch})")
            # Load previous metrics if available
            metrics = {}
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'rb') as f:
                        metrics = pickle.load(f)
                    best_test_acc = float(np.max(metrics.get('test_acc', [0.0])))
                    best_epoch = int(np.argmax(metrics.get('test_acc', [0.0]))) + 1
                    total_time_trained = metrics.get('total_time', 0.0)
                except:
                    metrics = {}
        else:
            if not debug:
                print("No previous checkpoint found. Starting from scratch.")
            metrics = {}
    else:
        metrics = {}

    # Initialize metrics containers
    metrics.update({
        'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [],
        'epoch_times': [], 'best_epoch': best_epoch,
        'best_test_acc': best_test_acc, 'best_test_loss': best_test_loss,
        'total_time': total_time_trained, 'time_per_epoch': 0.0
    })

    # --- Build data loaders once ---
    train_loader = DataLoader(
        train_generator_class(**train_generator_kwargs),
        batch_size=batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_generator_class(**test_generator_kwargs),
        batch_size=batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data,
        pin_memory=True
    )

    early_stopping_counter = 0

    # --- Epoch training loop ---
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()

        # Train on one epoch and then evaluate
        train_results = train_model(model, train_loader, optimizer, criterion, scaler, device, debug=debug)
        test_results = test_model(model, test_loader, criterion, device, debug=debug)

        # Record metrics
        epoch_time = time.perf_counter() - epoch_start_time
        metrics['epoch_times'].append(epoch_time)
        metrics['train_loss'].append(train_results['loss'])
        metrics['test_loss'].append(test_results['loss'])
        metrics['train_acc'].append(train_results['acc'])
        metrics['test_acc'].append(test_results['acc'])

        # --- Check for improvement and early stopping ---
        current_test_loss = test_results['loss']
        if current_test_loss < best_test_loss:
            # New best model: save and reset early-stopping counter
            best_test_loss = current_test_loss
            best_test_acc = test_results['acc']
            best_epoch = epoch
            metrics['best_epoch'] = best_epoch
            metrics['best_test_acc'] = best_test_acc
            metrics['best_test_loss'] = best_test_loss
            early_stopping_counter = 0
            if not debug:
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_test_acc': best_test_acc,
                    'best_test_loss': best_test_loss
                }, best_model_path)
        else:
            # Increment counter if loss hasn’t improved by `delta`
            if current_test_loss > best_test_loss - delta:
                early_stopping_counter += 1

        # Save a full checkpoint every epoch
        if not debug:
            checkpoint_path = os.path.join(save_dir, f"{os.path.splitext(best_model_name)[0]}_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, checkpoint_path)

        # Optional progress printout
        if debug or epoch % 5 == 0:
            print(f"GPU {device.index if hasattr(device, 'index') else 'CPU'} | "
                  f"Time {epoch_time:.1f}s | Best Test Acc: {best_test_acc:.4f}\n"
                  f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['acc']:.4f} | "
                  f"Test Loss: {test_results['loss']:.4f} | Test Acc: {test_results['acc']:.4f}")

        # Stop if patience exceeded
        if early_stopping_counter >= patience:
            if not debug:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
            break

    # --- Final metrics and inference with best model ---
    metrics['total_time'] = total_time_trained + sum(metrics['epoch_times'])
    metrics['time_per_epoch'] = np.mean(metrics['epoch_times']) if metrics['epoch_times'] else 0

    if not debug:
        # Load best weights for final inference
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_generator = test_generator_class(**test_generator_kwargs)
        final_results_per_event = run_inference_with_generator(model, test_generator, device, debug=debug)

        # Store per-event and concatenated results
        final_metrics = {
            **metrics,
            'final_results_per_event': final_results_per_event,
            'num_events_evaluated': len(final_results_per_event)
        }
        all_preds = np.concatenate([ev['preds'] for ev in final_results_per_event])
        all_scores = np.concatenate([ev['scores'] for ev in final_results_per_event])
        all_labels = (np.concatenate([ev['labels'] for ev in final_results_per_event])
                      if final_results_per_event[0]['labels'] is not None else None)
        final_metrics.update({
            'final_preds_concatenated': all_preds,
            'final_scores_concatenated': all_scores,
            'final_labels_concatenated': all_labels,
        })

        # Save final metrics to disk
        save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)
        total_min, total_sec = divmod(metrics['total_time'], 60)
        print(f"\nTraining complete in {int(total_min)}m {total_sec:.1f}s")
        print(f"Best model at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")

    return final_metrics, model, best_model_path

def train_single_gpu(gpu_id: int, config: Dict, shared_data: Tuple):
    """
    Train a single model on one GPU with the given configuration and shared data.
    Handles model setup, data loading, training, and checkpointing for that GPU.
    """
    try:
        print(f" Starting training on GPU {gpu_id} - {config['model_name']}")
        print(f"   Generator flags: {config['generator_flags']}")
        print(f"   Model flags: {config['model_flags']}")
        
        # Select the target GPU and free any cached memory to maximize available resources
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.empty_cache()
        
        # Unpack the pre-loaded shared dataset (reduces disk I/O across processes)
        scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs = shared_data
        
        # Create the loss function, optionally using class weights to handle imbalance
        if config.get('weighted', False):
            weight_tensor = compute_weight_tensor(labels_for_neighbor_pairs,
                                                  config['num_classes'],
                                                  device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"GPU {gpu_id}: Using weighted loss function")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"GPU {gpu_id}: Using standard loss function")
        
        # Build the model with custom hyperparameters and flags from the config
        model = MultiEdgeClassifier(
            input_dim=config['num_features'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['num_classes'],
            device=device,
            num_layers=config['model_flags']['num_layers'],
            layer_weights=config['model_flags']['layer_weights'],
            softmax=config['model_flags']['softmax'],
            debug=config.get('debug', False)
        ).to(device)
        
        # Use Adam optimizer with configured learning rate and weight decay
        optimizer = optim.Adam(model.parameters(),
                               lr=config['lr'],
                               weight_decay=config['weight_decay'])
        
        # Assemble generator arguments, merging global data with per-run flags
        gen_kwargs = {
            'features_dict': scaled_data,
            'neighbor_pairs': neighbor_pairs_list,
            'labels': labels_for_neighbor_pairs,
            'class_counts': config['class_counts'],
            'batch_size': config['batch_size'],
            'unscaled_data_dict': unscaled_data,
            'debug': config.get('debug', False),
            **config['generator_flags']  # include generator-specific options
        }
        train_kwargs = {**gen_kwargs, 'mode': 'train'}
        test_kwargs = {**gen_kwargs, 'mode': 'test'}
        
        # Launch the training loop and collect metrics and the best model path
        metrics, model, model_path = run_model(
            model=model,
            batch_size=config['batch_size'],
            save_dir=config['save_dir'],
            best_model_name=config['model_name'],
            train_generator_class=MultiClassBatchGenerator,
            test_generator_class=MultiClassBatchGenerator,
            train_generator_kwargs=train_kwargs,
            test_generator_kwargs=test_kwargs,
            epochs=config['epochs'],
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            unscaled_data_dict=unscaled_data,
            debug=config.get('debug', False),
            resume=config.get('resume', True),
            patience=config.get('patience', 10),
            delta=config.get('delta', 0.0001)
        )
        
        # Report summary for this GPU after training
        print(f" GPU {gpu_id} training completed!")
        print(f"   Best accuracy: {metrics['best_test_acc']:.4f}")
        print(f"   Total time: {metrics['total_time']:.1f}s")
        print(f"   Model saved to: {model_path}")
        
        return 0  # Return 0 to indicate success
        
    except Exception as e:
        # Log any errors and return non-zero to indicate failure
        print(f" GPU {gpu_id} training failed: {e}")
        traceback.print_exc()
        return 1


def main():
    """
    Coordinate training across multiple GPUs.
    Loads data once, spawns a process per GPU, monitors progress,
    and summarizes final results.
    """
    print("=" * 50)
    print(" Multi-GPU Particle Physics Classification Training")
    print("=" * 50)
    
    # Load shared data into memory once to avoid redundant reads
    print("\n Loading shared data...")
    load_path = "/storage/mxg1065/datafiles"
    try:
        shared_data = load_shared_data(load_path)
        print(" Shared data loaded successfully!")
    except Exception as e:
        print(f" Failed to load data: {e}")
        return

    # Detect available GPUs and display their specs
    available_gpus = torch.cuda.device_count()
    print(f"\n Available GPUs: {available_gpus}")
    for i in range(available_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} - "
              f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    # Base hyperparameters and defaults shared by all GPU configs
    base_config = {
        'num_features': 3,
        'num_classes': 5,
        'hidden_dim': 128,
        'class_counts': {0: 121318, 1: 1113965, 2: 90603, 3: 95347, 4: 20476},
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'epochs': 200,
        'batch_size': 1,
        'save_dir': "/storage/mxg1065/padding_models",
        'resume': True,
        'patience': 20,
        'delta': 0.0001,
        'debug': False,
        'weighted': False,
        'generator_flags': {  # defaults for the data generator
            'padding': False,
            'is_bi_directional': True,
            'with_labels': False,
            'padding_class': 0,
            'train_ratio': 0.7
        },
        'model_flags': {     # defaults for the model
            'num_layers': 6,
            'layer_weights': False,
            'softmax': False
        }
    }

    # Define per-GPU configurations with small variations
    configs = [
        {**base_config,
         'model_name': "padding_with_ll_labeled_cc.pt",
         'description': "Padding with Lone-Lone Pairs but Labeling them as Cluster-Cluster",
         'generator_flags': {**base_config['generator_flags'], 'padding': True}},
        {**base_config,
         'model_name': "padding_with_tt_labeled_cc.pt",
         'description': "Padding with True-True Pairs but Labeling them as Cluster-Cluster",
         'generator_flags': {**base_config['generator_flags'], 'padding': True, 'padding_class': 1}},
        {**base_config,
         'model_name': "padding_with_ll_labeled_ll.pt",
         'description': "Padding with Lone-Lone Pairs and Labeling them as Lone-Lone",
         'generator_flags': {**base_config['generator_flags'], 'padding': True, 'with_labels': True}},
        {**base_config,
         'model_name': "padding_with_tt_labeled_tt.pt",
         'description': "Padding with True-True Pairs and Labeling them as True-True",
         'generator_flags': {**base_config['generator_flags'], 'padding': True,
                             'padding_class': 1, 'with_labels': True}}
    ]

    # Trim configs to match the number of detected GPUs
    configs = configs[:available_gpus]
    
    # Display planned training jobs for confirmation
    print(f"\n Configuring {len(configs)} models:")
    for i, config in enumerate(configs):
        print(f"   GPU {i}: {config['description']}")
        print(f"      → Model: {config['model_name']}")
        print(f"      → Hidden dim: {config['hidden_dim']}, LR: {config['lr']}")
        print(f"      → Weighted: {config['weighted']}")

    # Prompt user before starting multi-GPU training
    print(f"\n  This will start {len(configs)} training processes.")
    response = input("Continue? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return

    # Spawn one process per GPU and track them for monitoring
    print(f"\n Starting training processes...")
    processes = []
    start_times = []
    
    for gpu_id, config in enumerate(configs):
        try:
            # Clear cache on the specific GPU to maximize free memory
            print(f"   Clearing cache for GPU {gpu_id}...")
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # ensure cleanup completes
            
            # Show available memory for that GPU
            free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
            print(f"   GPU {gpu_id} free memory before start: {free_mem:.2f} GB")
            
            # Launch training as a separate process
            print(f"   Starting process on GPU {gpu_id}: {config['description']}")
            p = mp.Process(
                target=train_single_gpu,
                args=(gpu_id, config, shared_data),
                daemon=False
            )
            p.start()
            processes.append(p)
            start_times.append(time.time())
            
            # Stagger launches to avoid simultaneous heavy GPU allocation
            if gpu_id < len(configs) - 1:
                time.sleep(25)
                    
        except Exception as e:
            print(f" Failed to start process for GPU {gpu_id}: {e}")

    # Periodically monitor running processes and report progress
    print(f"\n Monitoring {len(processes)} training processes...")
    print("   Press Ctrl+C to stop all processes")
    finished = set()
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(30)  # Check every 30 seconds
            for i, p in enumerate(processes):
                if not p.is_alive() and i not in finished:
                    exitcode = p.exitcode
                    if exitcode == 0:
                        print(f" GPU {i} finished training successfully.")
                    else:
                        print(f" GPU {i} crashed with exit code {exitcode}.")
                    finished.add(i)
            alive_count = sum(p.is_alive() for p in processes)
            elapsed = time.time() - min(start_times) if start_times else 0
            print(f"   {alive_count}/{len(processes)} processes still running - "
                  f"Elapsed: {elapsed/60:.1f} min")

        print("\n  All processes finished!")

    except KeyboardInterrupt:
        # Gracefully terminate all processes on user interrupt
        print(f"\n  Keyboard interrupt received. Stopping all processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("All processes terminated.")

    # Ensure all processes have completely exited
    print("\n Waiting for all processes to complete...")
    for i, p in enumerate(processes):
        p.join()
        print(f"   Process {i} completed with exitcode: {p.exitcode}")

    print("\n" + "=" * 50)
    print(" All training processes completed!")
    print("=" * 50)
    
    # Summarize training results by reading each GPU's metrics file
    print("\n Training Summary:")
    for i, config in enumerate(configs):
        model_path = os.path.join(config['save_dir'], config['model_name'])
        metrics_path = os.path.splitext(model_path)[0] + ".pkl"
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
                best_acc = metrics.get('best_test_acc', 0)
                total_time = metrics.get('total_time', 0)
                min_time, sec_time = divmod(total_time, 60)
                print(f"   GPU {i}: {config['description']}")
                print(f"      Best Accuracy: {best_acc:.4f}")
                print(f"      Total Time: {int(min_time)}m {sec_time:.1f}s")
                print(f"      Model: {model_path}")
            except:
                print(f"   GPU {i}: Could not load metrics")
        else:
            print(f"   GPU {i}: No results found")

    print("\n Multi-GPU training completed successfully!")


if __name__ == "__main__":
    # Ensure correct multiprocessing start method for CUDA
    mp.set_start_method('spawn', force=True)
    
    # Free any cached GPU memory before starting
    torch.cuda.empty_cache()
    
    # Handle graceful shutdown on SIGINT/SIGTERM
    def signal_handler(sig, frame):
        print("\n Received shutdown signal. Exiting gracefully...")
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Kick off the multi-GPU training workflow
    main()