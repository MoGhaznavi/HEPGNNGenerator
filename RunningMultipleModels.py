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


# Set multiprocessing start method early
mp.set_start_method('spawn', force=True)

# Set global torch settings for better performance
torch.backends.cudnn.benchmark = True # Limits determinism
torch.set_num_threads(2)  # Limit CPU threads per process

def load_pickle(filename: str) -> Any:
    """Load regular pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_shared_data(load_path: str) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Load all shared data once to be used across all GPU processes
    Returns: (scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs)
    """
    print("Loading shared data...")
    
    scaled_data = load_pickle(os.path.join(load_path, "scaled_data_1000events.pkl"))
    unscaled_data = load_pickle(os.path.join(load_path, "data_1000events.pkl"))
    neighbor_pairs_list = load_pickle(os.path.join(load_path, "neighbor_pairs_list.pkl"))
    labels_for_neighbor_pairs = load_pickle(os.path.join(load_path, "labels_for_neighbor_pairs_1000events.pkl"))
    
    print(f"Loaded data shapes:")
    print(f"  scaled_data['data_0']: {scaled_data['data_0'].shape}")
    print(f"  unscaled_data['data_0']: {unscaled_data['data_0'].shape}")
    print(f"  neighbor_pairs_list: {neighbor_pairs_list.shape}")
    print(f"  labels_for_neighbor_pairs: {labels_for_neighbor_pairs.shape}")

    return scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs


def compute_weight_tensor(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute class weights for imbalanced dataset"""
    counts = np.bincount(labels.flatten())
    weights = 1 / np.sqrt(counts + 1e-5)
    weights /= (weights.sum() * num_classes)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def save_data_pickle(filename: str, directory: str, data: Any) -> None:
    """Save data to pickle file"""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def find_latest_checkpoint(save_dir: str, base_name: str) -> Optional[Tuple[int, str]]:
    """Find the latest checkpoint file"""
    pattern = re.compile(f"{re.escape(os.path.splitext(base_name)[0])}_epoch(\\d+).pt")
    checkpoints = []
    for fname in os.listdir(save_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, os.path.join(save_dir, fname)))
    return max(checkpoints, key=lambda x: x[0]) if checkpoints else None

# class MultiClassBatchGenerator(IterableDataset):
#     """
#     Optimized batch generator for multi-class classification
#     Precomputes samples to avoid redundant computation during training
#     """
    
#     def __init__(self, features_dict: Dict, neighbor_pairs: np.ndarray, labels: np.ndarray, 
#                  class_counts: Dict, mode: str = 'train', is_bi_directional: bool = True, 
#                  batch_size: int = 1, train_ratio: float = 0.7, padding: bool = False, 
#                  with_labels: bool = False, padding_class: int = 0, debug: bool = False,
#                  unscaled_data_dict: Optional[Dict] = None):
        
#         self.debug = debug
#         self.is_bi_directional = is_bi_directional
#         self.batch_size = batch_size
#         self.padding = padding
#         self.with_labels = with_labels
#         self.padding_class = padding_class

#         # Scaled features (already normalized)
#         self.features_dict = {
#             k: torch.as_tensor(v, dtype=torch.float32)
#             for k, v in features_dict.items()
#         }

#         # Unscaled features (for analysis/logging)
#         if unscaled_data_dict is not None:
#             self.unscaled_features_dict = self._precompute_unscaled_features(unscaled_data_dict)
#         else:
#             self.unscaled_features_dict = None

#         # Labels and pairs
#         self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
#         self.labels = torch.tensor(labels, dtype=torch.long)
#         self.class_counts = class_counts

#         # Train/test split
#         num_events = len(features_dict)
#         train_size = int(num_events * train_ratio)
#         self.event_indices = list(range(train_size)) if mode == 'train' else list(range(train_size, num_events))

#         # Precompute all samples during initialization
#         self.precomputed_samples = self._precompute_all_samples()
        
#         if self.debug:
#             print(f"MultiClassBatchGenerator initialized with {len(self.precomputed_samples)} samples")
#             print(f"Mode: {mode}, Events: {len(self.event_indices)}")

#     def _precompute_unscaled_features(self, unscaled_data_dict: Dict) -> Dict:
#         """Precompute unscaled features once during initialization"""
#         unscaled_features = {}
#         for k, v in unscaled_data_dict.items():
#             tensor_v = torch.as_tensor(v, dtype=torch.float32)
#             # If tensor has 4 columns, original code applies
#             if tensor_v.shape[1] == 4:
#                 eta = tensor_v[:, 1]
#                 phi = torch.atan2(tensor_v[:, 3], tensor_v[:, 2])
#                 tensor_v = torch.cat([tensor_v[:, 0:1], eta[:, None], phi[:, None]], dim=1)
#             elif tensor_v.shape[1] == 3:
#                 # Already [SNR, η, φ], so just transform η if needed
#                 eta = tensor_v[:, 1]
#                 phi = tensor_v[:, 2]  # φ is already present
#                 tensor_v = torch.stack([tensor_v[:, 0], eta, phi], dim=1)
#             else:
#                 raise ValueError(f"Unexpected feature dimension {tensor_v.shape[1]} for key {k}")
#             unscaled_features[k] = tensor_v
#         return unscaled_features

#     def _build_class_indices(self, event_id: int) -> Dict[int, torch.Tensor]:
#         """Build class indices for a specific event"""
#         class_indices = {}
#         for cls in self.class_counts.keys():
#             mask = self.labels[event_id] == cls
#             class_indices[cls] = torch.where(mask)[0]
#         return class_indices

#     def _sample_edges(self, event_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Sample edges for a specific event with class balancing"""
#         class_indices = self._build_class_indices(event_id)
#         sampled_indices = []
#         padded_mask = []
        
#         for cls, count in self.class_counts.items():
#             indices = class_indices.get(cls, torch.tensor([], dtype=torch.long))
#             if len(indices) > 0:
#                 selected = indices[torch.randperm(len(indices))[:min(count, len(indices))]]
#                 sampled_indices.append(selected)
#                 padded_mask.extend([True] * len(selected))

#         sampled = torch.cat(sampled_indices) if sampled_indices else torch.tensor([], dtype=torch.long)

#         if self.padding and len(sampled) < sum(self.class_counts.values()):
#             pad_size = sum(self.class_counts.values()) - len(sampled)
#             pad_indices = class_indices.get(self.padding_class, torch.tensor([], dtype=torch.long))
#             if len(pad_indices) > 0:
#                 pad_indices = pad_indices[:pad_size]
#                 sampled = torch.cat([sampled, pad_indices])
#                 padded_mask.extend([False] * len(pad_indices))

#         return sampled, torch.tensor(padded_mask, dtype=torch.bool)

#     def _precompute_all_samples(self) -> List[Tuple]:
#         """Precompute all samples to avoid computation during iteration"""
#         samples = []
        
#         if self.debug:
#             all_eta_ranges = {cls: {'min': np.inf, 'max': -np.inf} for cls in range(5)}
#             global_eta_min = np.inf
#             global_eta_max = -np.inf
        
#         for event_id in self.event_indices:
#             edge_sample_idx, padded_mask = self._sample_edges(event_id)
            
#             if len(edge_sample_idx) == 0:
#                 continue  # Skip events with no samples
                
#             selected_pairs = self.neighbor_pairs[edge_sample_idx].T
#             selected_labels = self.labels[event_id, edge_sample_idx].clone()

#             if self.padding and not self.with_labels:
#                 selected_labels[~padded_mask] = self.padding_class

#             x_scaled = self.features_dict[f"data_{event_id}"]
#             x_unscaled = self.unscaled_features_dict.get(f"data_{event_id}") if self.unscaled_features_dict else None

#             samples.append(
#                 (x_scaled, selected_pairs, selected_pairs.clone(), 
#                  selected_labels.unsqueeze(1), x_unscaled)
#             )
            
#             if self.debug and x_unscaled is not None:
#                 # Update global η range
#                 event_eta_min = x_unscaled[:, 1].min().item()
#                 event_eta_max = x_unscaled[:, 1].max().item()
#                 global_eta_min = min(global_eta_min, event_eta_min)
#                 global_eta_max = max(global_eta_max, event_eta_max)
                
#                 # Update per-class η ranges
#                 for cls in range(5):
#                     cls_mask = selected_labels == cls
#                     if cls_mask.any():
#                         cls_pairs = selected_pairs[:, cls_mask]
#                         cls_eta0 = x_unscaled[cls_pairs[0]][:, 1]
#                         cls_eta1 = x_unscaled[cls_pairs[1]][:, 1]
#                         cls_eta = torch.cat([cls_eta0, cls_eta1])
                        
#                         all_eta_ranges[cls]['min'] = min(all_eta_ranges[cls]['min'], cls_eta.min().item())
#                         all_eta_ranges[cls]['max'] = max(all_eta_ranges[cls]['max'], cls_eta.max().item())
        
#         if self.debug:
#             print(f"\nGlobal η range across all events: {global_eta_min:.6f} → {global_eta_max:.6f}")
#             for cls in range(5):
#                 if all_eta_ranges[cls]['min'] != np.inf:
#                     print(f"Class {cls} η range: {all_eta_ranges[cls]['min']:.6f} → {all_eta_ranges[cls]['max']:.6f}")
#                 else:
#                     print(f"Class {cls}: no samples")
        
#         return samples

#     def __iter__(self):
#         """Iterate through precomputed samples"""
#         for sample in self.precomputed_samples:
#             yield sample

#     def __len__(self):
#         return len(self.precomputed_samples)

#     @staticmethod
#     def collate_data(batch: List[Tuple]) -> Tuple:
#         """Collate function for DataLoader"""
#         x_list = [b[0] for b in batch]
#         edge_index_list = [b[1] for b in batch]
#         edge_index_out_list = [b[2] for b in batch]
#         y_batch = torch.cat([b[3] for b in batch], dim=0)
    
#         # Handle optional unscaled features
#         if batch[0][4] is None:
#             unscaled_list = None
#         else:
#             unscaled_list = [b[4] for b in batch]
    
#         return x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list

from torch.utils.data import IterableDataset
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class MultiClassBatchGenerator(IterableDataset):
    """
    Optimized batch generator for multi-class classification.
    Precomputes samples to avoid redundant computation during training.
    """

    def __init__(
        self,
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
        self.debug = debug
        self.is_bi_directional = is_bi_directional
        self.batch_size = batch_size
        self.padding = padding
        self.with_labels = with_labels
        self.padding_class = padding_class

        # --- Scaled features ---
        self.features_dict = {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in features_dict.items()
        }

        # --- Optional unscaled features (for analysis/logging) ---
        if unscaled_data_dict is not None:
            self.unscaled_features_dict = self._precompute_unscaled_features(
                unscaled_data_dict
            )
        else:
            self.unscaled_features_dict = None

        # --- Neighbor pairs and labels ---
        self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.class_counts = class_counts

        # --- Train/test split by event ---
        num_events = len(features_dict)
        train_size = int(num_events * train_ratio)
        self.event_indices = (
            list(range(train_size))
            if mode == "train"
            else list(range(train_size, num_events))
        )

        # --- Precompute all samples once ---
        self.precomputed_samples = self._precompute_all_samples()

        if self.debug:
            print(
                f"MultiClassBatchGenerator initialized with "
                f"{len(self.precomputed_samples)} samples"
            )
            print(f"Mode: {mode}, Events: {len(self.event_indices)}")

    # ------------------------------------------------------------------ #
    def _precompute_unscaled_features(self, unscaled_data_dict: Dict) -> Dict:
        """Prepare unscaled features in [SNR, η, φ] format if needed."""
        unscaled_features = {}
        for k, v in unscaled_data_dict.items():
            tensor_v = torch.as_tensor(v, dtype=torch.float32)
            if tensor_v.shape[1] == 4:
                eta = tensor_v[:, 1]
                phi = torch.atan2(tensor_v[:, 3], tensor_v[:, 2])
                tensor_v = torch.cat([tensor_v[:, 0:1], eta[:, None], phi[:, None]], dim=1)
            elif tensor_v.shape[1] == 3:
                # Already [SNR, η, φ]
                tensor_v = torch.stack(
                    [tensor_v[:, 0], tensor_v[:, 1], tensor_v[:, 2]], dim=1
                )
            else:
                raise ValueError(
                    f"Unexpected feature dimension {tensor_v.shape[1]} for key {k}"
                )
            unscaled_features[k] = tensor_v
        return unscaled_features

    # ------------------------------------------------------------------ #
    def _build_class_indices(self, event_id: int) -> Dict[int, torch.Tensor]:
        """Return indices of neighbor pairs belonging to each class for one event."""
        class_indices = {}
        for cls in self.class_counts.keys():
            mask = self.labels[event_id] == cls
            class_indices[cls] = torch.where(mask)[0]
        return class_indices

    # ------------------------------------------------------------------ #
    def _sample_edges(
        self, event_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample neighbor pairs for one event,
        return sampled indices, padded_mask, and processed out_labels.
        """
        class_indices = self._build_class_indices(event_id)
        sampled_indices: List[torch.Tensor] = []
        padded_mask: List[bool] = []

        # --- balanced sampling by class ---
        for cls, count in self.class_counts.items():
            indices = class_indices.get(cls, torch.tensor([], dtype=torch.long))
            if len(indices) > 0:
                selected = indices[torch.randperm(len(indices))[: min(count, len(indices))]]
                sampled_indices.append(selected)
                padded_mask.extend([True] * len(selected))

        sampled = (
            torch.cat(sampled_indices)
            if sampled_indices
            else torch.tensor([], dtype=torch.long)
        )

        # --- optional padding with specified class ---
        if self.padding and len(sampled) < sum(self.class_counts.values()):
            pad_size = sum(self.class_counts.values()) - len(sampled)
            pad_indices = class_indices.get(
                self.padding_class, torch.tensor([], dtype=torch.long)
            )
            if len(pad_indices) > 0:
                pad_indices = pad_indices[:pad_size]
                sampled = torch.cat([sampled, pad_indices])
                padded_mask.extend([False] * len(pad_indices))

        padded_mask = torch.tensor(padded_mask, dtype=torch.bool)

        # --- build output labels based on with_labels flag ---
        true_labels = self.labels[event_id][sampled]
        if self.with_labels:
            out_labels = true_labels  # keep real labels for everything
        else:
            # keep real labels for sampled nodes, but mark padded nodes as 4
            out_labels = true_labels.clone()
            out_labels[~padded_mask] = 4

        return sampled, padded_mask, out_labels

    # ------------------------------------------------------------------ #
    def _precompute_all_samples(self) -> List[Tuple]:
        """Precompute all samples to avoid computation during iteration."""
        samples: List[Tuple] = []

        if self.debug:
            all_eta_ranges = {cls: {"min": np.inf, "max": -np.inf} for cls in range(5)}
            global_eta_min, global_eta_max = np.inf, -np.inf

        for event_id in self.event_indices:
            edge_sample_idx, padded_mask, out_labels = self._sample_edges(event_id)
            if len(edge_sample_idx) == 0:
                continue

            selected_pairs = self.neighbor_pairs[edge_sample_idx].T
            x_scaled = self.features_dict[f"data_{event_id}"]
            x_unscaled = (
                self.unscaled_features_dict.get(f"data_{event_id}")
                if self.unscaled_features_dict
                else None
            )

            samples.append(
                (
                    x_scaled,
                    selected_pairs,
                    selected_pairs.clone(),
                    out_labels.unsqueeze(1),
                    x_unscaled,
                )
            )

            if self.debug and x_unscaled is not None:
                event_eta_min = x_unscaled[:, 1].min().item()
                event_eta_max = x_unscaled[:, 1].max().item()
                global_eta_min = min(global_eta_min, event_eta_min)
                global_eta_max = max(global_eta_max, event_eta_max)
                for cls in range(5):
                    cls_mask = out_labels == cls
                    if cls_mask.any():
                        cls_pairs = selected_pairs[:, cls_mask]
                        cls_eta0 = x_unscaled[cls_pairs[0]][:, 1]
                        cls_eta1 = x_unscaled[cls_pairs[1]][:, 1]
                        cls_eta = torch.cat([cls_eta0, cls_eta1])
                        all_eta_ranges[cls]["min"] = min(
                            all_eta_ranges[cls]["min"], cls_eta.min().item()
                        )
                        all_eta_ranges[cls]["max"] = max(
                            all_eta_ranges[cls]["max"], cls_eta.max().item()
                        )

        if self.debug:
            print(
                f"\nGlobal η range across all events: {global_eta_min:.6f} → {global_eta_max:.6f}"
            )
            for cls in range(5):
                if all_eta_ranges[cls]["min"] != np.inf:
                    print(
                        f"Class {cls} η range: "
                        f"{all_eta_ranges[cls]['min']:.6f} → {all_eta_ranges[cls]['max']:.6f}"
                    )
                else:
                    print(f"Class {cls}: no samples")

        return samples

    # ------------------------------------------------------------------ #
    def __iter__(self):
        for sample in self.precomputed_samples:
            yield sample

    def __len__(self):
        return len(self.precomputed_samples)

    # ------------------------------------------------------------------ #
    @staticmethod
    def collate_data(batch: List[Tuple]) -> Tuple:
        """
        Collate function for DataLoader.
        Returns (x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list).
        """
        x_list = [b[0] for b in batch]
        edge_index_list = [b[1] for b in batch]
        edge_index_out_list = [b[2] for b in batch]
        y_batch = torch.cat([b[3] for b in batch], dim=0)

        unscaled_list = None if batch[0][4] is None else [b[4] for b in batch]
        return x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list


# class MultiEdgeClassifier(nn.Module):
#     """Graph Neural Network for edge classification"""
    
#     def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, 
#                  num_layers: int = 6, layer_weights: bool = False, softmax: bool = False, 
#                  debug: bool = False):
#         super().__init__()
#         self.device = device
#         self.debug = debug
#         self.layer_weights_enabled = layer_weights
#         self.softmax = softmax
#         self.num_layers = num_layers

#         # Node embedding
#         self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
#         # GCN layers with batch normalization
#         self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
#         self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
#         # Final classification layer
#         self.fc = nn.Linear(2 * hidden_dim, output_dim)

#         # Layer weights (optional)
#         if self.layer_weights_enabled:
#             self.layer_weights = nn.Parameter(torch.ones(num_layers))
#         else:
#             self.layer_weights = None

#     def forward(self, x_list: List[torch.Tensor], edge_index_list: List[torch.Tensor], 
#                 edge_index_out_list: List[torch.Tensor], y_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        
#         if self.debug:
#             total_start = time.perf_counter()
#             timings = {
#                 "weight_prep": 0.0,
#                 "move_to_device": 0.0,
#                 "node_embedding": 0.0,
#                 "layers": [0.0 for _ in range(self.num_layers)],
#                 "edge_repr": 0.0,
#                 "final_fc": 0.0
#             }

#         all_edge_reprs = []

#         # Normalize weights if enabled
#         if self.debug:
#             t0 = time.perf_counter()
#         if self.layer_weights_enabled:
#             weights = torch.softmax(self.layer_weights, dim=0) if self.softmax else self.layer_weights
#         else:
#             weights = None
#         if self.debug:
#             timings["weight_prep"] += time.perf_counter() - t0

#         # Process each graph in the batch
#         for x, processed_edges, original_edges in zip(x_list, edge_index_list, edge_index_out_list):
#             if self.debug:
#                 t1 = time.perf_counter()
#             x = x.to(self.device, non_blocking=True)
#             processed_edges = processed_edges.to(self.device, non_blocking=True)
#             if self.debug:
#                 timings["move_to_device"] += time.perf_counter() - t1

#             # Node embedding
#             if self.debug:
#                 t2 = time.perf_counter()
#             x_embed = self.node_embedding(x)
#             if self.debug:
#                 timings["node_embedding"] += time.perf_counter() - t2

#             # GCN layers with residual connections
#             for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
#                 if self.debug:
#                     t_layer = time.perf_counter()
#                 h = torch.relu(bn(conv(x_embed, processed_edges)))
#                 if self.layer_weights_enabled:
#                     h = weights[i] * h
#                 x_embed = x_embed + h  # Residual connection
#                 if self.debug:
#                     timings["layers"][i] += time.perf_counter() - t_layer

#             # Build edge representations
#             if self.debug:
#                 t3 = time.perf_counter()
#             src, dst = original_edges[0], original_edges[1]
#             edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
#             all_edge_reprs.append(edge_repr)
#             if self.debug:
#                 timings["edge_repr"] += time.perf_counter() - t3

#         # Final classification
#         if self.debug:
#             t4 = time.perf_counter()
#         out = self.fc(torch.cat(all_edge_reprs, dim=0))
#         if self.debug:
#             timings["final_fc"] += time.perf_counter() - t4
#             total_time = time.perf_counter() - total_start

#             print("\n[Forward Pass Timing Summary]")
#             print(f"  Weight preparation: {timings['weight_prep']:.6f} s")
#             print(f"  Move to device: {timings['move_to_device']:.6f} s")
#             print(f"  Node embedding: {timings['node_embedding']:.6f} s")
#             for i, t in enumerate(timings["layers"]):
#                 print(f"  Layer {i}: {t:.6f} s")
#             print(f"  Edge representation build: {timings['edge_repr']:.6f} s")
#             print(f"  Final FC: {timings['final_fc']:.6f} s")
#             print(f"  TOTAL forward: {total_time:.6f} s\n")

#         return out

# def train_model(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, 
#                 criterion: nn.Module, scaler: GradScaler, device: torch.device, 
#                 debug: bool = False) -> Dict[str, float]:
#     """
#     Optimized training function for single epoch
#     """
#     model.train()
#     total_loss = correct = total = 0
#     start_train = time.perf_counter()

#     for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list) in enumerate(loader):
#         if debug:
#             start_batch = time.perf_counter()

#         # Move batch to device (optimized with list comprehensions)
#         x_list = [x.to(device, non_blocking=True) for x in x_list]
#         edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
#         edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
#         y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

#         optimizer.zero_grad(set_to_none=True)  # Faster zero_grad

#         # Forward + loss with mixed precision
#         with torch.amp.autocast(device_type="cuda"):
#             scores = model(x_list, edge_idx_list, edge_idx_out_list)
#             loss = criterion(scores, y_batch)

#         # Backward + optimizer step
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # Metrics
#         total_loss += loss.item() * len(y_batch)
#         preds = scores.argmax(dim=1)
#         correct += (preds == y_batch).sum().item()
#         total += len(y_batch)

#         if debug and batch_idx % 10 == 0:  # Reduced debug output frequency
#             batch_time = time.perf_counter() - start_batch
#             print(f"Batch {batch_idx+1}: loss={loss.item():.4f}, time={batch_time:.3f}s")

#     epoch_time = time.perf_counter() - start_train
    
#     if debug:
#         print(f"Training epoch completed in {epoch_time:.2f}s")

#     return {
#         'loss': total_loss / total if total > 0 else 0,
#         'acc': correct / total if total > 0 else 0
#     }

# def test_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
#                device: torch.device, debug: bool = False) -> Dict[str, float]:
#     """
#     Optimized testing function
#     """
#     model.eval()
#     total_loss = correct = total = 0
#     start_test = time.perf_counter()

#     with torch.no_grad():
#         for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list) in enumerate(loader):
#             # Move batch to device
#             x_list = [x.to(device, non_blocking=True) for x in x_list]
#             edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
#             edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
#             y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

#             # Forward + loss with mixed precision
#             with torch.amp.autocast(device_type="cuda"):
#                 scores = model(x_list, edge_idx_list, edge_idx_out_list)
#                 loss = criterion(scores, y_batch)

#             # Metrics
#             total_loss += loss.item() * len(y_batch)
#             preds = scores.argmax(dim=1)
#             correct += (preds == y_batch).sum().item()
#             total += len(y_batch)

#     test_time = time.perf_counter() - start_test
    
#     if debug:
#         print(f"Testing completed in {test_time:.2f}s")

#     return {
#         'loss': total_loss / total if total > 0 else 0,
#         'acc': correct / total if total > 0 else 0
#     }

# @torch.no_grad()
# def run_inference(model, loader, device, unscaled_data_dict=None, debug=False):
#     """
#     Inference for MultiEdgeClassifier with optional unscaled features.
#     Returns predictions, scores (all classes), labels, features, and neighbor pairs.
#     """
#     model.eval()
#     if debug:
#         start_time = time.perf_counter()

#     # Storage
#     all_preds, all_scores, all_labels = [], [], []
#     all_features_i, all_features_j = [], []
#     all_features_i_unscaled, all_features_j_unscaled = [], []
#     all_neighbor_pairs = []

#     has_unscaled = unscaled_data_dict is not None

#     for batch in loader:
#         # Unpack batch
#         x_list, edge_index_list, edge_index_out_list, y_batch, event_idx = batch

#         # Move to device
#         x_list = [x.to(device) for x in x_list]
#         edge_index_list = [e.to(device) for e in edge_index_list]
#         edge_index_out_list = [e.to(device) for e in edge_index_out_list]
#         if y_batch is not None:
#             y_batch = y_batch.to(device)

#         # Forward pass
#         out = model(x_list, edge_index_list, edge_index_out_list, y_batch)
#         preds = out.argmax(dim=1).cpu()
#         scores = torch.softmax(out, dim=1).cpu()  # keep **all class probabilities**
#         all_preds.append(preds)
#         all_scores.append(scores)
#         if y_batch is not None:
#             all_labels.append(y_batch.cpu())

#         # Edge features for scaled values
#         src_nodes, dst_nodes = edge_index_out_list[0]  # assuming single graph per batch
#         feats_i_scaled = x_list[0][src_nodes].cpu()
#         feats_j_scaled = x_list[0][dst_nodes].cpu()
#         all_features_i.append(feats_i_scaled)
#         all_features_j.append(feats_j_scaled)

#         # Neighbor pairs (src, dst indices)
#         all_neighbor_pairs.append(torch.stack([src_nodes.cpu(), dst_nodes.cpu()], dim=1))

#         # Optional unscaled features
#         if has_unscaled:
#             src_indices = src_nodes.cpu().numpy()
#             dst_indices = dst_nodes.cpu().numpy()
#             feats_i_unscaled = np.zeros((len(src_nodes), 3), dtype=np.float32)
#             feats_j_unscaled = np.zeros((len(dst_nodes), 3), dtype=np.float32)

#             event_idx_np = event_idx.cpu().numpy()
#             for evt in np.unique(event_idx_np):
#                 mask_src = event_idx_np[src_indices] == evt
#                 mask_dst = event_idx_np[dst_indices] == evt
#                 if mask_src.any():
#                     feats_i_unscaled[mask_src] = unscaled_data_dict[f"data_{evt}"][src_indices[mask_src]]
#                 if mask_dst.any():
#                     feats_j_unscaled[mask_dst] = unscaled_data_dict[f"data_{evt}"][dst_indices[mask_dst]]

#             all_features_i_unscaled.append(torch.tensor(feats_i_unscaled, dtype=torch.float32))
#             all_features_j_unscaled.append(torch.tensor(feats_j_unscaled, dtype=torch.float32))

#     # Concatenate results
#     preds = torch.cat(all_preds).numpy()
#     scores = torch.cat(all_scores).numpy()  # shape = (num_edges, num_classes)
#     labels = torch.cat(all_labels).numpy() if all_labels else None
#     features_i_scaled = torch.cat(all_features_i).numpy()
#     features_j_scaled = torch.cat(all_features_j).numpy()
#     neighbor_pairs = torch.cat(all_neighbor_pairs).numpy() if all_neighbor_pairs else None

#     # Reconstruct phi from scaled features
#     features_i = np.stack([
#         features_i_scaled[:, 0],  # SNR_scaled
#         features_i_scaled[:, 1],  # eta
#         np.arctan2(features_i_scaled[:, 3], features_i_scaled[:, 2])  # phi
#     ], axis=1)
#     features_j = np.stack([
#         features_j_scaled[:, 0],
#         features_j_scaled[:, 1],
#         np.arctan2(features_j_scaled[:, 3], features_j_scaled[:, 2])
#     ], axis=1)

#     result = {
#         "preds": preds,
#         "scores": scores,
#         "labels": labels,
#         "features_i": features_i,
#         "features_j": features_j,
#         "neighbor_pairs": neighbor_pairs,
#     }

#     if has_unscaled and all_features_i_unscaled:
#         feats_i_unscaled = torch.cat(all_features_i_unscaled).numpy()
#         feats_j_unscaled = torch.cat(all_features_j_unscaled).numpy()
#         result["features_i_unscaled"] = feats_i_unscaled
#         result["features_j_unscaled"] = feats_j_unscaled

#     if debug:
#         elapsed = time.perf_counter() - start_time
#         print(f"[DEBUG] Inference completed in {elapsed:.2f} sec for {len(loader)} batches.")

#     return result

# def run_model(model: nn.Module, batch_size: int, save_dir: str, best_model_name: str,
#               train_generator_class: type, test_generator_class: type,
#               train_generator_kwargs: Dict, test_generator_kwargs: Dict,
#               epochs: int, device: torch.device, optimizer: optim.Optimizer, 
#               criterion: nn.Module, unscaled_data_dict: Optional[Dict] = None, 
#               lr: float = 1e-3, resume: bool = True, patience: int = 10, 
#               delta: float = 0.0001, debug: bool = False) -> Tuple[Dict, nn.Module, str]:
#     """
#     Training loop with per-epoch checkpoints and final best model inference.
#     """

#     # Make save directory
#     if not debug:
#         os.makedirs(save_dir, exist_ok=True)
    
#     model_path = os.path.join(save_dir, best_model_name)
#     best_model_path = os.path.join(save_dir, f"best_{best_model_name}")
#     metrics_path = os.path.splitext(model_path)[0] + ".pkl"

#     # Initialize training state
#     start_epoch = 1
#     best_test_acc = 0.0
#     best_test_loss = float("inf")
#     best_epoch = 0
#     total_time_trained = 0.0
#     scaler = torch.amp.GradScaler(device="cuda")

#     # Resume if checkpoint exists
#     if resume:
#         checkpoint_result = find_latest_checkpoint(save_dir, best_model_name)
#         if checkpoint_result:
#             resumed_epoch, latest_checkpoint_path = checkpoint_result
#             checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=True)
#             model.load_state_dict(checkpoint['model_state_dict'])
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             scaler.load_state_dict(checkpoint['scaler_state_dict'])
#             start_epoch = resumed_epoch + 1
#             if not debug:
#                 print(f"Resuming from checkpoint: {latest_checkpoint_path} (epoch {resumed_epoch})")
#             if os.path.exists(metrics_path):
#                 try:
#                     with open(metrics_path, 'rb') as f:
#                         metrics = pickle.load(f)
#                     best_test_acc = float(np.max(metrics.get('test_acc', [0.0])))
#                     best_epoch = int(np.argmax(metrics.get('test_acc', [0.0]))) + 1
#                     total_time_trained = metrics.get('total_time', 0.0)
#                 except:
#                     metrics = {}
#             else:
#                 metrics = {}
#         else:
#             if not debug:
#                 print("No previous checkpoint found. Starting from scratch.")
#             metrics = {}
#     else:
#         metrics = {}

#     # Initialize metrics
#     metrics.update({
#         'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [],
#         'epoch_times': [], 'best_epoch': best_epoch,
#         'best_test_acc': best_test_acc, 'best_test_loss': best_test_loss,
#         'total_time': total_time_trained, 'time_per_epoch': 0.0
#     })

#     # Create DataLoaders once
#     train_loader = DataLoader(
#         train_generator_class(**train_generator_kwargs),
#         batch_size=batch_size,
#         collate_fn=MultiClassBatchGenerator.collate_data,
#         pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_generator_class(**test_generator_kwargs),
#         batch_size=batch_size,
#         collate_fn=MultiClassBatchGenerator.collate_data,
#         pin_memory=True
#     )

#     early_stopping_counter = 0
    
#     # Epoch loop
#     for epoch in range(start_epoch, epochs + 1):
#         epoch_start_time = time.perf_counter()
    
#         # Train and test
#         train_results = train_model(model, train_loader, optimizer, criterion, scaler, device, debug=debug)
#         test_results = test_model(model, test_loader, criterion, device, debug=debug)
    
#         # Metrics update
#         epoch_time = time.perf_counter() - epoch_start_time
#         metrics['epoch_times'].append(epoch_time)
#         metrics['train_loss'].append(train_results['loss'])
#         metrics['test_loss'].append(test_results['loss'])
#         metrics['train_acc'].append(train_results['acc'])
#         metrics['test_acc'].append(test_results['acc'])
    
#         # Current test loss
#         current_test_loss = test_results['loss']
    
#         # Check for real improvement
#         if current_test_loss < best_test_loss:
#             best_test_loss = current_test_loss
#             best_test_acc = test_results['acc']
#             best_epoch = epoch
#             metrics['best_epoch'] = best_epoch
#             metrics['best_test_acc'] = best_test_acc
#             metrics['best_test_loss'] = best_test_loss
#             early_stopping_counter = 0  # reset counter on real improvement
    
#             # Save best model
#             if not debug:
#                 torch.save({
#                     'epoch': best_epoch,
#                     'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'scaler_state_dict': scaler.state_dict(),
#                     'best_test_acc': best_test_acc,
#                     'best_test_loss': best_test_loss
#                 }, best_model_path)
#         else:
#             # Increment counter only if improvement < delta
#             if current_test_loss > best_test_loss - delta:
#                 early_stopping_counter += 1
    
#         # Save checkpoint every epoch
#         if not debug:
#             checkpoint_path = os.path.join(save_dir, f"{os.path.splitext(best_model_name)[0]}_epoch{epoch}.pt")
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scaler_state_dict': scaler.state_dict(),
#             }, checkpoint_path)
    
#         # Print progress
#         if debug or epoch % 5 == 0:
#             print(f"GPU {device.index if hasattr(device, 'index') else 'CPU'} | "
#                   f"Time {epoch_time:.1f}s | Best Test Acc: {best_test_acc:.4f}\n"
#                   f"Epoch {epoch:03d}/{epochs} | "
#                   f"Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['acc']:.4f} | "
#                   f"Test Loss: {test_results['loss']:.4f} | Test Acc: {test_results['acc']:.4f}")
    
#         # Early stopping check
#         if early_stopping_counter >= patience:
#             if not debug:
#                 print(f"\nEarly stopping triggered at epoch {epoch}!")
#             break

#     # Update total time
#     metrics['total_time'] = total_time_trained + sum(metrics['epoch_times'])
#     metrics['time_per_epoch'] = np.mean(metrics['epoch_times']) if metrics['epoch_times'] else 0

#     # Load best model and run final inference
#     if not debug:
#         checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         final_results = run_inference(model, test_loader, device, debug=debug)

#         final_metrics = {
#             **metrics,
#             'final_preds': final_results['preds'],
#             'final_scores': final_results['scores'],
#             'final_labels': final_results['labels'],
#             'final_features_i': final_results['features_i'],
#             'final_features_j': final_results['features_j'],
#             'final_neighbor_pairs': final_results['neighbor_pairs']
#         }
#         if 'features_i_unscaled' in final_results:
#             final_metrics['final_features_i_unscaled'] = final_results['features_i_unscaled']
#             final_metrics['final_features_j_unscaled'] = final_results['features_j_unscaled']

#         save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)
#         total_min, total_sec = divmod(metrics['total_time'], 60)
#         print(f"\nTraining complete in {int(total_min)}m {total_sec:.1f}s")
#         print(f"Best model at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")

#     return final_metrics, model, best_model_path

class MultiEdgeClassifier(nn.Module):
    """Graph Neural Network for edge classification"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: torch.device, 
                 num_layers: int = 6, layer_weights: bool = False, softmax: bool = False, 
                 debug: bool = False):
        super().__init__()
        self.device = device
        self.debug = debug
        self.layer_weights_enabled = layer_weights
        self.softmax = softmax
        self.num_layers = num_layers

        # Node embedding
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers with batch normalization
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Final classification layer
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

        # Layer weights (optional)
        if self.layer_weights_enabled:
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
        else:
            self.layer_weights = None

    def forward(self, x_list: List[torch.Tensor], edge_index_list: List[torch.Tensor], 
                edge_index_out_list: List[torch.Tensor], y_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        if self.debug:
            total_start = time.perf_counter()
            timings = {
                "weight_prep": 0.0,
                "move_to_device": 0.0,
                "node_embedding": 0.0,
                "layers": [0.0 for _ in range(self.num_layers)],
                "edge_repr": 0.0,
                "final_fc": 0.0
            }

        all_edge_reprs = []

        # Normalize weights if enabled
        if self.debug:
            t0 = time.perf_counter()
        if self.layer_weights_enabled:
            weights = torch.softmax(self.layer_weights, dim=0) if self.softmax else self.layer_weights
        else:
            weights = None
        if self.debug:
            timings["weight_prep"] += time.perf_counter() - t0

        # Process each graph in the batch
        for x, processed_edges, original_edges in zip(x_list, edge_index_list, edge_index_out_list):
            if self.debug:
                t1 = time.perf_counter()
            x = x.to(self.device, non_blocking=True)
            processed_edges = processed_edges.to(self.device, non_blocking=True)
            if self.debug:
                timings["move_to_device"] += time.perf_counter() - t1

            # Node embedding
            if self.debug:
                t2 = time.perf_counter()
            x_embed = self.node_embedding(x)
            if self.debug:
                timings["node_embedding"] += time.perf_counter() - t2

            # GCN layers with residual connections
            for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
                if self.debug:
                    t_layer = time.perf_counter()
                h = torch.relu(bn(conv(x_embed, processed_edges)))
                if self.layer_weights_enabled:
                    h = weights[i] * h
                x_embed = x_embed + h  # Residual connection
                if self.debug:
                    timings["layers"][i] += time.perf_counter() - t_layer

            # Build edge representations
            if self.debug:
                t3 = time.perf_counter()
            src, dst = original_edges[0], original_edges[1]
            edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
            all_edge_reprs.append(edge_repr)
            if self.debug:
                timings["edge_repr"] += time.perf_counter() - t3

        # Final classification
        if self.debug:
            t4 = time.perf_counter()
        out = self.fc(torch.cat(all_edge_reprs, dim=0))
        if self.debug:
            timings["final_fc"] += time.perf_counter() - t4
            total_time = time.perf_counter() - total_start

            print("\n[Forward Pass Timing Summary]")
            print(f"  Weight preparation: {timings['weight_prep']:.6f} s")
            print(f"  Move to device: {timings['move_to_device']:.6f} s")
            print(f"  Node embedding: {timings['node_embedding']:.6f} s")
            for i, t in enumerate(timings["layers"]):
                print(f"  Layer {i}: {t:.6f} s")
            print(f"  Edge representation build: {timings['edge_repr']:.6f} s")
            print(f"  Final FC: {timings['final_fc']:.6f} s")
            print(f"  TOTAL forward: {total_time:.6f} s\n")

        return out

def train_model(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, scaler: GradScaler, device: torch.device, 
                debug: bool = False) -> Dict[str, float]:
    """
    Optimized training function for single epoch
    """
    model.train()
    total_loss = correct = total = 0
    start_train = time.perf_counter()

    for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list) in enumerate(loader):
        if debug:
            start_batch = time.perf_counter()

        # Move batch to device (optimized with list comprehensions)
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
        edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
        y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

        optimizer.zero_grad(set_to_none=True)  # Faster zero_grad

        # Forward + loss with mixed precision
        with torch.amp.autocast(device_type="cuda"):
            scores = model(x_list, edge_idx_list, edge_idx_out_list)
            loss = criterion(scores, y_batch)

        # Backward + optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        total_loss += loss.item() * len(y_batch)
        preds = scores.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

        if debug and batch_idx % 10 == 0:  # Reduced debug output frequency
            batch_time = time.perf_counter() - start_batch
            print(f"Batch {batch_idx+1}: loss={loss.item():.4f}, time={batch_time:.3f}s")

    epoch_time = time.perf_counter() - start_train
    
    if debug:
        print(f"Training epoch completed in {epoch_time:.2f}s")

    return {
        'loss': total_loss / total if total > 0 else 0,
        'acc': correct / total if total > 0 else 0
    }

def test_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
               device: torch.device, debug: bool = False) -> Dict[str, float]:
    """
    Optimized testing function
    """
    model.eval()
    total_loss = correct = total = 0
    start_test = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list) in enumerate(loader):
            # Move batch to device
            x_list = [x.to(device, non_blocking=True) for x in x_list]
            edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
            edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
            y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

            # Forward + loss with mixed precision
            with torch.amp.autocast(device_type="cuda"):
                scores = model(x_list, edge_idx_list, edge_idx_out_list)
                loss = criterion(scores, y_batch)

            # Metrics
            total_loss += loss.item() * len(y_batch)
            preds = scores.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    test_time = time.perf_counter() - start_test
    
    if debug:
        print(f"Testing completed in {test_time:.2f}s")

    return {
        'loss': total_loss / total if total > 0 else 0,
        'acc': correct / total if total > 0 else 0
    }

@torch.no_grad()
def run_inference_with_generator(model, generator, device, debug=False):
    """
    Inference using the generator, storing both full features and per-event results.
    """
    model.eval()
    all_results = []
    
    for i, (x_scaled, edge_index, edge_index_out, y, x_unscaled) in enumerate(generator):
        if debug and i % 10 == 0:
            print(f"Processing event {i}")
        
        # Move to device
        x_scaled = x_scaled.to(device)
        edge_index = edge_index.to(device)
        edge_index_out = edge_index_out.to(device)
        
        # Forward pass
        out = model([x_scaled], [edge_index], [edge_index_out])
        preds = out.argmax(dim=1).cpu().numpy()
        scores = torch.softmax(out, dim=1).cpu().numpy()
        
        # Edge endpoints
        src_nodes = edge_index_out[0].cpu()
        dst_nodes = edge_index_out[1].cpu()
        
        # Features (scaled + unscaled)
        feats_i_scaled = x_scaled[src_nodes].cpu().numpy()
        feats_j_scaled = x_scaled[dst_nodes].cpu().numpy()
        feats_i_unscaled = x_unscaled[src_nodes].numpy()
        feats_j_unscaled = x_unscaled[dst_nodes].numpy()
        
        # η values from unscaled features
        eta_i = feats_i_unscaled[:, 1]   # assuming column 1 = η
        eta_j = feats_j_unscaled[:, 1]
        
        # Store results for this event
        event_results = {
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
        }
        all_results.append(event_results)
        
        if debug and i >= 4:  # Just process 5 events for debugging
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
    Training loop with per-epoch checkpoints and final best model inference.
    """

    # Make save directory
    if not debug:
        os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, best_model_name)
    best_model_path = os.path.join(save_dir, f"best_{best_model_name}")
    metrics_path = os.path.splitext(model_path)[0] + ".pkl"

    # Initialize training state
    start_epoch = 1
    best_test_acc = 0.0
    best_test_loss = float("inf")
    best_epoch = 0
    total_time_trained = 0.0
    scaler = torch.amp.GradScaler(device="cuda")

    # Resume if checkpoint exists
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
                metrics = {}
        else:
            if not debug:
                print("No previous checkpoint found. Starting from scratch.")
            metrics = {}
    else:
        metrics = {}

    # Initialize metrics
    metrics.update({
        'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [],
        'epoch_times': [], 'best_epoch': best_epoch,
        'best_test_acc': best_test_acc, 'best_test_loss': best_test_loss,
        'total_time': total_time_trained, 'time_per_epoch': 0.0
    })

    # Create DataLoaders once
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
    
    # Epoch loop
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()
    
        # Train and test
        train_results = train_model(model, train_loader, optimizer, criterion, scaler, device, debug=debug)
        test_results = test_model(model, test_loader, criterion, device, debug=debug)
    
        # Metrics update
        epoch_time = time.perf_counter() - epoch_start_time
        metrics['epoch_times'].append(epoch_time)
        metrics['train_loss'].append(train_results['loss'])
        metrics['test_loss'].append(test_results['loss'])
        metrics['train_acc'].append(train_results['acc'])
        metrics['test_acc'].append(test_results['acc'])
    
        # Current test loss
        current_test_loss = test_results['loss']
    
        # Check for real improvement
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            best_test_acc = test_results['acc']
            best_epoch = epoch
            metrics['best_epoch'] = best_epoch
            metrics['best_test_acc'] = best_test_acc
            metrics['best_test_loss'] = best_test_loss
            early_stopping_counter = 0  # reset counter on real improvement
    
            # Save best model
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
            # Increment counter only if improvement < delta
            if current_test_loss > best_test_loss - delta:
                early_stopping_counter += 1
    
        # Save checkpoint every epoch
        if not debug:
            checkpoint_path = os.path.join(save_dir, f"{os.path.splitext(best_model_name)[0]}_epoch{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, checkpoint_path)
    
        # Print progress
        if debug or epoch % 5 == 0:
            print(f"GPU {device.index if hasattr(device, 'index') else 'CPU'} | "
                  f"Time {epoch_time:.1f}s | Best Test Acc: {best_test_acc:.4f}\n"
                  f"Epoch {epoch:03d}/{epochs} | "
                  f"Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['acc']:.4f} | "
                  f"Test Loss: {test_results['loss']:.4f} | Test Acc: {test_results['acc']:.4f}")
    
        # Early stopping check
        if early_stopping_counter >= patience:
            if not debug:
                print(f"\nEarly stopping triggered at epoch {epoch}!")
            break

    # Update total time
    metrics['total_time'] = total_time_trained + sum(metrics['epoch_times'])
    metrics['time_per_epoch'] = np.mean(metrics['epoch_times']) if metrics['epoch_times'] else 0

    # Load best model and run final inference with the new generator
    if not debug:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create a test generator for the new inference function
        test_generator = test_generator_class(**test_generator_kwargs)
        
        # Use the new inference function
        final_results_per_event = run_inference_with_generator(model, test_generator, device, debug=debug)

        # Store the per-event results directly
        final_metrics = {
            **metrics,
            'final_results_per_event': final_results_per_event,  # Store the list of event results
            'num_events_evaluated': len(final_results_per_event)
        }

        # Also store concatenated versions for backward compatibility if needed
        all_preds = np.concatenate([event['preds'] for event in final_results_per_event])
        all_scores = np.concatenate([event['scores'] for event in final_results_per_event])
        all_labels = np.concatenate([event['labels'] for event in final_results_per_event]) if final_results_per_event[0]['labels'] is not None else None
        
        final_metrics.update({
            'final_preds_concatenated': all_preds,
            'final_scores_concatenated': all_scores,
            'final_labels_concatenated': all_labels,
        })

        save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)
        total_min, total_sec = divmod(metrics['total_time'], 60)
        print(f"\nTraining complete in {int(total_min)}m {total_sec:.1f}s")
        print(f"Best model at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")

    return final_metrics, model, best_model_path

def train_single_gpu(gpu_id: int, config: Dict, shared_data: Tuple):
    """
    Train a single model on a specific GPU with custom flags
    """
    try:
        print(f" Starting training on GPU {gpu_id} - {config['model_name']}")
        print(f"   Generator flags: {config['generator_flags']}")
        print(f"   Model flags: {config['model_flags']}")
        
        # Set device + CLEAR MEMORY
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.empty_cache()
        
        # Extract shared data
        scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs = shared_data
        
        # Compute class weights if needed
        if config.get('weighted', False):
            weight_tensor = compute_weight_tensor(labels_for_neighbor_pairs, config['num_classes'], device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"GPU {gpu_id}: Using weighted loss function")
        else:
            criterion = nn.CrossEntropyLoss(weight=None)
            print(f"GPU {gpu_id}: Using standard loss function")
        
        # Model setup with custom flags
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
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        
        # Generator setup with custom flags
        gen_kwargs = {
            'features_dict': scaled_data,
            'neighbor_pairs': neighbor_pairs_list,
            'labels': labels_for_neighbor_pairs,
            'class_counts': config['class_counts'],
            'batch_size': config['batch_size'],
            'unscaled_data_dict': unscaled_data,
            'debug': config.get('debug', False),
            # Add the generator-specific flags
            **config['generator_flags']
        }
        
        train_kwargs = {**gen_kwargs, 'mode': 'train'}
        test_kwargs = {**gen_kwargs, 'mode': 'test'}
        
        # Run training
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
        
        print(f" GPU {gpu_id} training completed!")
        print(f"   Best accuracy: {metrics['best_test_acc']:.4f}")
        print(f"   Total time: {metrics['total_time']:.1f}s")
        print(f"   Model saved to: {model_path}")
        
        return 0  # Success return code
        
    except Exception as e:
        print(f" GPU {gpu_id} training failed: {e}")
        traceback.print_exc()
        return 1  # Failure return code

def main():
    """
    Main function to run multiple models across multiple GPUs
    """
    print("=" * 50)
    print(" Multi-GPU Particle Physics Classification Training")
    print("=" * 50)
    
    # Load shared data once (saves memory and I/O)
    print("\n Loading shared data...")
    load_path = "/storage/mxg1065/datafiles"
    
    try:
        shared_data = load_shared_data(load_path)
        print(" Shared data loaded successfully!")
        
    except Exception as e:
        print(f" Failed to load data: {e}")
        return

    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"\n Available GPUs: {available_gpus}")
    for i in range(available_gpus):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} - {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

    # Configuration for each GPU
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
        
        # Generator flags with default values
        'generator_flags': {
            'padding': False,
            'is_bi_directional': True,
            'with_labels': False,
            'padding_class': 0,
            'train_ratio': 0.7
        },
        
        # Model flags with default values
        'model_flags': {
            'num_layers': 6,
            'layer_weights': False,
            'softmax': False
        }
    }

    # Configurations for each GPU with different flags
    configs = [
        {   # GPU 0
            **base_config,
            'model_name': "padding_with_ll_labeled_cc.pt",
            'description': "Padding with Lone-Lone Pairs but Labeling them as Cluster-Cluster",
            'generator_flags': {**base_config['generator_flags'], 'padding': True}
        },
        {   # GPU 1
            **base_config,
            'model_name': "padding_with_tt_labeled_cc.pt",
            'description': "Padding with True-True Pairs but Labeling them as Cluster-Cluster",
            'generator_flags': {**base_config['generator_flags'], 'padding': True, 'padding_class': 1}
        },
        {   # GPU 2
            **base_config,
            'model_name': "padding_with_ll_labeled_ll.pt",
            'description': "Padding with Lone-Lone Pairs and Labeling them as Lone-Lone",
            'generator_flags': {**base_config['generator_flags'], 'padding': True, 'with_labels': True}
        },
        {   # GPU 3
            **base_config,
            'model_name': "padding_with_tt_labeled_tt.pt",
            'description': "Padding with True-True Pairs and Labeling them as True-True",
            'generator_flags': {**base_config['generator_flags'], 'padding': True, 'padding_class': 1, 'with_labels': True}
        }
    ]

    # Limit to available GPUs
    configs = configs[:available_gpus]
    
    print(f"\n Configuring {len(configs)} models:")
    for i, config in enumerate(configs):
        print(f"   GPU {i}: {config['description']}")
        print(f"      → Model: {config['model_name']}")
        print(f"      → Hidden dim: {config['hidden_dim']}, LR: {config['lr']}")
        print(f"      → Weighted: {config['weighted']}")

    # Ask for confirmation
    print(f"\n  This will start {len(configs)} training processes.")
    response = input("Continue? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return

    # Start training processes
    print(f"\n Starting training processes...")
    processes = []
    start_times = []
    
    for gpu_id, config in enumerate(configs):
        try:
            # 1. Clear GPU cache for THIS specific GPU before starting
            print(f"   Clearing cache for GPU {gpu_id}...")
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                # Optional: Force a synchronization to ensure cleanup is done
                torch.cuda.synchronize()
            
            # 2. Check and report free memory
            free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3  # Free memory in GB
            print(f"   GPU {gpu_id} free memory before start: {free_mem:.2f} GB")
            
            # 3. Start the process
            print(f"   Starting process on GPU {gpu_id}: {config['description']}")
            
            p = mp.Process(
                target=train_single_gpu,
                args=(gpu_id, config, shared_data),
                daemon=False
            )
            p.start()
            processes.append(p)
            start_times.append(time.time())
            
            # 4. Increased stagger time and wait for process to initialize
            if gpu_id < len(configs) - 1:
                time.sleep(25)  # Increase to 25 seconds for more crucial initialization
                    
        except Exception as e:
            print(f" Failed to start process for GPU {gpu_id}: {e}")

    # Monitor processes
    print(f"\n Monitoring {len(processes)} training processes...")
    print("   Press Ctrl+C to stop all processes")
    
    finished = set()
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(30)  # Check every 30 seconds
            
            # Check each process individually
            for i, p in enumerate(processes):
                if not p.is_alive() and i not in finished:
                    exitcode = p.exitcode
                    if exitcode == 0:
                        print(f" GPU {i} finished training successfully.")
                    else:
                        print(f" GPU {i} crashed with exit code {exitcode}.")
                    finished.add(i)

            # Print status update
            alive_count = sum(p.is_alive() for p in processes)
            elapsed = time.time() - min(start_times) if start_times else 0
            print(f"   {alive_count}/{len(processes)} processes still running - Elapsed: {elapsed/60:.1f} min")

        print("\n  All processes finished!")

    except KeyboardInterrupt:
        print(f"\n  Keyboard interrupt received. Stopping all processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        print("All processes terminated.")

    # Wait for all processes to finish
    print("\n Waiting for all processes to complete...")
    for i, p in enumerate(processes):
        p.join()
        print(f"   Process {i} completed with exitcode: {p.exitcode}")

    print("\n" + "=" * 50)
    print(" All training processes completed!")
    print("=" * 50)
    
    # Final summary
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
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Clear CUDA cache before starting
    torch.cuda.empty_cache()
    
    # Add graceful shutdown handling
    def signal_handler(sig, frame):
        print("\n Received shutdown signal. Exiting gracefully...")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main function
    main()
