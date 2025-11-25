# class MultiClassBatchGenerator(IterableDataset):
#     """
#     Optimized IterableDataset for large events (~1.2M edges):
#       • Uses all neighbor pairs per event
#       • Supports unscaled features for analysis
#       • Simplified, memory-efficient design
#     """

#     def __init__(
#         self,
#         features_dict: Dict[int, np.ndarray],
#         neighbor_pairs: np.ndarray,
#         labels: np.ndarray,
#         mode: str = "train",
#         is_bi_directional: bool = True,
#         batch_size: int = 1,
#         train_ratio: float = 0.7,
#         debug: bool = False,
#         unscaled_data_dict: Optional[Dict[int, np.ndarray]] = None,
#     ):
#         # Config
#         self.debug = debug
#         self.is_bi_directional = is_bi_directional
#         self.batch_size = batch_size

#         # Store scaled features as float32 tensors
#         self.features_dict = {k: torch.as_tensor(v, dtype=torch.float32)
#                               for k, v in features_dict.items()}

#         # Optional unscaled features for analysis
#         self.unscaled_features_dict = (
#             self._precompute_unscaled_features(unscaled_data_dict)
#             if unscaled_data_dict is not None else None
#         )

#         # Neighbor pairs and labels
#         self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
#         self.labels = torch.tensor(labels, dtype=torch.long)

#         # Train/test split
#         num_events = len(features_dict)
#         split_idx = int(num_events * train_ratio)
#         self.event_indices = (list(range(split_idx)) if mode == "train"
#                               else list(range(split_idx, num_events)))

#         # Precompute all samples for fast iteration
#         self.precomputed_samples = self._precompute_all_samples()

#         if self.debug:
#             print(f"Initialized with {len(self.precomputed_samples)} samples")
#             print(f"Mode: {mode}, Events: {len(self.event_indices)}")

#     def _precompute_unscaled_features(self, unscaled_data_dict: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
#         """Convert raw detector data to [E, η, φ] format for analysis, keyed by integer event_id."""
#         unscaled_features: Dict[int, torch.Tensor] = {}
#         for event_id, arr in unscaled_data_dict.items():
#             t = torch.as_tensor(arr, dtype=torch.float32)
#             if t.shape[1] == 4:  # (E, px, py, pz) -> (E, η, φ)
#                 eta = t[:, 1]
#                 phi = torch.atan2(t[:, 3], t[:, 2])
#                 t = torch.cat([t[:, 0:1], eta[:, None], phi[:, None]], dim=1)
#             elif t.shape[1] != 3:
#                 raise ValueError(f"Unexpected feature dimension {t.shape[1]} for event {event_id}")
#             unscaled_features[event_id] = t
#         return unscaled_features
    
#     def _precompute_all_samples(self) -> List[Tuple]:
#         """
#         Precompute all event samples for fast iteration.
#         Handle both integer and string keys.
#         """
#         samples: List[Tuple] = []
#         for event_id in self.event_indices:
#             # Try to get the key - handle both integer and string formats
#             event_key = event_id
#             if event_key not in self.features_dict:
#                 # Try string format
#                 event_key = f"data_{event_id}"
#                 if event_key not in self.features_dict:
#                     raise KeyError(f"Event ID {event_id} not found in features_dict")
            
#             # Use all neighbor pairs for this event with random shuffling
#             num_pairs = self.neighbor_pairs.shape[0]
#             shuffled_indices = torch.randperm(num_pairs)  # Random permutation like old generator
            
#             pairs = self.neighbor_pairs[shuffled_indices].T
#             x_scaled = self.features_dict[event_key]
#             x_unscaled = (self.unscaled_features_dict.get(event_key)
#                           if self.unscaled_features_dict else None)
            
#             # FIX: Use event_id for label indexing like old generator
#             out_labels = self.labels[event_id][shuffled_indices]  # Use event_id instead of idx

#             samples.append((x_scaled, pairs, pairs.clone(), out_labels.unsqueeze(1), x_unscaled))
#         return samples

#     # ----- IterableDataset interface ------------------------------------------

#     def __iter__(self):
#         for s in self.precomputed_samples:
#             yield s

#     def __len__(self):
#         return len(self.precomputed_samples)

#     @staticmethod
#     def collate_data(batch: List[Tuple]) -> Tuple:
#         """Combine list of samples into batch format."""
#         x_list = [b[0] for b in batch]
#         edge_index_list = [b[1] for b in batch]
#         edge_index_out_list = [b[2] for b in batch]
#         y_batch = torch.cat([b[3] for b in batch], dim=0)
#         unscaled_list = None if batch[0][4] is None else [b[4] for b in batch]
#         return x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list


# # Model: Graph Neural Network for edge classification
# class MultiEdgeClassifier(nn.Module):
#     """
#     Graph Convolutional Network (GCN) that classifies edges between nodes.
#     Supports optional layer weighting, softmax scaling, and debug timing.
#     """

#     def __init__(self,
#                  input_dim: int,
#                  hidden_dim: int,
#                  output_dim: int,
#                  device: torch.device,
#                  num_layers: int = 6,
#                  layer_weights: bool = False,
#                  softmax: bool = False,
#                  debug: bool = False):
#         super().__init__()
#         self.device = device
#         self.debug = debug
#         self.layer_weights_enabled = layer_weights
#         self.softmax = softmax
#         self.num_layers = num_layers

#         # Initial node embedding layer
#         self.node_embedding = nn.Linear(input_dim, hidden_dim)

#         # Stack of GCN + BatchNorm layers
#         self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
#         self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])

#         # Final edge classification layer
#         self.fc = nn.Linear(2 * hidden_dim, output_dim)

#         # Optional learnable layer weights
#         self.layer_weights = nn.Parameter(torch.ones(num_layers)) if layer_weights else None

#     def forward(self,
#                 x_list: List[torch.Tensor],
#                 edge_index_list: List[torch.Tensor],
#                 edge_index_out_list: List[torch.Tensor],
#                 y_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Forward pass:
#           • Embed nodes
#           • Apply multiple GCN layers with residual connections
#           • Concatenate representations of edge endpoints
#           • Predict edge classes
#         """
#         if self.debug:
#             total_start = time.perf_counter()
#             timings = {"weight_prep": 0.0, "move_to_device": 0.0,
#                        "node_embedding": 0.0,
#                        "layers": [0.0] * self.num_layers,
#                        "edge_repr": 0.0, "final_fc": 0.0}

#         all_edge_reprs = []

#         # Optional layer-weight normalization
#         if self.layer_weights_enabled:
#             weights = (torch.softmax(self.layer_weights, dim=0)
#                        if self.softmax else self.layer_weights)
#         else:
#             weights = None

#         # Process each graph in the batch
#         for x, proc_edges, orig_edges in zip(x_list, edge_index_list, edge_index_out_list):
#             x = x.to(self.device, non_blocking=True)
#             proc_edges = proc_edges.to(self.device, non_blocking=True)

#             # Initial embedding
#             x_embed = self.node_embedding(x)

#             # Apply GCN layers with residual connections
#             for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
#                 h = torch.relu(bn(conv(x_embed, proc_edges)))
#                 if weights is not None:
#                     h = weights[i] * h
#                 x_embed = x_embed + h

#             # Build edge-level representations
#             src, dst = orig_edges[0], orig_edges[1]
#             edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
#             all_edge_reprs.append(edge_repr)

#         # Final classification across all edges in the batch
#         out = self.fc(torch.cat(all_edge_reprs, dim=0))
#         return out


# Standard Library
import argparse
import lzma
import os
import pickle
import re
import signal
import sys
import time
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Libraries
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
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

def load_shared_data(load_path: str) -> Tuple[Dict, Dict, np.ndarray, np.ndarray, List]:
    """
    Load all data files with consistent ordering and cluster information.
    """
    print("Loading shared data with consistent ordering and cluster info...")

    # Load your existing scaled data
    scaled_data = load_pickle(os.path.join(load_path, "scaled_data_1000events.pkl"))
    labels_for_neighbor_pairs = load_pickle(os.path.join(load_path, "labels_ordered.pkl"))
    events_data = load_pickle(os.path.join(load_path, "events_data_complete.pkl"))
    cells_list = load_pickle(os.path.join(load_path, "cells_list_ordered.pkl"))

    # For unscaled_data, create from events_data with consistent ordering
    unscaled_data = {}
    cluster_info = {}  # NEW: Store cluster information per event
    
    for event_idx, event_dict in enumerate(events_data):
        event_key = f"data_{event_idx}"
        valid_indices = event_dict['valid_cell_indices']
        
        # Preserve original order from valid_cell_indices
        snr_raw = np.array(event_dict['cell_SNR'][valid_indices])
        eta = np.array(event_dict['cell_eta'][valid_indices])
        phi = np.array(event_dict['cell_phi'][valid_indices])
        
        unscaled_data[event_key] = np.column_stack([snr_raw, eta, phi]).astype(np.float32)
        
        # NEW: Extract cluster information for this event
        cluster_info[event_key] = {
            'cell_cluster_index': np.array(event_dict['cell_cluster_index'][valid_indices]),
            'cluster_eta': np.array(event_dict['cluster_eta']),
            'cluster_phi': np.array(event_dict['cluster_phi']), 
            'cluster_e': np.array(event_dict['cluster_e']),
            'cluster_to_cell_indices': event_dict['cluster_to_cell_indices'],
            'valid_cell_indices': valid_indices  # Preserve the order
        }

    # CRITICAL FIX: Load the EXACT neighbor_pairs_list that matches the labels
    print("Loading neighbor pairs that match labels shape...")
    try:
        # Try to load the existing neighbor_pairs_list.pkl that was created with your data generation
        neighbor_pairs_list = load_pickle(os.path.join(load_path, "neighbor_pairs_list.pkl"))
        print(f"  Loaded neighbor_pairs_list: {neighbor_pairs_list.shape}")
        
    except FileNotFoundError:
        print("  neighbor_pairs_list.pkl not found, creating from connectivity_list with proper deduplication...")
        
        # Load connectivity_list and apply the SAME deduplication as your original code
        connectivity_list = load_pickle(os.path.join(load_path, "connectivity_list_ordered.pkl"))
        
        # Replicate the exact logic from your data generation code
        corrected_neighbor_pairs = []
        for cell_idx, neighbors in connectivity_list:
            for neighbor_idx in neighbors:
                corrected_neighbor_pairs.append((cell_idx, neighbor_idx))
        
        # Apply the SAME duplicate removal as your original code
        def canonical_form(t):
            return tuple(sorted(t))

        def remove_permutation_variants(tuple_list):
            unique_tuples = set(canonical_form(t) for t in tuple_list)
            return [tuple(sorted(t)) for t in unique_tuples]

        unique_neighbor_pairs = remove_permutation_variants(corrected_neighbor_pairs)
        neighbor_pairs_list = np.array(unique_neighbor_pairs, dtype=np.int32)
        print(f"  Created neighbor_pairs_list: {neighbor_pairs_list.shape}")

    # VERIFY THE CRITICAL MATCH
    expected_pairs = labels_for_neighbor_pairs.shape[1]
    actual_pairs = neighbor_pairs_list.shape[0]
    
    print("Data verification:")
    print(f"  scaled_data events: {len(scaled_data)}")
    print(f"  events_data events: {len(events_data)}") 
    print(f"  labels shape: {labels_for_neighbor_pairs.shape} (expects {expected_pairs} pairs)")
    print(f"  neighbor_pairs_list: {neighbor_pairs_list.shape} (has {actual_pairs} pairs)")
    
    if expected_pairs != actual_pairs:
        print(f"❌ CRITICAL MISMATCH: Labels expect {expected_pairs} pairs but have {actual_pairs} pairs!")
        print(f"   This suggests the data generation and loading logic are inconsistent.")
        raise ValueError(f"Label-pair count mismatch: {expected_pairs} vs {actual_pairs}")
    
    print("✅ All dimensions match correctly!")

    return scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs, cluster_info


def compute_inverse_freq_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Calculate improved per-class weights for severe imbalance."""
    counts = np.bincount(labels.flatten())
    
    # Inverse frequency
    weights = 1.0 / (counts + 1e-5)
    weights = weights / weights.sum() * num_classes
    
    print(f"Class counts: {counts}")
    print(f"Computed weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_focal_weights(labels: np.ndarray, num_classes: int, device: torch.device, 
                         alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss inspired weighting - focuses on hard examples."""
    counts = np.bincount(labels.flatten())
    total = len(labels)
    
    # Base weights from class frequency
    class_weights = (total - counts) / (counts + 1e-5)
    
    # Apply focal scaling
    weights = alpha * class_weights ** gamma
    weights = weights / weights.sum() * num_classes
    
    print(f"Focal weights - alpha: {alpha}, gamma: {gamma}")
    print(f"Class counts: {counts}")
    print(f"Computed weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32, device=device)

def compute_log_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Logarithmic weighting - less extreme than inverse frequency."""
    counts = np.bincount(labels.flatten())
    
    # Logarithmic scaling
    weights = 1.0 / np.log1p(counts + 1e-5)  # log1p = log(1 + x)
    weights = weights / weights.sum() * num_classes
    
    print(f"Logarithmic weights")
    print(f"Class counts: {counts}")
    print(f"Computed weights: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32, device=device)

def compute_manual_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    """Manual weights prioritizing important but rare classes."""
    # Based on your accuracy targets and class importance
    manual_weights = np.array([
        0.1,    # Lone-Lone (class 0) - strongly downweight
        10.0,   # True-True (class 1) - highly prioritize
        8.0,    # Cluster-Lone (class 2)
        8.0,    # Lone-Cluster (class 3)  
        15.0    # Cluster-Cluster (class 4) - most rare, highest weight
    ])
    
    # Normalize
    weights = manual_weights / manual_weights.sum() * num_classes
    
    print(f"Manual strategic weights")
    print(f"Computed weights: {weights}")
    
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

# --- DIRECT PARQUET SAVING FUNCTIONS (ENHANCED) --------------------------------

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            max_val, min_val = df[col].max(), df[col].min()
            if min_val >= 0:
                if max_val < 256:
                    df[col] = df[col].astype('uint8')
                elif max_val < 65536:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('uint32')
            else:
                if min_val > -128 and max_val < 127:
                    df[col] = df[col].astype('int8')
                elif min_val > -32768 and max_val < 32767:
                    df[col] = df[col].astype('int16')
                else:
                    df[col] = df[col].astype('int32')
    return df

def _process_single_event_comprehensive(event_result: Dict, model_name: str = None, 
                                      cluster_info: Dict = None) -> pd.DataFrame:
    """
    Process a single event into a comprehensive DataFrame with derived features.
    Now includes cluster information and preserves input order.
    """
    event_id = event_result['event_id']
    preds = np.array(event_result['preds'], dtype=np.int8)
    scores = np.array(event_result['scores'], dtype=np.float32)
    labels = np.array(event_result['labels'], dtype=np.int8) if event_result['labels'] is not None else None
    neighbor_pairs = np.array(event_result['neighbor_pairs'], dtype=np.int32)
    features_i = np.array(event_result['features_i_unscaled'], dtype=np.float32)
    features_j = np.array(event_result['features_j_unscaled'], dtype=np.float32)
    
    num_edges = len(preds)

    # --- Extract base features ---
    snr_i, eta_i, phi_i = features_i[:, 0], features_i[:, 1], features_i[:, 2]
    snr_j, eta_j, phi_j = features_j[:, 0], features_j[:, 1], features_j[:, 2]

    # --- NEW: Extract cluster information ---
    cluster_index_i = np.full(num_edges, -1, dtype=np.int16)
    cluster_index_j = np.full(num_edges, -1, dtype=np.int16)
    cluster_eta_i = np.full(num_edges, -999.0, dtype=np.float32)
    cluster_eta_j = np.full(num_edges, -999.0, dtype=np.float32)
    cluster_phi_i = np.full(num_edges, -999.0, dtype=np.float32)
    cluster_phi_j = np.full(num_edges, -999.0, dtype=np.float32)
    cluster_e_i = np.full(num_edges, -1.0, dtype=np.float32)
    cluster_e_j = np.full(num_edges, -1.0, dtype=np.float32)
    
    if cluster_info is not None:
        event_key = f"data_{event_id}"
        if event_key in cluster_info:
            cell_cluster_index = cluster_info[event_key]['cell_cluster_index']
            cluster_eta = cluster_info[event_key]['cluster_eta']
            cluster_phi = cluster_info[event_key]['cluster_phi']
            cluster_e = cluster_info[event_key]['cluster_e']
            
            # Map source cells to their clusters
            valid_src_indices = neighbor_pairs[:, 0] < len(cell_cluster_index)
            src_cluster_indices = cell_cluster_index[neighbor_pairs[valid_src_indices, 0]]
            cluster_index_i[valid_src_indices] = src_cluster_indices
            
            # Map target cells to their clusters  
            valid_dst_indices = neighbor_pairs[:, 1] < len(cell_cluster_index)
            dst_cluster_indices = cell_cluster_index[neighbor_pairs[valid_dst_indices, 1]]
            cluster_index_j[valid_dst_indices] = dst_cluster_indices
            
            # Get cluster properties for source cells
            valid_cluster_src = (src_cluster_indices >= 0) & (src_cluster_indices < len(cluster_eta))
            valid_indices_src = valid_src_indices[valid_cluster_src]
            cluster_eta_i[valid_indices_src] = cluster_eta[src_cluster_indices[valid_cluster_src]]
            cluster_phi_i[valid_indices_src] = cluster_phi[src_cluster_indices[valid_cluster_src]]
            cluster_e_i[valid_indices_src] = cluster_e[src_cluster_indices[valid_cluster_src]]
            
            # Get cluster properties for target cells
            valid_cluster_dst = (dst_cluster_indices >= 0) & (dst_cluster_indices < len(cluster_eta))
            valid_indices_dst = valid_dst_indices[valid_cluster_dst]
            cluster_eta_j[valid_indices_dst] = cluster_eta[dst_cluster_indices[valid_cluster_dst]]
            cluster_phi_j[valid_indices_dst] = cluster_phi[dst_cluster_indices[valid_cluster_dst]]
            cluster_e_j[valid_indices_dst] = cluster_e[dst_cluster_indices[valid_cluster_dst]]

    # --- Derived features ---
    delta_eta = np.abs(eta_i - eta_j).astype(np.float32)
    delta_phi = np.abs(phi_i - phi_j).astype(np.float32)
    delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi).astype(np.float32)
    spatial_distance = np.sqrt(delta_eta**2 + delta_phi**2).astype(np.float32)
    snr_ratio = np.divide(snr_i, snr_j, out=np.full_like(snr_i, np.inf, dtype=np.float32), where=snr_j != 0)
    avg_snr = ((snr_i + snr_j) / 2).astype(np.float32)
    snr_sum = (snr_i + snr_j).astype(np.float32)
    snr_product = (snr_i * snr_j).astype(np.float32)
    
    # --- NEW: Cluster-based features ---
    same_cluster = (cluster_index_i == cluster_index_j) & (cluster_index_i >= 0)
    cluster_delta_eta = np.abs(cluster_eta_i - cluster_eta_j).astype(np.float32)
    cluster_delta_phi = np.abs(cluster_phi_i - cluster_phi_j).astype(np.float32)
    cluster_delta_phi = np.minimum(cluster_delta_phi, 2*np.pi - cluster_delta_phi).astype(np.float32)
    cluster_spatial_distance = np.sqrt(cluster_delta_eta**2 + cluster_delta_phi**2).astype(np.float32)

    # --- Confidence & predictions ---
    confidence = scores[np.arange(num_edges), preds].astype(np.float32)
    confidence_classes = [scores[:, i].astype(np.float32) for i in range(scores.shape[1])]
    true_labels = labels if labels is not None else np.full(num_edges, -1, dtype=np.int8)
    is_correct = np.where(true_labels != -1, preds == true_labels, None)

    # --- Prediction uncertainty measures ---
    sorted_scores = np.sort(scores, axis=1)
    confidence_margin = (sorted_scores[:, -1] - sorted_scores[:, -2]).astype(np.float32)
    entropy = (-np.sum(scores * np.log(scores + 1e-8), axis=1)).astype(np.float32)

    # --- Build comprehensive event DataFrame ---
    df_event = pd.DataFrame({
        'event_id': np.full(num_edges, event_id, dtype=np.uint16),
        'edge_id': np.arange(num_edges, dtype=np.uint32),
        'source_id': neighbor_pairs[:, 0].astype(np.uint32),
        'target_id': neighbor_pairs[:, 1].astype(np.uint32),
        'true_label': true_labels,
        'pred_label': preds,
        'is_correct': is_correct.astype(bool) if is_correct is not None else np.full(num_edges, False, dtype=bool),
        'confidence': confidence,
        'confidence_margin': confidence_margin,
        'prediction_entropy': entropy,
        'confidence_class_0': confidence_classes[0],
        'confidence_class_1': confidence_classes[1],
        'confidence_class_2': confidence_classes[2],
        'confidence_class_3': confidence_classes[3],
        'confidence_class_4': confidence_classes[4],
        'snr_source': snr_i.astype(np.float32),
        'eta_source': eta_i.astype(np.float32),
        'phi_source': phi_i.astype(np.float32),
        'snr_target': snr_j.astype(np.float32),
        'eta_target': eta_j.astype(np.float32),
        'phi_target': phi_j.astype(np.float32),
        'delta_eta': delta_eta,
        'delta_phi': delta_phi,
        'spatial_distance': spatial_distance,
        'snr_ratio': snr_ratio,
        'avg_snr': avg_snr,
        'snr_sum': snr_sum,
        'snr_product': snr_product,
        'event_size': np.full(num_edges, num_edges, dtype=np.uint32),
        
        # --- NEW: Cluster information ---
        'cluster_index_source': cluster_index_i,
        'cluster_index_target': cluster_index_j,
        'cluster_eta_source': cluster_eta_i,
        'cluster_eta_target': cluster_eta_j, 
        'cluster_phi_source': cluster_phi_i,
        'cluster_phi_target': cluster_phi_j,
        'cluster_e_source': cluster_e_i,
        'cluster_e_target': cluster_e_j,
        'same_cluster': same_cluster.astype(bool),
        'cluster_delta_eta': cluster_delta_eta,
        'cluster_delta_phi': cluster_delta_phi,
        'cluster_spatial_distance': cluster_spatial_distance,
    })

    # Add model name if provided
    if model_name:
        df_event['model_name'] = model_name

    # --- Add class flags for easy filtering ---
    for i in range(5):
        df_event[f'is_class_{i}'] = (df_event['true_label'] == i).astype(bool)
        df_event[f'pred_is_class_{i}'] = (df_event['pred_label'] == i).astype(bool)
        df_event[f'correct_class_{i}'] = ((df_event['true_label'] == i) & (df_event['is_correct'] == True)).astype(bool)

    return df_event

def save_inference_results_directly(final_results_per_event: List[Dict], save_dir: str, 
                                  model_name: str, cluster_info: Dict = None, chunk_size: int = 50) -> str:
    """
    Save inference results directly to Parquet without PKL intermediate.
    Now includes cluster information and preserves input order.
    """
    print(f"💾 Saving results directly to Parquet for {model_name}...")
    print(f"   Processing {len(final_results_per_event)} events in chunks of {chunk_size}")
    
    os.makedirs(save_dir, exist_ok=True)
    temp_chunk_files = []
    
    # Process events in chunks
    for start_idx in range(0, len(final_results_per_event), chunk_size):
        end_idx = min(start_idx + chunk_size, len(final_results_per_event))
        chunk_events = final_results_per_event[start_idx:end_idx]
        
        if start_idx % 100 == 0:  # Progress reporting
            print(f"   Processing chunk {start_idx}-{end_idx}...")
        
        chunk_dfs = []
        
        # Process each event in this chunk
        for event_result in chunk_events:
            df_event = _process_single_event_comprehensive(
                event_result, model_name, cluster_info
            )
            chunk_dfs.append(df_event)
        
        # Combine chunk and optimize memory
        if chunk_dfs:
            df_chunk = pd.concat(chunk_dfs, ignore_index=True)
            df_chunk = optimize_dataframe_memory(df_chunk)
            
            # Save chunk to temporary file
            temp_file = os.path.join(save_dir, f"temp_{model_name}_chunk_{start_idx:06d}.parquet")
            df_chunk.to_parquet(temp_file, index=False)
            temp_chunk_files.append(temp_file)
            
            print(f"     ✅ Saved chunk: {len(df_chunk):,} rows")
            
            # Clean up memory
            del df_chunk, chunk_dfs
            import gc
            gc.collect()
    
    # Combine all chunks efficiently
    print("   Combining all chunks...")
    final_dfs = []
    
    for i, temp_file in enumerate(temp_chunk_files):
        if i % 10 == 0:
            print(f"     Reading chunk {i+1}/{len(temp_chunk_files)}...")
        
        df_chunk = pd.read_parquet(temp_file)
        final_dfs.append(df_chunk)
        
        # Clean up temp file
        os.remove(temp_file)
    
    # Final combination and save
    print("   Final concatenation...")
    df_final = pd.concat(final_dfs, ignore_index=True)
    
    # Save final comprehensive file
    parquet_path = os.path.join(save_dir, f"comprehensive_results_{model_name}.parquet")
    df_final.to_parquet(parquet_path, index=False)
    
    print(f"💾 Direct save complete: {parquet_path}")
    print(f"   - Total rows: {len(df_final):,}")
    print(f"   - Total events: {len(final_results_per_event)}")
    print(f"   - File size: {os.path.getsize(parquet_path) / 1024**2:.1f} MB")
    print(f"   - Columns: {len(df_final.columns)} (including cluster information)")
    
    return parquet_path

class MultiClassBatchGenerator(IterableDataset):
    """
    Optimized IterableDataset for large events (~1.2M edges):
      • Uses all neighbor pairs per event
      • Supports unscaled features for analysis
      • FIXED: First 70% training, last 30% testing with consistent ordering
    """

    def __init__(
        self,
        features_dict: Dict[int, np.ndarray],
        neighbor_pairs: np.ndarray,
        labels: np.ndarray,
        mode: str = "train",
        is_bi_directional: bool = True,
        batch_size: int = 1,
        train_ratio: float = 0.7,  # FIXED: Consistent 70/30 split
        debug: bool = False,
        unscaled_data_dict: Optional[Dict[int, np.ndarray]] = None,
        cluster_info_dict: Optional[Dict[int, Dict]] = None,  # NEW: Cluster information
    ):
        # Config
        self.debug = debug
        self.is_bi_directional = is_bi_directional
        self.batch_size = batch_size
        self.cluster_info_dict = cluster_info_dict  # NEW: Store cluster info

        # Store scaled features as float32 tensors
        self.features_dict = {k: torch.as_tensor(v, dtype=torch.float32)
                              for k, v in features_dict.items()}

        # Optional unscaled features for analysis
        self.unscaled_features_dict = (
            self._precompute_unscaled_features(unscaled_data_dict)
            if unscaled_data_dict is not None else None
        )

        # Neighbor pairs and labels
        self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

        # FIXED: Consistent train/test split - first 70% training, last 30% testing
        num_events = len(features_dict)
        split_idx = int(num_events * train_ratio)
        
        if mode == "train":
            self.event_indices = list(range(0, split_idx))  # First 70%
        else:  # "test"
            self.event_indices = list(range(split_idx, num_events))  # Last 30%
        
        print(f"📊 {mode.upper()} SET: Events {self.event_indices[0]}-{self.event_indices[-1]} "
              f"({len(self.event_indices)} events)")

        # Precompute all samples for fast iteration
        self.precomputed_samples = self._precompute_all_samples()

        if self.debug:
            print(f"Initialized with {len(self.precomputed_samples)} samples")
            print(f"Mode: {mode}, Events: {len(self.event_indices)}")

    def _precompute_unscaled_features(self, unscaled_data_dict: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        """Convert raw detector data to [E, η, φ] format for analysis, keyed by integer event_id."""
        unscaled_features: Dict[int, torch.Tensor] = {}
        for event_id, arr in unscaled_data_dict.items():
            t = torch.as_tensor(arr, dtype=torch.float32)
            if t.shape[1] == 4:  # (E, px, py, pz) -> (E, η, φ)
                eta = t[:, 1]
                phi = torch.atan2(t[:, 3], t[:, 2])
                t = torch.cat([t[:, 0:1], eta[:, None], phi[:, None]], dim=1)
            elif t.shape[1] != 3:
                raise ValueError(f"Unexpected feature dimension {t.shape[1]} for event {event_id}")
            unscaled_features[event_id] = t
        return unscaled_features
    
    def _precompute_all_samples(self) -> List[Tuple]:
        """
        Precompute all event samples for fast iteration.
        Handle both integer and string keys with consistent ordering.
        """
        samples: List[Tuple] = []
        for event_id in self.event_indices:
            # Try to get the key - handle both integer and string formats
            event_key = event_id
            if event_key not in self.features_dict:
                # Try string format
                event_key = f"data_{event_id}"
                if event_key not in self.features_dict:
                    raise KeyError(f"Event ID {event_id} not found in features_dict")
            
            # NEW: Get cluster information for this event if available
            cluster_info = None
            if self.cluster_info_dict and event_key in self.cluster_info_dict:
                cluster_info = self.cluster_info_dict[event_key]
            
            # Use all neighbor pairs for this event with consistent ordering
            # REMOVED: Random shuffling to preserve input order
            pairs = self.neighbor_pairs.T  # Keep original order
            x_scaled = self.features_dict[event_key]
            x_unscaled = (self.unscaled_features_dict.get(event_key)
                          if self.unscaled_features_dict else None)
            
            # FIX: Use event_id for label indexing with consistent order
            out_labels = self.labels[event_id]  # Remove shuffling to preserve order

            samples.append((x_scaled, pairs, pairs.clone(), out_labels.unsqueeze(1), x_unscaled, cluster_info))
        return samples

    # ----- IterableDataset interface ------------------------------------------

    def __iter__(self):
        for s in self.precomputed_samples:
            yield s

    def __len__(self):
        return len(self.precomputed_samples)

    @staticmethod
    def collate_data(batch: List[Tuple]) -> Tuple:
        """Combine list of samples into batch format."""
        x_list = [b[0] for b in batch]
        edge_index_list = [b[1] for b in batch]
        edge_index_out_list = [b[2] for b in batch]
        y_batch = torch.cat([b[3] for b in batch], dim=0)
        unscaled_list = None if batch[0][4] is None else [b[4] for b in batch]
        cluster_info_list = None if batch[0][5] is None else [b[5] for b in batch]  # NEW: Cluster info
        return x_list, edge_index_list, edge_index_out_list, y_batch, unscaled_list, cluster_info_list


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

    for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list, cluster_info_list) in enumerate(loader):
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
        for x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list, cluster_info_list in loader:
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
    Now preserves input order and includes cluster information.
    """
    model.eval()
    all_results = []

    for i, (x_scaled, edge_index, edge_index_out, y, x_unscaled, cluster_info) in enumerate(generator):
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

        # Extract endpoints of each predicted edge (preserving order)
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

        # Store all inference results for this event with cluster info
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
            "cluster_info": cluster_info,  # NEW: Include cluster information
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
              cluster_info: Optional[Dict] = None,  # NEW: Cluster information
              lr: float = 1e-3, resume: bool = True, patience: int = 10,
              delta: float = 0.0001, debug: bool = False,
              # NEW: Direct saving parameters
              save_direct_to_parquet: bool = True) -> Tuple[Dict, nn.Module, str]:
    """
    Main training loop with consistent ordering and cluster information.
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
            # Increment counter if loss hasn't improved by `delta`
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

        # NEW: Save directly to Parquet with cluster information
        if save_direct_to_parquet:
            try:
                print(f"\n{'='*50}")
                print("SAVING RESULTS DIRECTLY TO PARQUET...")
                print(f"{'='*50}")
                
                # Extract model name from best_model_name
                model_base_name = os.path.splitext(best_model_name)[0]
                
                # Save comprehensive results directly to Parquet with cluster info
                parquet_path = save_inference_results_directly(
                    final_results_per_event, save_dir, model_base_name, cluster_info
                )
                
                # Save lightweight training metrics as small PKL
                training_metrics = {
                    'train_loss': final_metrics['train_loss'],
                    'test_loss': final_metrics['test_loss'],
                    'train_acc': final_metrics['train_acc'], 
                    'test_acc': final_metrics['test_acc'],
                    'best_test_acc': final_metrics['best_test_acc'],
                    'best_test_loss': final_metrics['best_test_loss'],
                    'best_epoch': final_metrics['best_epoch'],
                    'total_time': final_metrics['total_time'],
                    'time_per_epoch': final_metrics['time_per_epoch'],
                    'parquet_results_path': parquet_path,
                    'train_events': train_generator_kwargs.get('event_indices', '0-699'),  # Track split
                    'test_events': test_generator_kwargs.get('event_indices', '700-999')
                }
                save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), training_metrics)
                
                print("✅ Results saved directly to Parquet!")
                print(f"   📊 Training metrics: {metrics_path}")
                print(f"   📁 Analysis results: {parquet_path}")
                print(f"   🎯 Train events: {training_metrics['train_events']}")
                print(f"   🧪 Test events: {training_metrics['test_events']}")
                
            except Exception as e:
                print(f"⚠ Direct Parquet saving failed: {e}")
                # Fallback: save the old way
                print("🔄 Falling back to PKL saving...")
                save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)
        else:
            # Original PKL saving (for compatibility)
            save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)

        total_min, total_sec = divmod(metrics['total_time'], 60)
        print(f"\nTraining complete in {int(total_min)}m {total_sec:.1f}s")
        print(f"Best model at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")

    return final_metrics, model, best_model_path

def train_single_gpu(gpu_id: int, config: Dict, shared_data: Tuple):
    """
    Train a single model on one GPU with the given configuration and shared data.
    Now includes cluster information and consistent ordering.
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
        scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs, cluster_info = shared_data
        
        # Create the loss function, optionally using class weights to handle imbalance
        if config.get('weighted', False):
            weight_strategy = config.get('weight_strategy', 'inverse')
            
            if weight_strategy == 'focal':
                weight_tensor = compute_focal_weights(
                    labels_for_neighbor_pairs, config['num_classes'], device,
                    alpha=config.get('focal_alpha', 0.25),
                    gamma=config.get('focal_gamma', 2.0)
                )
            elif weight_strategy == 'logarithmic':
                weight_tensor = compute_log_weights(
                    labels_for_neighbor_pairs, config['num_classes'], device
                )
            elif weight_strategy == 'manual':
                weight_tensor = compute_manual_weights(
                    labels_for_neighbor_pairs, config['num_classes'], device
                )
            else:  # default inverse frequency
                weight_tensor = compute_inverse_freq_weights(
                    labels_for_neighbor_pairs, config['num_classes'], device
                )
                
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"GPU {gpu_id}: Using {weight_strategy} weighted loss")
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
            'batch_size': config['batch_size'],
            'unscaled_data_dict': unscaled_data,
            'cluster_info_dict': cluster_info,  # NEW: Include cluster information
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
            cluster_info=cluster_info,  # NEW: Pass cluster information
            debug=config.get('debug', False),
            resume=config.get('resume', True),
            patience=config.get('patience', 10),
            delta=config.get('delta', 0.0001),
            # NEW: Use direct Parquet saving instead of conversion
            save_direct_to_parquet=True  # Always save directly to Parquet
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
        # NEW: Load cluster information along with other data
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
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'epochs': 200,
        'batch_size': 1,
        'save_dir': "/storage/mxg1065/fixed_batch_size_models",
        'resume': True,
        'patience': 20,
        'delta': 0.0001,
        'debug': False,
        'weighted': False,
        'generator_flags': {  # defaults for the data generator
            'is_bi_directional': True,
            'train_ratio': 0.7  # FIXED: Consistent 70/30 split
        },
        'model_flags': {     # defaults for the model
            'num_layers': 6,
            'layer_weights': False,
            'softmax': False
        }
    }

    # Define per-GPU configurations with small variations
    configs = [
        # GPU 0
        {**base_config,
        'model_name': "test_model.pt",
        'description': "test model",
        'batch_size': 1}
        #,
        
        # # GPU 1  
        # {**base_config,
        #  'model_name': "nine_layer_model.pt", 
        #  'description': "Model with nine layers",
        #  'model_flags': {**base_config['model_flags'], 'num_layers': 9}},
        
        # # GPU 2
        #{**base_config,
        # 'model_name': "twelve_layer_model.pt", 
        # 'description': "Model with twelve layers",
        # 'model_flags': {**base_config['model_flags'], 'num_layers': 12}},
        
       # # GPU 3
       # {**base_config,
       #  'model_name': "fifteen_layer_model.pt", 
       #  'description': "Model with fifteen layers",
       #  'model_flags': {**base_config['model_flags'], 'num_layers': 15}}
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
        print(f"      → Train/Test: 70%/30% (events 0-699/700-999)")

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
                parquet_path = metrics.get('parquet_results_path', 'Not saved')
                train_events = metrics.get('train_events', 'Unknown')
                test_events = metrics.get('test_events', 'Unknown')
                print(f"   GPU {i}: {config['description']}")
                print(f"      Best Accuracy: {best_acc:.4f}")
                print(f"      Total Time: {int(min_time)}m {sec_time:.1f}s")
                print(f"      Train Events: {train_events}")
                print(f"      Test Events: {test_events}")
                print(f"      Model: {model_path}")
                print(f"      Results: {parquet_path}")
            except:
                print(f"   GPU {i}: Could not load metrics")
        else:
            print(f"   GPU {i}: No results found")

    print("\n Multi-GPU training completed successfully!")

# ... (keep the existing test_data_integrity, diagnose_connectivity_issue, quick_data_check functions)

if __name__ == "__main__":
    # Ensure correct multiprocessing start method for CUDA
    mp.set_start_method('spawn', force=True)
    
    # Free any cached GPU memory before starting
    torch.cuda.empty_cache()
    
    # Handle graceful shutdown on SIGINT/SIGTERM
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal. Exiting gracefully...")
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # REMOVED: Conversion mode (no longer needed)
    
    # Add quick data check before starting training
    print("🚀 Initializing Multi-GPU Training System")
    print("=" * 50)
    
    load_path = "/storage/mxg1065/datafiles"
    
    # Step 1: Quick file existence check
    print("\n📋 Step 1: Checking data files...")
    files_ok = quick_data_check(load_path)
    
    if not files_ok:
        print("❌ Critical: Required data files missing or corrupted!")
        print("   Please ensure all data files are generated before training.")
        sys.exit(1)
    
    # Step 2: Comprehensive data integrity test
    print("\n🔍 Step 2: Running data integrity tests...")
    try:
        # Load data for integrity testing
        scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs, cluster_info = load_shared_data(load_path)
        
        # FIRST: Run diagnosis to see the exact issue (NOW INSIDE THE TRY BLOCK)
        diagnose_connectivity_issue(load_path, cell_index=1)
        
        # THEN: Run comprehensive integrity tests
        integrity_ok = test_data_integrity(
            load_path, scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs
        )
        
        if not integrity_ok:
            print("❌ Data integrity tests failed!")
            response = input("Continue training anyway? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Training aborted by user.")
                sys.exit(1)
            else:
                print("⚠️  Continuing training with potential data issues...")
        else:
            print("✅ Data integrity verified!")
            
    except Exception as e:
        print(f"❌ Error during data integrity testing: {e}")
        response = input("Continue training anyway? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training aborted by user.")
            sys.exit(1)
        else:
            print("⚠️  Continuing training despite testing errors...")
    
    # Step 3: Start the main training workflow
    print("\n🎯 Step 3: Starting multi-GPU training...")
    print("=" * 50)
    main()