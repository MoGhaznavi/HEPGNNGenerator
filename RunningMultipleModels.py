'''
For training with auto-conversion:
python3 RunningMultipleModels.py

For converting existing results
python3 RunningMultipleModels.py convert path/to/results.pkl
'''

# Standard Library
import argparse
import datetime
import gc
import glob
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

# Parquet support
import pyarrow as pa
import pyarrow.parquet as pq

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


def load_shared_data(data_dir: str):
    """
    Load dataset from the 4-file format we created.
    CORRECTED version that matches our dataset structure.
    """
    import h5py
    
    print("📥 Loading dataset from:", data_dir)
    
    # --- 1️⃣ Cells (File 4) ---
    cells_path = os.path.join(data_dir, "cells.npy")
    cells = np.load(cells_path)
    num_cells = cells.shape[0]
    print(f"  ✓ Loaded {num_cells} cells from {cells_path}")
    
    # --- 2️⃣ Pairs/Connectivity (File 1) ---
    pairs_path = os.path.join(data_dir, "pairs.npy")
    neighbor_pairs_array = np.load(pairs_path).astype(np.int32)
    num_edges = neighbor_pairs_array.shape[0]
    print(f"  ✓ Loaded {num_edges} edges from {pairs_path}")
    
    # --- 3️⃣ Events HDF5 (File 2) ---
    events_h5_path = os.path.join(data_dir, "events.h5")
    print(f"  Loading HDF5: {events_h5_path}")
    
    scaled_data_dict: Dict[int, np.ndarray] = {}
    unscaled_data_dict: Dict[int, np.ndarray] = {}
    cluster_info_dict: Dict[int, Dict[str, np.ndarray]] = {}
    
    with h5py.File(events_h5_path, 'r') as h5f:
        num_events = h5f.attrs['num_events']
        print(f"  Found {num_events} events")
        
        # CRITICAL: Load datasets that our code created
        # 1. Scaled SNR
        if 'cell/cell_SNR_scaled' in h5f:
            snr_scaled_all = h5f['cell/cell_SNR_scaled'][:]  # (1000, 187650)
            print(f"  ✓ SNR scaled: {snr_scaled_all.shape}")
        else:
            raise ValueError("Missing cell_SNR_scaled - dataset not built correctly")
        
        # 2. Raw SNR  
        if 'cell/cell_SNR_raw' in h5f:
            snr_raw_all = h5f['cell/cell_SNR_raw'][:]  # (1000, 187650)
            print(f"  ✓ SNR raw: {snr_raw_all.shape}")
        else:
            # If raw not found, use scaled as raw
            snr_raw_all = snr_scaled_all.copy()
            print("  ⚠️  Using scaled as raw (no raw found)")
        
        # 3. Eta and Phi - CORRECTED: Our code saves them as cell/cell_eta, cell/cell_phi
        if 'cell/cell_eta' in h5f and 'cell/cell_phi' in h5f:
            eta_all = h5f['cell/cell_eta'][:]  # (1000, 187650) - event 0 values repeated
            phi_all = h5f['cell/cell_phi'][:]  # (1000, 187650) - event 0 values repeated
            print(f"  ✓ Eta: {eta_all.shape}, Phi: {phi_all.shape}")
        else:
            # Fallback to cells.npy
            eta_all = np.tile(cells['eta_event0'].astype(np.float32), (num_events, 1))
            phi_all = np.tile(cells['phi_event0'].astype(np.float32), (num_events, 1))
            print("  ⚠️  Using eta/phi from cells.npy")
        
        # 4. Cluster indices - CORRECTED: Our code saves them as cell/cell_cluster_index
        if 'cell/cell_cluster_index' in h5f:
            cluster_idx_all = h5f['cell/cell_cluster_index'][:]  # (1000, 187650)
            print(f"  ✓ Cluster indices: {cluster_idx_all.shape}")
        else:
            # Check structured clusters
            if 'clusters_structured' in h5f:
                print("  ℹ️  Found structured clusters (per-cluster data)")
                cluster_idx_all = np.zeros((num_events, num_cells), dtype=np.int32)
            else:
                cluster_idx_all = np.zeros((num_events, num_cells), dtype=np.int32)
                print("  ⚠️  No cluster indices found, using zeros")
        
        # Build feature matrices for each event
        print(f"  Building features for {num_events} events...")
        for ev in range(num_events):
            # SCALED features: [snr_scaled, eta, phi]
            features_scaled = np.stack([
                snr_scaled_all[ev],  # Already scaled by RobustScaler
                eta_all[ev],         # Original eta (not scaled)
                phi_all[ev]          # Original phi (not scaled)
            ], axis=1).astype(np.float32)
            scaled_data_dict[ev] = features_scaled
            
            # UNSCALED/RAW features: [snr_raw, eta, phi]
            features_unscaled = np.stack([
                snr_raw_all[ev],     # Original SNR values
                eta_all[ev],         # Original eta
                phi_all[ev]          # Original phi
            ], axis=1).astype(np.float32)
            unscaled_data_dict[ev] = features_unscaled
            
            # Cluster information
            cluster_info_dict[ev] = {
                'cell_cluster_index': cluster_idx_all[ev].astype(np.int32)
            }
            
            # Progress
            if ev % 100 == 0 and ev > 0:
                print(f"    Processed {ev}/{num_events} events")
    
    # --- 4️⃣ Labels (File 3) ---
    labels_path = os.path.join(data_dir, "labels.npy")
    labels_array = np.load(labels_path).astype(np.int8)
    print(f"  ✓ Labels: {labels_array.shape}")
    
    # Verify everything matches
    print("\n✅ DATASET VERIFICATION:")
    print(f"   Events: {num_events} (expected: 1000)")
    print(f"   Cells: {num_cells} (expected: 187650)")
    print(f"   Edges: {num_edges} (expected: 1,250,242)")
    print(f"   Labels shape: {labels_array.shape} (events × edges)")
    print(f"   Feature shape per event: {list(scaled_data_dict[0].shape)} (cells × 3)")
    
    # Quick sanity check
    assert num_events == 1000, f"Expected 1000 events, got {num_events}"
    assert num_cells == 187650, f"Expected 187650 cells, got {num_cells}"
    assert labels_array.shape == (num_events, num_edges), \
        f"Labels shape mismatch: {labels_array.shape} != ({num_events}, {num_edges})"
    
    print("\n🎉 Dataset loaded successfully and verified!")
    
    return scaled_data_dict, unscaled_data_dict, neighbor_pairs_array, labels_array, cluster_info_dict


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


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types."""
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type == 'float64':
            # Downcast float64 to float32
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif col_type == 'int64':
            # Downcast int64 based on range
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= 0:
                if c_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif c_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif c_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
                else:
                    df[col] = df[col].astype(np.uint64)
            else:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
    
    return df


def _process_single_event_comprehensive(event_result: Dict, model_name: str = None, 
                                      cluster_info_dict: Dict = None,
                                      events_h5_path: str = None) -> pd.DataFrame:
    """
    Process a single event into a comprehensive DataFrame with derived features.
    UPDATED for new 4-file dataset format with structured cluster database.
    
    Args:
        event_result: Dictionary with event data
        model_name: Name of the model
        cluster_info_dict: Dictionary with cluster indices per event
        events_h5_path: Path to events.h5 for loading structured cluster data
    """
    import h5py
    
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

    # --- ENHANCED: Load cluster information from new format ---
    cluster_index_i = np.full(num_edges, -1, dtype=np.int32)
    cluster_index_j = np.full(num_edges, -1, dtype=np.int32)
    cluster_eta_i = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_eta_j = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_phi_i = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_phi_j = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_e_i = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_e_j = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_num_cells_i = np.full(num_edges, -1, dtype=np.int32)
    cluster_num_cells_j = np.full(num_edges, -1, dtype=np.int32)
    
    # Load cluster indices (basic)
    if cluster_info_dict is not None and event_id in cluster_info_dict:
        # Get cluster index for cells (shape: [num_cells])
        cell_cluster_index = cluster_info_dict[event_id]['cell_cluster_index']
        
        # Map source cells to their clusters
        valid_src_indices = neighbor_pairs[:, 0] < len(cell_cluster_index)
        if valid_src_indices.any():
            cluster_index_i[valid_src_indices] = cell_cluster_index[neighbor_pairs[valid_src_indices, 0]]
        
        # Map target cells to their clusters  
        valid_dst_indices = neighbor_pairs[:, 1] < len(cell_cluster_index)
        if valid_dst_indices.any():
            cluster_index_j[valid_dst_indices] = cell_cluster_index[neighbor_pairs[valid_dst_indices, 1]]
    
    # --- ENHANCED: Load structured cluster properties from HDF5 ---
    if events_h5_path and os.path.exists(events_h5_path):
        try:
            with h5py.File(events_h5_path, 'r') as h5f:
                if 'clusters_structured' in h5f:
                    cluster_grp = h5f['clusters_structured']
                    
                    # Load cluster properties
                    if 'cluster_properties' in cluster_grp:
                        cluster_props = cluster_grp['cluster_properties'][:]
                        
                        # Create lookup dictionaries for fast access
                        # Map (event_id, cluster_idx_in_event) -> global_cluster_id
                        event_cluster_map = {}
                        for i, props in enumerate(cluster_props):
                            ev_id = props['event_id']
                            cl_idx = props['cluster_idx_in_event']
                            if ev_id not in event_cluster_map:
                                event_cluster_map[ev_id] = {}
                            event_cluster_map[ev_id][cl_idx] = i
                        
                        # If this event has clusters in the structured database
                        if event_id in event_cluster_map:
                            # Get the map for this event
                            ev_cluster_map = event_cluster_map[event_id]
                            
                            # For each unique cluster in this event, load properties
                            unique_clusters_i = np.unique(cluster_index_i[cluster_index_i >= 0])
                            unique_clusters_j = np.unique(cluster_index_j[cluster_index_j >= 0])
                            all_unique_clusters = np.unique(np.concatenate([unique_clusters_i, unique_clusters_j]))
                            
                            # Create property dictionaries
                            cluster_prop_dict = {}
                            for cl_idx in all_unique_clusters:
                                if cl_idx in ev_cluster_map:
                                    global_id = ev_cluster_map[cl_idx]
                                    props = cluster_props[global_id]
                                    cluster_prop_dict[cl_idx] = {
                                        'eta': props['eta'],
                                        'phi': props['phi'],
                                        'energy': props['energy'],
                                        'num_cells': props['num_cells']
                                    }
                            
                            # Assign properties to edges
                            for edge_idx in range(num_edges):
                                cl_i = cluster_index_i[edge_idx]
                                cl_j = cluster_index_j[edge_idx]
                                
                                if cl_i >= 0 and cl_i in cluster_prop_dict:
                                    props_i = cluster_prop_dict[cl_i]
                                    cluster_eta_i[edge_idx] = props_i['eta']
                                    cluster_phi_i[edge_idx] = props_i['phi']
                                    cluster_e_i[edge_idx] = props_i['energy']
                                    cluster_num_cells_i[edge_idx] = props_i['num_cells']
                                
                                if cl_j >= 0 and cl_j in cluster_prop_dict:
                                    props_j = cluster_prop_dict[cl_j]
                                    cluster_eta_j[edge_idx] = props_j['eta']
                                    cluster_phi_j[edge_idx] = props_j['phi']
                                    cluster_e_j[edge_idx] = props_j['energy']
                                    cluster_num_cells_j[edge_idx] = props_j['num_cells']
                            
                            # Debug info
                            clusters_with_props = len(cluster_prop_dict)
                            print(f"✅ Event {event_id}: Loaded properties for {clusters_with_props} clusters from structured database")
                            
                        else:
                            print(f"⚠️  Event {event_id}: Not found in structured cluster database")
                    
                    else:
                        print(f"⚠️  Event {event_id}: No cluster_properties in structured database")
                        
                else:
                    # Try old format: cluster property branches directly in HDF5
                    # These would be cell_to_cluster_* branches in the cell/ group
                    print(f"ℹ️  Event {event_id}: Using direct cluster property branches")
                    
                    # Check for cluster property branches
                    cell_group = h5f['cell']
                    cluster_prop_branches = {
                        'eta': 'cell_to_cluster_eta',
                        'phi': 'cell_to_cluster_phi',
                        'energy': 'cell_to_cluster_e'
                    }
                    
                    loaded_props = {}
                    for prop_name, branch_name in cluster_prop_branches.items():
                        if branch_name in cell_group:
                            loaded_props[prop_name] = cell_group[branch_name][event_id]
                    
                    if loaded_props:
                        # For each edge, get cluster properties from source and target cells
                        for edge_idx in range(num_edges):
                            src_idx = neighbor_pairs[edge_idx, 0]
                            dst_idx = neighbor_pairs[edge_idx, 1]
                            
                            if 'eta' in loaded_props:
                                cluster_eta_i[edge_idx] = loaded_props['eta'][src_idx]
                                cluster_eta_j[edge_idx] = loaded_props['eta'][dst_idx]
                            if 'phi' in loaded_props:
                                cluster_phi_i[edge_idx] = loaded_props['phi'][src_idx]
                                cluster_phi_j[edge_idx] = loaded_props['phi'][dst_idx]
                            if 'energy' in loaded_props:
                                cluster_e_i[edge_idx] = loaded_props['energy'][src_idx]
                                cluster_e_j[edge_idx] = loaded_props['energy'][dst_idx]
                        
                        print(f"✅ Event {event_id}: Loaded cluster properties from direct branches")
                    
        except Exception as e:
            print(f"⚠️  Event {event_id}: Error loading structured cluster data: {str(e)[:100]}")
    else:
        print(f"ℹ️  Event {event_id}: No HDF5 path provided for structured cluster data")

    # --- Derived features ---
    delta_eta = np.abs(eta_i - eta_j).astype(np.float32)
    delta_phi = np.abs(phi_i - phi_j).astype(np.float32)
    delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi).astype(np.float32)
    spatial_distance = np.sqrt(delta_eta**2 + delta_phi**2).astype(np.float32)
    
    # Safe SNR ratio calculation
    snr_ratio = np.zeros_like(snr_i, dtype=np.float32)
    mask_nonzero = snr_j != 0
    snr_ratio[mask_nonzero] = snr_i[mask_nonzero] / snr_j[mask_nonzero]
    snr_ratio[~mask_nonzero] = np.inf
    
    avg_snr = ((snr_i + snr_j) / 2).astype(np.float32)
    snr_sum = (snr_i + snr_j).astype(np.float32)
    snr_product = (snr_i * snr_j).astype(np.float32)
    
    # --- ENHANCED: Cluster-based features ---
    same_cluster = (cluster_index_i == cluster_index_j) & (cluster_index_i >= 0)
    
    # Cluster spatial features (if cluster properties available)
    cluster_delta_eta = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_delta_phi = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_spatial_distance = np.full(num_edges, np.nan, dtype=np.float32)
    
    mask_cluster_props = (cluster_eta_i != np.nan) & (cluster_eta_j != np.nan) & \
                         (cluster_phi_i != np.nan) & (cluster_phi_j != np.nan)
    
    if mask_cluster_props.any():
        cluster_delta_eta[mask_cluster_props] = np.abs(cluster_eta_i[mask_cluster_props] - cluster_eta_j[mask_cluster_props])
        cluster_delta_phi_temp = np.abs(cluster_phi_i[mask_cluster_props] - cluster_phi_j[mask_cluster_props])
        cluster_delta_phi[mask_cluster_props] = np.minimum(cluster_delta_phi_temp, 2*np.pi - cluster_delta_phi_temp)
        cluster_spatial_distance[mask_cluster_props] = np.sqrt(
            cluster_delta_eta[mask_cluster_props]**2 + cluster_delta_phi[mask_cluster_props]**2
        )
    
    # Cluster energy features
    cluster_e_ratio = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_e_sum = np.full(num_edges, np.nan, dtype=np.float32)
    cluster_e_avg = np.full(num_edges, np.nan, dtype=np.float32)
    
    mask_e_props = (cluster_e_i != np.nan) & (cluster_e_j != np.nan)
    if mask_e_props.any():
        mask_e_nonzero = mask_e_props & (cluster_e_j[mask_e_props] != 0)
        cluster_e_ratio[mask_e_nonzero] = cluster_e_i[mask_e_nonzero] / cluster_e_j[mask_e_nonzero]
        cluster_e_sum[mask_e_props] = cluster_e_i[mask_e_props] + cluster_e_j[mask_e_props]
        cluster_e_avg[mask_e_props] = (cluster_e_i[mask_e_props] + cluster_e_j[mask_e_props]) / 2
    
    # --- Confidence & predictions ---
    confidence = scores[np.arange(num_edges), preds].astype(np.float32)
    confidence_classes = [scores[:, i].astype(np.float32) for i in range(scores.shape[1])]
    true_labels = labels if labels is not None else np.full(num_edges, -1, dtype=np.int8)
    is_correct = np.where(true_labels != -1, preds == true_labels, False)

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
        'is_correct': is_correct.astype(bool),
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
        
        # --- ENHANCED: Cluster information from new format ---
        'cluster_index_source': cluster_index_i,
        'cluster_index_target': cluster_index_j,
        'same_cluster': same_cluster.astype(bool),
        'cluster_eta_source': cluster_eta_i,
        'cluster_phi_source': cluster_phi_i,
        'cluster_e_source': cluster_e_i,
        'cluster_num_cells_source': cluster_num_cells_i,
        'cluster_eta_target': cluster_eta_j,
        'cluster_phi_target': cluster_phi_j,
        'cluster_e_target': cluster_e_j,
        'cluster_num_cells_target': cluster_num_cells_j,
        'cluster_delta_eta': cluster_delta_eta,
        'cluster_delta_phi': cluster_delta_phi,
        'cluster_spatial_distance': cluster_spatial_distance,
        'cluster_e_ratio': cluster_e_ratio,
        'cluster_e_sum': cluster_e_sum,
        'cluster_e_avg': cluster_e_avg,
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


def save_results_directly(final_results_per_event: List[Dict], save_dir: str, 
                          model_name: str, cluster_info_dict: Dict = None,
                          events_h5_path: str = None,  # NEW PARAMETER
                          chunk_size: int = 50) -> str:
    """
    Save inference results directly to a single Parquet file in a memory-efficient way.
    Updated parameter name from cluster_info to cluster_info_dict.
    """
    print(f"💾 Saving results directly to Parquet for {model_name}...")
    print(f"   Processing {len(final_results_per_event)} events in chunks of {chunk_size}")

    os.makedirs(save_dir, exist_ok=True)
    parquet_path = os.path.join(save_dir, f"comprehensive_results_{model_name}.parquet")

    # Remove existing file if present
    if os.path.exists(parquet_path):
        os.remove(parquet_path)

    # Process events in chunks
    first_chunk = True
    schema = None
    
    for start_idx in range(0, len(final_results_per_event), chunk_size):
        end_idx = min(start_idx + chunk_size, len(final_results_per_event))
        chunk_events = final_results_per_event[start_idx:end_idx]

        if start_idx % 100 == 0:
            print(f"   Processing chunk {start_idx}-{end_idx}...")

        # Process each event in the chunk
        chunk_dfs = []
        for event_result in chunk_events:
            df_event = _process_single_event_comprehensive(
                event_result, model_name, cluster_info_dict, events_h5_path
            )
            df_event = optimize_dataframe_memory(df_event)
            chunk_dfs.append(df_event)
        
        # Combine chunk and write to Parquet
        if chunk_dfs:
            df_chunk = pd.concat(chunk_dfs, ignore_index=True)
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            
            if first_chunk:
                pq.write_table(table, parquet_path)
                schema = table.schema
                first_chunk = False
            else:
                # Append chunk to Parquet
                with pq.ParquetWriter(parquet_path, schema, use_dictionary=True, compression="snappy") as writer:
                    writer.write_table(table)
            
            del df_chunk, table, chunk_dfs
            gc.collect()

    print(f"💾 Direct save complete: {parquet_path}")
    return parquet_path


# Generator
class MultiClassBatchGenerator(IterableDataset):
    """
    Optimized IterableDataset for large events (~1.25M edges):
      • Uses all neighbor pairs per event
      • Supports unscaled features for analysis
      • FIXED: First 70% training, last 30% testing with consistent ordering
      • Updated for new dataset format with labels as 2D array
    """

    def __init__(
        self,
        features_dict: Dict[int, np.ndarray],
        neighbor_pairs: np.ndarray,
        labels: np.ndarray,  # Now a 2D array: (num_events, num_pairs)
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
        self.num_events = len(features_dict)
        self.num_pairs = neighbor_pairs.shape[0]

        # NEW: Validate labels shape BEFORE converting to tensor
        if labels.ndim != 2:
            raise ValueError(f"Labels must be 2D array (num_events, num_pairs), got shape {labels.shape}")
        
        if labels.shape[0] != self.num_events:
            raise ValueError(f"Labels first dimension ({labels.shape[0]}) must equal num_events ({self.num_events})")
        
        if labels.shape[1] != self.num_pairs:
            raise ValueError(f"Labels second dimension ({labels.shape[1]}) must equal num_pairs ({self.num_pairs})")
        
        print(f"✅ Labels shape validated: {labels.shape} matches {self.num_events} events × {self.num_pairs} edges")

        # Store scaled features as float32 tensors
        self.features_dict = {k: torch.as_tensor(v, dtype=torch.float32)
                              for k, v in features_dict.items()}

        # Optional unscaled features for analysis
        self.unscaled_features_dict = (
            self._precompute_unscaled_features(unscaled_data_dict)
            if unscaled_data_dict is not None else None
        )

        # Neighbor pairs and labels (labels is now 2D array)
        self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)  # shape: (num_events, num_pairs)

        # FIXED: Consistent train/test split - first 70% training, last 30% testing
        split_idx = int(self.num_events * train_ratio)
        
        if mode == "train":
            self.event_indices = list(range(0, split_idx))  # First 70%
            self.event_indices_tensor = torch.arange(0, split_idx, dtype=torch.long)
        else:  # "test" or "val"
            self.event_indices = list(range(split_idx, self.num_events))  # Last 30%
            self.event_indices_tensor = torch.arange(split_idx, self.num_events, dtype=torch.long)
        
        print(f"📊 {mode.upper()} SET: Events {self.event_indices[0] if self.event_indices else 'N/A'}-"
              f"{self.event_indices[-1] if self.event_indices else 'N/A'} "
              f"({len(self.event_indices)} events)")

        # Precompute all samples for fast iteration
        self.precomputed_samples = self._precompute_all_samples()

        if self.debug:
            print(f"Initialized with {len(self.precomputed_samples)} samples")
            print(f"Mode: {mode}, Events: {len(self.event_indices)}")
            print(f"Features dict keys: {sorted(list(self.features_dict.keys()))[:5]}...")
            print(f"Labels shape: {self.labels.shape}")

    def _precompute_unscaled_features(self, unscaled_data_dict: Dict[int, np.ndarray]) -> Dict[int, torch.Tensor]:
        """Convert raw detector data to [snr, η, φ] format for analysis, keyed by integer event_id."""
        unscaled_features: Dict[int, torch.Tensor] = {}
        for event_id, arr in unscaled_data_dict.items():
            # arr should already be [snr, eta, phi] from load_shared_data
            if arr.shape[1] == 3:
                t = torch.as_tensor(arr, dtype=torch.float32)
                # Verify it's [snr, eta, phi]
                if self.debug and event_id == 0:
                    print(f"Event 0 unscaled features shape: {t.shape}")
                    print(f"  SNR range: {t[:, 0].min():.2f} to {t[:, 0].max():.2f}")
                    print(f"  Eta range: {t[:, 1].min():.2f} to {t[:, 1].max():.2f}")
                    print(f"  Phi range: {t[:, 2].min():.2f} to {t[:, 2].max():.2f}")
            else:
                raise ValueError(f"Unexpected feature dimension {arr.shape[1]} for event {event_id}. Expected 3 [snr, eta, phi]")
            unscaled_features[event_id] = t
        return unscaled_features
        
    def _precompute_all_samples(self) -> List[Tuple]:
        """
        Precompute all event samples for fast iteration.
        Labels are now accessed by event index from 2D array.
        """
        samples: List[Tuple] = []
        
        for event_idx in self.event_indices:
            # Verify event exists in features_dict
            if event_idx not in self.features_dict:
                raise KeyError(f"Event index {event_idx} not found in features_dict. "
                             f"Available keys: {sorted(list(self.features_dict.keys()))[:10]}...")
            
            # Get scaled features for this event
            x_scaled = self.features_dict[event_idx]
            
            # Get unscaled features if available
            x_unscaled = (self.unscaled_features_dict.get(event_idx)
                          if self.unscaled_features_dict else None)
            
            # Get labels for this event from 2D array
            # IMPORTANT: event_idx must be valid index into self.labels
            if event_idx >= self.labels.shape[0]:
                raise IndexError(f"Event index {event_idx} out of bounds for labels array with shape {self.labels.shape}")
            
            out_labels = self.labels[event_idx]  # Shape: [num_pairs]
            
            # Ensure labels are 2D: [num_pairs, 1] for compatibility
            out_labels = out_labels.unsqueeze(1) if out_labels.dim() == 1 else out_labels
            
            # Get cluster info if available
            cluster_info = (self.cluster_info_dict.get(event_idx) 
                           if self.cluster_info_dict else None)
            
            # Use all neighbor pairs for this event
            pairs = self.neighbor_pairs.T  # Shape: [2, num_pairs]
            
            samples.append((x_scaled, pairs, pairs.clone(), 
                           out_labels, x_unscaled, cluster_info))
            
            if self.debug and event_idx == 0:
                print(f"Sample 0 debug:")
                print(f"  x_scaled shape: {x_scaled.shape}")
                print(f"  pairs shape: {pairs.shape}")
                print(f"  out_labels shape: {out_labels.shape}")
                print(f"  x_unscaled shape: {x_unscaled.shape if x_unscaled is not None else 'None'}")
                print(f"  cluster_info: {cluster_info.keys() if cluster_info else 'None'}")
        
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
        cluster_info_list = None if batch[0][5] is None else [b[5] for b in batch]
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
                debug: bool = False, accumulation_steps: int = 1) -> Dict[str, float]:
    """
    Train the model for one epoch with optional gradient accumulation.
    """
    model.train()
    total_loss = correct = total = 0
    start_train = time.perf_counter()
    
    optimizer.zero_grad()  # Initial gradient reset

    for batch_idx, (x_list, edge_idx_list, edge_idx_out_list, y_batch, unscaled_list, cluster_info_list) in enumerate(loader):
        if debug:
            start_batch = time.perf_counter()

        # Move all tensors in the batch to GPU/CPU device
        x_list = [x.to(device, non_blocking=True) for x in x_list]
        edge_idx_list = [e.to(device, non_blocking=True) for e in edge_idx_list]
        edge_idx_out_list = [e.to(device, non_blocking=True) for e in edge_idx_out_list]
        y_batch = y_batch.to(device, non_blocking=True).squeeze(1)

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type="cuda"):
            scores = model(x_list, edge_idx_list, edge_idx_out_list)
            loss = criterion(scores, y_batch) / accumulation_steps  # Scale loss for accumulation

        # Backpropagation with gradient scaling
        scaler.scale(loss).backward()

        # Only update weights after accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Track loss and accuracy
        total_loss += loss.item() * len(y_batch) * accumulation_steps  # Rescale
        preds = scores.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

        # Optional debug info every 10 batches
        if debug and batch_idx % 10 == 0:
            print(f"Batch {batch_idx+1}: loss={loss.item()*accumulation_steps:.4f}, "
                  f"time={time.perf_counter() - start_batch:.3f}s")

    # Handle remaining gradients if not on accumulation boundary
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()

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
    Run inference event-by-event using MultiClassBatchGenerator.
    Updated for new dataset format with correct event IDs.
    """

    model.eval()
    all_results = []

    for i, (x_scaled, edge_index, edge_index_out, y,
            x_unscaled, cluster_info) in enumerate(generator):

        # Get actual event ID from generator's event_indices
        # generator.event_indices contains the actual event numbers
        if hasattr(generator, 'event_indices') and i < len(generator.event_indices):
            event_id = generator.event_indices[i]
        else:
            event_id = i  # Fallback
        
        if debug and i % 10 == 0:
            print(f"Processing event {event_id} (batch index {i})")

        # Move tensors to device
        x_scaled = x_scaled.to(device)
        edge_index = edge_index.to(device)
        edge_index_out = edge_index_out.to(device)

        # Forward pass
        out = model([x_scaled], [edge_index], [edge_index_out])
        preds = out.argmax(dim=1).cpu().numpy()
        scores = torch.softmax(out, dim=1).cpu().numpy()

        # Edge endpoints (preserves order)
        src_nodes = edge_index_out[0].cpu()
        dst_nodes = edge_index_out[1].cpu()

        # Always have scaled features
        feats_i_scaled = x_scaled[src_nodes].cpu().numpy()
        feats_j_scaled = x_scaled[dst_nodes].cpu().numpy()

        # Unscaled features OPTIONAL
        if x_unscaled is not None:
            x_unscaled = x_unscaled.cpu()
            feats_i_unscaled = x_unscaled[src_nodes].numpy()
            feats_j_unscaled = x_unscaled[dst_nodes].numpy()
            # Extract eta (feature index 1) and phi (feature index 2)
            eta_i = feats_i_unscaled[:, 1]
            eta_j = feats_j_unscaled[:, 1]
            phi_i = feats_i_unscaled[:, 2]  # Optional: add phi if needed
            phi_j = feats_j_unscaled[:, 2]
        else:
            feats_i_unscaled = None
            feats_j_unscaled = None
            eta_i = None
            eta_j = None
            phi_i = None
            phi_j = None

        # Handle labels (might be 2D: [num_edges, 1])
        labels_np = None
        if y is not None:
            if y.dim() == 2 and y.shape[1] == 1:
                labels_np = y.squeeze(1).numpy()  # Remove extra dimension
            else:
                labels_np = y.numpy()

        all_results.append({
            "event_id": event_id,  # Use actual event ID
            "preds": preds,
            "scores": scores,
            "labels": labels_np,
            "neighbor_pairs": edge_index_out.cpu().numpy().T,  # Shape: [num_edges, 2]
            "features_i_scaled": feats_i_scaled,  # Shape: [num_edges, 3]
            "features_j_scaled": feats_j_scaled,  # Shape: [num_edges, 3]
            "features_i_unscaled": feats_i_unscaled,  # Shape: [num_edges, 3] or None
            "features_j_unscaled": feats_j_unscaled,  # Shape: [num_edges, 3] or None
            "eta_i": eta_i,  # Shape: [num_edges] or None
            "eta_j": eta_j,  # Shape: [num_edges] or None
            "phi_i": phi_i,  # Optional: add if needed
            "phi_j": phi_j,  # Optional: add if needed
            "cluster_info": cluster_info,  # Dict or None
        })

        if debug and i == 0:
            print(f"Event {event_id} debug:")
            print(f"  x_scaled shape: {x_scaled.shape}")
            print(f"  edge_index_out shape: {edge_index_out.shape}")
            print(f"  preds shape: {preds.shape}")
            print(f"  scores shape: {scores.shape}")
            print(f"  labels shape: {labels_np.shape if labels_np is not None else 'None'}")
            print(f"  feats_i_scaled shape: {feats_i_scaled.shape}")
            if feats_i_unscaled is not None:
                print(f"  feats_i_unscaled shape: {feats_i_unscaled.shape}")
                print(f"  eta_i shape: {eta_i.shape}")

        if debug and i >= 4:
            print("Debug mode: stopping after 5 events")
            break

    return all_results
 

def run_model(model, batch_size, save_dir, best_model_name,
              train_generator_class, test_generator_class,
              train_generator_kwargs, test_generator_kwargs,
              epochs, device, optimizer, criterion,
              unscaled_data_dict=None, cluster_info=None,
              events_h5_path=None,  # NEW PARAMETER
              lr=1e-3, resume=True, patience=10, delta=1e-4,
              debug=False, save_direct_to_parquet=True):

    # -------------------------------------------------
    # Setup: ensure dirs and filenames
    # -------------------------------------------------
    if not debug:
        os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, best_model_name)
    best_model_path = os.path.join(save_dir, f"best_{best_model_name}")
    metrics_path = os.path.splitext(model_path)[0] + ".pkl"

    scaler = torch.amp.GradScaler(device="cuda")
    best_epoch = 0
    best_test_loss = float("inf")
    best_test_acc = 0.0
    start_epoch = 1
    total_time_trained = 0.0

    # -------------------------------------------------
    # Resume from checkpoint
    # -------------------------------------------------
    metrics = {}

    if resume:
        chk = find_latest_checkpoint(save_dir, best_model_name)
        if chk:
            resumed_epoch, chk_path = chk
            checkpoint = torch.load(chk_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = resumed_epoch + 1

            if not debug:
                print(f"[Resume] Loaded checkpoint: {chk_path}")

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'rb') as f:
                        metrics = pickle.load(f)
                    best_test_loss = metrics.get("best_test_loss", best_test_loss)
                    best_test_acc = metrics.get("best_test_acc", best_test_acc)
                    best_epoch = metrics.get("best_epoch", 0)
                    total_time_trained = metrics.get("total_time", 0.0)
                except:
                    metrics = {}
        else:
            if not debug:
                print("[Resume] No checkpoint found. Starting fresh.")

    # -------------------------------------------------
    # Metrics containers
    # -------------------------------------------------
    metrics.update({
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "epoch_times": [],
        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "total_time": total_time_trained,
        "time_per_epoch": 0.0
    })

    # -------------------------------------------------
    # Build data loaders once
    # -------------------------------------------------
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

    early_counter = 0

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()

        train_res = train_model(model, train_loader, optimizer, criterion, scaler, device, debug)
        test_res = test_model(model, test_loader, criterion, device, debug)

        dt = time.perf_counter() - t0

        metrics["epoch_times"].append(dt)
        metrics["train_loss"].append(train_res["loss"])
        metrics["train_acc"].append(train_res["acc"])
        metrics["test_loss"].append(test_res["loss"])
        metrics["test_acc"].append(test_res["acc"])

        # -------------------------------------------------
        # Best model logic (test loss)
        # -------------------------------------------------
        if test_res["loss"] < best_test_loss - delta:
            best_test_loss = test_res["loss"]
            best_test_acc = test_res["acc"]
            best_epoch = epoch
            metrics["best_test_loss"] = best_test_loss
            metrics["best_test_acc"] = best_test_acc
            metrics["best_epoch"] = best_epoch
            early_counter = 0

            if not debug:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }, best_model_path)
        else:
            early_counter += 1

        # -------------------------------------------------
        # Save checkpoint every epoch
        # -------------------------------------------------
        if not debug:
            chk_path = os.path.join(save_dir, f"{os.path.splitext(best_model_name)[0]}_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, chk_path)

        if debug or epoch % 5 == 0:
            print(f"[Epoch {epoch}] Time {dt:.1f}s  "
                  f"TrainLoss={train_res['loss']:.4f}  TestLoss={test_res['loss']:.4f}  "
                  f"TestAcc={test_res['acc']:.4f}  BestAcc={best_test_acc:.4f}")

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------
        if early_counter >= patience:
            if not debug:
                print(f"[Early Stop] Triggered at epoch {epoch}")
            break

    # -------------------------------------------------
    # Finalize metrics
    # -------------------------------------------------
    metrics["total_time"] += sum(metrics["epoch_times"])
    metrics["time_per_epoch"] = np.mean(metrics["epoch_times"])

    # -------------------------------------------------
    # Final inference (best model)
    # -------------------------------------------------
    if not debug:
        chk = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(chk["model_state_dict"])

        test_gen = test_generator_class(**test_generator_kwargs)
        final_results = run_inference_with_generator(model, test_gen, device, debug)

        metrics["num_events_evaluated"] = len(final_results)
        metrics["final_results_per_event"] = final_results

        # Flatten predictions
        metrics["final_preds_concatenated"] = np.concatenate([ev["preds"] for ev in final_results])
        metrics["final_scores_concatenated"] = np.concatenate([ev["scores"] for ev in final_results])
        metrics["final_labels_concatenated"] = (
            np.concatenate([ev["labels"] for ev in final_results])
            if final_results[0]["labels"] is not None else None
        )

        # -------------------------------------------------
        # Save directly to Parquet
        # -------------------------------------------------
        if save_direct_to_parquet:
            model_base = os.path.splitext(best_model_name)[0]
            parquet_path = save_results_directly(
                final_results, 
                save_dir, 
                model_base, 
                cluster_info_dict=cluster_info,
                events_h5_path=events_h5_path  # PASSED CORRECTLY
            )

            training_metrics = {
                "train_loss": metrics["train_loss"],
                "test_loss": metrics["test_loss"],
                "train_acc": metrics["train_acc"],
                "test_acc": metrics["test_acc"],
                "best_test_acc": metrics["best_test_acc"],
                "best_test_loss": metrics["best_test_loss"],
                "best_epoch": metrics["best_epoch"],
                "total_time": metrics["total_time"],
                "time_per_epoch": metrics["time_per_epoch"],
                "parquet_results_path": parquet_path,
                "train_events": train_generator_kwargs.get("event_indices", None),
                "test_events": test_generator_kwargs.get("event_indices", None),
            }

            save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), training_metrics)

            print("\n[Saved] Direct Parquet export complete")
            print("Training metrics saved to:", metrics_path)
            print("Parquet results saved to:", parquet_path)

        else:
            save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), metrics)

        total_m, total_s = divmod(metrics["total_time"], 60)
        print(f"\nTraining complete in {int(total_m)}m {total_s:.1f}s")
        print(f"Best Epoch {best_epoch} | Best Test Acc = {best_test_acc:.4f}")

    return metrics, model, best_model_path


def train_single_gpu(gpu_id: int, config: Dict, shared_data: Tuple):
    """
    Train a single model on one GPU with the given configuration and shared data.
    UPDATED for new 4-file dataset format.
    """
    try:
        print(f" Starting training on GPU {gpu_id} - {config['model_name']}")
        print(f"   Generator flags: {config['generator_flags']}")
        print(f"   Model flags: {config['model_flags']}")

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.empty_cache()

        # Unpack shared data
        scaled_data, unscaled_data, neighbor_pairs_list, labels_for_neighbor_pairs, cluster_info = shared_data

        # Create loss function
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
            else:
                weight_tensor = compute_inverse_freq_weights(
                    labels_for_neighbor_pairs, config['num_classes'], device
                )

            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"GPU {gpu_id}: Using {weight_strategy} weighted loss")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"GPU {gpu_id}: Using standard loss function")

        # Build model
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

        optimizer = optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # ---------------------------------------------------------
        # CLEANED generator kwargs — NO batch_size here!
        # ---------------------------------------------------------
        gen_kwargs = {
            'features_dict': scaled_data,
            'neighbor_pairs': neighbor_pairs_list,
            'labels': labels_for_neighbor_pairs,
            'unscaled_data_dict': unscaled_data,
            'cluster_info_dict': cluster_info,
            'debug': config.get('debug', False),
            **config['generator_flags'],
        }

        # Split into train/test
        train_kwargs = {**gen_kwargs, 'mode': 'train'}
        test_kwargs = {**gen_kwargs, 'mode': 'test'}

        # ---------------------------------------------------------
        # Call cleaned-up run_model() WITH events_h5_path
        # ---------------------------------------------------------
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
            cluster_info=cluster_info,
            events_h5_path="/storage/mxg1065/datafiles/events.h5",  # NEW: Add this
            debug=config.get('debug', False),
            resume=config.get('resume', True),
            patience=config.get('patience', 10),
            delta=config.get('delta', 0.0001),
            save_direct_to_parquet=True
        )

        print(f" GPU {gpu_id} training completed!")
        print(f"   Best accuracy: {metrics['best_test_acc']:.4f}")
        print(f"   Total time: {metrics['total_time']:.1f}s")
        print(f"   Model saved to: {model_path}")

        return 0

    except Exception as e:
        print(f" GPU {gpu_id} training failed: {e}")
        traceback.print_exc()
        return 1


def log(msg):
    """Simple timestamped logger."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def main():
    """
    Multi-GPU launcher with extended logging output.
    """
    print("=" * 60)
    log("Multi-GPU Particle Physics Classification Training Initialized")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. LOAD SHARED DATA
    # -------------------------------------------------------------
    log("Starting dataset load...")
    load_path = "/storage/mxg1065/datafiles"
    try:
        shared_data = load_shared_data(load_path)
        log("Dataset successfully loaded into shared memory.")
    except Exception as e:
        log(f"[FATAL] Dataset loading failed: {e}")
        return

    # -------------------------------------------------------------
    # 2. DETECT GPUs
    # -------------------------------------------------------------
    available_gpus = torch.cuda.device_count()
    log(f"Detected {available_gpus} CUDA GPU(s).")

    for i in range(available_gpus):
        props = torch.cuda.get_device_properties(i)
        log(f" GPU {i}: {torch.cuda.get_device_name(i)} "
            f"with {props.total_memory/1024**3:.1f} GB memory")

    # -------------------------------------------------------------
    # 3. BASE CONFIG
    # -------------------------------------------------------------
    base_config = {
        'num_features': 3,
        'num_classes': 5,
        'hidden_dim': 128,
        'lr': 1e-3,
        'weight_decay': 5e-4,
        'epochs': 50,
        'batch_size': 1,
        'save_dir': "/storage/mxg1065/fixed_batch_size_models",
        'resume': True,
        'patience': 20,
        'delta': 0.0001,
        'debug': False,
        'weighted': False,
        'generator_flags': {
            'is_bi_directional': True,
            'train_ratio': 0.7,
        },
        'model_flags': {
            'num_layers': 6,
            'layer_weights': False,
            'softmax': False,
        }
    }

    # -------------------------------------------------------------
    # 4. PER-GPU CONFIGS
    # -------------------------------------------------------------
    configs = [
        # GPU 0
        {**base_config,
        'model_name': "new_bs1_model.pt",
        'description': "This is the model where the new data files/structures are used",
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

    configs = configs[:available_gpus]

    log("Training jobs configured:")
    for i, cfg in enumerate(configs):
        log(f" GPU {i}: {cfg['description']}")
        log(f"      Model: {cfg['model_name']}")
        log(f"      LR={cfg['lr']}  HiddenDim={cfg['hidden_dim']}  Batch={cfg['batch_size']}")
        log("      Train/Test split: 70% / 30%")

    # -------------------------------------------------------------
    # 5. USER CONFIRMATION
    # -------------------------------------------------------------
    resp = input("\nStart training? (y/n): ").strip().lower()
    if resp not in ['y', 'yes']:
        log("Training canceled by user.")
        return

    # -------------------------------------------------------------
    # 6. LAUNCH TRAINING PROCESSES
    # -------------------------------------------------------------
    log("Launching processes...")
    processes = []
    start_times = []

    for gpu_id, config in enumerate(configs):
        try:
            log(f"Preparing GPU {gpu_id}...")
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1024**3
            log(f" GPU {gpu_id} free memory BEFORE start: {free_mem:.2f} GB")

            p = mp.Process(
                target=train_single_gpu,
                args=(gpu_id, config, shared_data),
                daemon=False
            )
            p.start()

            log(f" Started process PID {p.pid} on GPU {gpu_id} ({config['model_name']}).")

            processes.append(p)
            start_times.append(time.time())

            # Stagger launches
            if gpu_id < len(configs) - 1:
                time.sleep(25)

        except Exception as e:
            log(f"[ERROR] Could not start GPU {gpu_id}: {e}")

    # -------------------------------------------------------------
    # 7. MONITOR TRAINING
    # -------------------------------------------------------------
    log("Monitoring processes every 30 seconds...")

    finished = set()
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(30)
            alive = sum(p.is_alive() for p in processes)

            # Per-process status
            for i, p in enumerate(processes):
                if not p.is_alive() and i not in finished:
                    exitcode = p.exitcode
                    if exitcode == 0:
                        log(f" GPU {i} training completed successfully.")
                    else:
                        log(f" GPU {i} crashed (Exit {exitcode}).")
                    finished.add(i)

            elapsed = time.time() - min(start_times)
            log(f" Status: {alive}/{len(processes)} running "
                f"(Elapsed {elapsed/60:.1f} min)")

        log("All GPU processes finished.")

    except KeyboardInterrupt:
        log("KeyboardInterrupt detected — terminating running processes.")
        for p in processes:
            if p.is_alive():
                p.terminate()
        log("All processes terminated manually.")

    # -------------------------------------------------------------
    # 8. PROCESS CLEANUP
    # -------------------------------------------------------------
    log("Waiting for process cleanup...")
    for i, p in enumerate(processes):
        p.join()
        log(f" Process {i} exit code: {p.exitcode}")

    log("Multi-GPU training complete.")

    # -------------------------------------------------------------
    # 9. FINAL SUMMARY
    # -------------------------------------------------------------
    log("Collecting final summaries:\n")
    for i, config in enumerate(configs):
        model_base = os.path.join(config['save_dir'], config['model_name'])
        metrics_path = os.path.splitext(model_base)[0] + ".pkl"

        if not os.path.exists(metrics_path):
            log(f" GPU {i}: Metrics file not found.")
            continue

        try:
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)

            log(f" GPU {i} Summary: {config['description']}")
            log(f"    Best Test Accuracy: {metrics.get('best_test_acc', 0):.4f}")
            log(f"    Total Time: {metrics.get('total_time', 0)/60:.1f} min")
            log(f"    Train Events: {metrics.get('train_events', '?')}")
            log(f"    Test Events: {metrics.get('test_events', '?')}")
            log(f"    Results Saved: {metrics.get('parquet_results_path', '?')}")
        except Exception:
            log(f" GPU {i}: Could not read metrics.")

    log("All training summaries printed. Done.")


# =========================================================
# Updated test functions for new format
# =========================================================
def quick_data_check(load_path: str):
    """Check for new 4-file format instead of old pickle files."""
    print("🔍 Quick data file check (NEW FORMAT)...")
    
    required_files = [
        "cells.npy",
        "pairs.npy", 
        "events.h5",
        "labels.npy",
        "dataset_manifest.json"
    ]
    
    all_files_exist = True
    for file in required_files:
        file_path = os.path.join(load_path, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024**3)  # GB
            print(f"   ✅ {file}: {size:.2f} GB")
        else:
            print(f"   ❌ {file} - missing")
            all_files_exist = False
    
    if all_files_exist:
        print("✅ All required new-format files found!")
    else:
        print("❌ Some files missing in new format!")
    
    return all_files_exist


def test_data_integrity(load_path: str):
    """
    Comprehensive test for new 4-file dataset format.
    """
    print("\n" + "="*60)
    print("DATA INTEGRITY TEST (NEW FORMAT)")
    print("="*60)
    
    all_tests_passed = True
    
    try:
        # Load data using our new function
        scaled_data, unscaled_data, neighbor_pairs_array, labels_array, cluster_info_dict = load_shared_data(load_path)
        
        # ----------------------------------------------------
        # 1. Basic shapes
        # ----------------------------------------------------
        print("1. Testing basic shapes...")
        num_events = len(scaled_data)
        num_cells = next(iter(scaled_data.values())).shape[0]
        num_pairs = neighbor_pairs_array.shape[0]
        
        print(f"   Events: {num_events}")
        print(f"   Cells per event: {num_cells}")
        print(f"   Pairs: {num_pairs}")
        
        assert labels_array.shape == (num_events, num_pairs), f"Labels shape {labels_array.shape} != ({num_events}, {num_pairs})"
        assert neighbor_pairs_array.shape[1] == 2, "Neighbor pairs should have 2 columns"
        print("   ✅ Basic shapes test passed")
        
        # ----------------------------------------------------
        # 2. Feature consistency
        # ----------------------------------------------------
        print("2. Testing feature consistency...")
        for ev in range(min(5, num_events)):
            scaled_features = scaled_data[ev]
            unscaled_features = unscaled_data[ev]
            
            assert scaled_features.shape == unscaled_features.shape, f"Shape mismatch for event {ev}"
            assert scaled_features.shape[1] == 3, f"Expected 3 features, got {scaled_features.shape[1]}"
            
            # Eta and phi should match (snr will differ scaled vs unscaled)
            np.testing.assert_array_equal(scaled_features[:, 1], unscaled_features[:, 1], 
                                        err_msg=f"Eta values differ for event {ev}")
            np.testing.assert_array_equal(scaled_features[:, 2], unscaled_features[:, 2], 
                                        err_msg=f"Phi values differ for event {ev}")
        print("   ✅ Feature consistency test passed")
        
        # ----------------------------------------------------
        # 3. Neighbor pairs validity
        # ----------------------------------------------------
        print("3. Testing neighbor pairs...")
        max_cell_index = num_cells - 1
        max_pair_index = neighbor_pairs_array.max()
        
        assert max_pair_index <= max_cell_index, f"Neighbor pair index {max_pair_index} exceeds max cell index {max_cell_index}"
        assert neighbor_pairs_array.min() >= 0, "Negative indices in neighbor pairs"
        
        self_loops = np.sum(neighbor_pairs_array[:, 0] == neighbor_pairs_array[:, 1])
        assert self_loops == 0, f"Found {self_loops} self-loops in neighbor pairs"
        
        print("   ✅ Neighbor pairs test passed")
        
        # ----------------------------------------------------
        # 4. Labels validity
        # ----------------------------------------------------
        print("4. Testing labels...")
        unique_labels = np.unique(labels_array)
        assert set(unique_labels).issubset({0, 1, 2, 3, 4}), f"Invalid labels found: {unique_labels}"
        
        label_counts = np.bincount(labels_array[0].flatten())
        print(f"   Label distribution in first event: {dict(zip(range(5), label_counts))}")
        print("   ✅ Labels test passed")
        
        # ----------------------------------------------------
        # 5. Data ranges
        # ----------------------------------------------------
        print("5. Testing data ranges...")
        for ev in range(min(3, num_events)):
            features = scaled_data[ev]
            
            assert not np.any(np.isnan(features)), f"NaN values found in event {ev}"
            assert not np.any(np.isinf(features)), f"Infinite values found in event {ev}"
            
            # Reasonable ranges for eta/phi
            assert np.all(features[:, 1] >= -5) and np.all(features[:, 1] <= 5), f"Eta out of range in event {ev}"
            assert np.all(features[:, 2] >= -4) and np.all(features[:, 2] <= 4), f"Phi out of range in event {ev}")
        
        print("   ✅ Data ranges test passed")
        
        # ----------------------------------------------------
        # 6. Cluster info
        # ----------------------------------------------------
        print("6. Testing cluster info...")
        if cluster_info_dict:
            assert len(cluster_info_dict) == num_events, f"Cluster info for {len(cluster_info_dict)} events != {num_events}"
            for ev in range(min(3, num_events)):
                assert 'cell_cluster_index' in cluster_info_dict[ev], f"Missing cluster index for event {ev}"
            print("   ✅ Cluster info test passed")
        else:
            print("   ℹ️  No cluster info available")
        
        all_tests_passed = True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    # ----------------------------------------------------
    # Final summary
    # ----------------------------------------------------
    print("\n" + "="*60)
    if all_tests_passed:
        print("🎉 ALL NEW FORMAT TESTS PASSED! 🎉")
    else:
        print("⚠️  SOME TESTS FAILED - Please check data consistency")
    print("="*60)
    
    return all_tests_passed


def verify_order_preservation(data_dir: str, parquet_path: str):
    """Verify order preservation with new dataset format - FIXED VERSION."""
    import pandas as pd
    print("\n🔍 Verifying order preservation (NEW FORMAT)...")
    
    try:
        # Load original data
        scaled_data, unscaled_data, neighbor_pairs_array, labels_array, _ = load_shared_data(data_dir)
        
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        
        print(f"  Parquet file info:")
        print(f"    Total rows: {len(df):,}")
        print(f"    Events in parquet: {df['event_id'].nunique()}")
        print(f"    Event IDs: {sorted(df['event_id'].unique())[:10]}...")
        
        # Find first event in parquet (not necessarily event 0)
        if df['event_id'].nunique() == 0:
            print("  ❌ No events found in parquet file!")
            return False
            
        first_event_in_parquet = df['event_id'].min()
        print(f"  First event in parquet: {first_event_in_parquet}")
        
        # Get matching event from original data
        event_0_mask = df['event_id'] == first_event_in_parquet
        n_test = min(100, sum(event_0_mask))
        
        if n_test == 0:
            print(f"  ❌ No edges found for event {first_event_in_parquet} in parquet!")
            return False
            
        # Get edges from original data for this event
        original_edges = neighbor_pairs_array[:n_test]  # Edge order is same for all events
        original_labels = labels_array[first_event_in_parquet, :n_test]  # Use correct event
        
        # Get edges from parquet
        parquet_edges = df[event_0_mask][['source_id', 'target_id']].values[:n_test]
        parquet_true = df[event_0_mask]['true_label'].values[:n_test]
        
        edges_match = np.array_equal(original_edges, parquet_edges)
        labels_match = np.array_equal(original_labels, parquet_true)
        
        print(f"  Edge order preserved: {'✅ YES' if edges_match else '❌ NO'}")
        print(f"  True label order preserved: {'✅ YES' if labels_match else '❌ NO'}")
        print(f"  Tested {n_test} edges from event {first_event_in_parquet}")
        
        if not edges_match and n_test > 0:
            mismatch_idx = np.where((original_edges != parquet_edges).any(axis=1))[0][0]
            print(f"  First mismatch at index {mismatch_idx}:")
            print(f"    Original: {original_edges[mismatch_idx]}")
            print(f"    Parquet: {parquet_edges[mismatch_idx]}")
        
        return edges_match and labels_match
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generator_compatibility(load_path: str):
    """Test that the generator works with new data format."""
    print("\n🧪 Testing generator compatibility...")
    
    scaled_data, unscaled_data, neighbor_pairs, labels, cluster_info = load_shared_data(load_path)
    
    generator = MultiClassBatchGenerator(
        features_dict=scaled_data,
        neighbor_pairs=neighbor_pairs,
        labels=labels,
        mode="train",
        is_bi_directional=True,
        batch_size=1,
        unscaled_data_dict=unscaled_data,
        cluster_info_dict=cluster_info,
        debug=True
    )
    
    try:
        for i, batch in enumerate(generator):
            x_scaled, edge_idx, edge_idx_out, y, x_unscaled, cluster_info = batch
            print(f"  Event {i}: {x_scaled.shape}, edges: {edge_idx.shape}, labels: {y.shape}")
            if i >= 2:
                break
        print("✅ Generator works correctly!")
        return True
    except Exception as e:
        print(f"❌ Generator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =========================================================
# Updated main execution block
# =========================================================
if __name__ == "__main__":
    
    # =========================================================
    #  Setup multiprocessing and GPU
    # =========================================================
    mp.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()
    
    # =========================================================
    #  Graceful shutdown handler
    # =========================================================
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal. Exiting gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # =========================================================
    #  Paths
    # =========================================================
    load_path = '/storage/mxg1065/datafiles'
    save_dir = "/storage/mxg1065/fixed_batch_size_models"
    
    # =========================================================
    #  Step 1: Check new format files
    # =========================================================
    print("🔍 Step 1: Checking new format files...")
    files_ok = quick_data_check(load_path)
    if not files_ok:
        print("❌ Critical: Required new-format files missing!")
        print("   Please run dataset creation script first.")
        sys.exit(1)
    
    # =========================================================
    #  Step 2: Test data integrity (new format)
    # =========================================================
    print("\n🔍 Step 2: Testing data integrity (new format)...")
    integrity_ok = test_data_integrity(load_path)
    
    if not integrity_ok:
        response = input("⚠️  Data integrity tests failed. Continue training anyway? (y/n): ").strip().lower()
        if response not in ["y", "yes"]:
            print("Training aborted due to data issues.")
            sys.exit(1)
        else:
            print("⚠️  Continuing training despite potential data issues...")
    else:
        print("✅ New format data integrity verified!")
    
    # =========================================================
    #  Step 3: Test generator compatibility
    # =========================================================
    print("\n🧪 Step 3: Testing generator compatibility...")
    if not test_generator_compatibility(load_path):
        response = input("❌ Generator test failed. Continue anyway? (y/n): ").strip().lower()
        if response not in ["y", "yes"]:
            sys.exit(1)
    else:
        print("✅ Generator works correctly!")
    
    # =========================================================
    #  Step 4: Start multi-GPU training
    # =========================================================
    print("\n🎯 Step 4: Starting multi-GPU training...")
    print("="*50)
    try:
        main()  # Your training workflow (uses new format)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # =========================================================
    #  Step 5: Post-training verification
    # =========================================================
    print("\n" + "="*50)
    print("📊 Step 5: Post-training verification")
    print("="*50)
    time.sleep(2)  # Wait for files to be fully written
    
    # Find and verify parquet file
    parquet_files = sorted(glob.glob(os.path.join(save_dir, "comprehensive_results_*.parquet")))
    if parquet_files:
        parquet_path = parquet_files[-1]  # last created file
        print(f"\n🔍 Verifying order preservation for: {os.path.basename(parquet_path)}")
        try:
            order_ok = verify_order_preservation(load_path, parquet_path)
            if order_ok:
                print("✅ ORDER PRESERVATION VERIFIED SUCCESSFULLY!")
            else:
                print("❌ ORDER PRESERVATION FAILED!")
        except Exception as e:
            print(f"⚠️ Could not verify order: {e}")
    else:
        print(f"⚠️ No comprehensive parquet file found in {save_dir}")
    
    print("\n" + "="*50)
    print("🎉 TRAINING PIPELINE COMPLETE!")
    print("="*50)