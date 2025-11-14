'''
For training with auto-conversion:
python3 RunningMultipleModels.py

For converting existing results
python3 RunningMultipleModels.py convert path/to/results.pkl
'''


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

# --- PKL to DataFrame/Tensor Conversion ---------------------------------------

def convert_results_to_dataframe(
    pkl_filepath: str, 
    max_events: Optional[int] = None, 
    chunk_size: int = 5,
    output_dir: Optional[str] = None,  # Add output_dir parameter
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convert the results pickle file to DataFrame and NumPy tensor.
    Memory-optimized version that combines everything at the end.
    """
    if verbose:
        print(f"Converting results: {pkl_filepath}")
    
    # FIX: Handle output directory
    if output_dir is None:
        # If pkl_filepath is absolute, use its directory
        if os.path.isabs(pkl_filepath):
            output_dir = os.path.dirname(pkl_filepath)
        else:
            # If relative, use current working directory
            output_dir = os.getcwd()
    
    # FIX: Ensure output_dir is never empty
    if not output_dir:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(pkl_filepath, "rb") as f:
        data = pickle.load(f)

    event_data = data["final_results_per_event"]
    n_events_max_all = len(event_data)
    n_events_max = n_events_max_all if max_events is None else min(max_events, n_events_max_all)

    if verbose:
        print(f"Processing {n_events_max} events in chunks of {chunk_size}")
        print(f"Output directory: {output_dir}")
    
    # CORRECTED COLUMN NAMES FOR 19 COLUMNS:
    column_names = [
        "event_id", "n_pairs", "preds", 
        "score_0", "score_1", "score_2", "score_3", "score_4",
        "labels", "neighbor_pairs_0", "neighbor_pairs_1",
        "e_i", "eta_i", "phi_i", "e_j", "eta_j", "phi_j",
        "eta_i_unscaled", "eta_j_unscaled"
    ]

    # Use lists instead of accumulating large arrays
    all_event_lengths = []
    temp_chunk_files = []
    total_rows = 0
    
    # First pass: calculate statistics without storing data
    if verbose:
        print("Calculating statistics...")
    
    n_pairs_max = 0
    for i in range(0, n_events_max, 50):  # Sample efficiently
        end_idx = min(i + 50, n_events_max)
        chunk_events = event_data[i:end_idx]
        chunk_max = int(np.max([ev["preds"].shape[0] for ev in chunk_events])) if chunk_events else 0
        n_pairs_max = max(n_pairs_max, chunk_max)
    
    if verbose:
        print(f"Global maximum pairs per event: {n_pairs_max}")

    # Process data in small chunks and save to temporary files
    for start_idx in range(0, n_events_max, chunk_size):
        end_idx = min(start_idx + chunk_size, n_events_max)
        chunk_events = event_data[start_idx:end_idx]
        chunk_size_actual = len(chunk_events)
        
        if verbose and start_idx % 50 == 0:  # Less verbose output
            print(f"Processing events {start_idx}-{end_idx}...")

        chunk_acc = []
        event_lengths_chunk = []
        
        # Process each event in this chunk
        for chunk_idx, event in enumerate(chunk_events):
            n_pairs_evt = event["preds"].shape[0]
            event_lengths_chunk.append(n_pairs_evt)
            
            for i_pair in range(n_pairs_evt):
                # Build data row
                row_data = [
                    event["event_id"],
                    n_pairs_evt,
                    int(event["preds"][i_pair]),
                    *event["scores"][i_pair].tolist(),
                    event["labels"][i_pair],
                    *event["neighbor_pairs"][i_pair].tolist(),
                    *event["features_i_unscaled"][i_pair].tolist(),
                    *event["features_j_unscaled"][i_pair].tolist(),
                    event["eta_i"][i_pair],
                    event["eta_j"][i_pair],
                ]

                # Ensure exactly 19 columns
                if len(row_data) != 19:
                    if len(row_data) < 19:
                        row_data.extend([0] * (19 - len(row_data)))
                    else:
                        row_data = row_data[:19]

                chunk_acc.append(row_data)

        # Save chunk to temporary file and track metadata
        if chunk_acc:
            df_chunk = pd.DataFrame(chunk_acc, columns=column_names)
            
            # Use more efficient data types to reduce memory
            for col in df_chunk.columns:
                if df_chunk[col].dtype == 'float64':
                    df_chunk[col] = df_chunk[col].astype('float32')
                elif df_chunk[col].dtype == 'int64':
                    df_chunk[col] = df_chunk[col].astype('int32')
            
            # FIX: Use the provided output_dir for temp files
            temp_file = os.path.join(output_dir, f"temp_chunk_{start_idx:06d}.parquet")
            df_chunk.to_parquet(temp_file, index=False)
            temp_chunk_files.append(temp_file)
            all_event_lengths.extend(event_lengths_chunk)
            total_rows += len(df_chunk)
            
            if verbose and start_idx % 50 == 0:
                print(f"  Saved chunk: {len(df_chunk):,} rows, total: {total_rows:,}")
        
        # Aggressive memory cleanup
        del df_chunk, chunk_acc
        import gc
        gc.collect()

    # Memory-efficient combination of chunks
    if verbose:
        print(f"\nCombining {len(temp_chunk_files)} chunks...")
    
    # Read chunks back in a memory-efficient way
    df_chunks = []
    for i, temp_file in enumerate(temp_chunk_files):
        if verbose and i % 10 == 0:
            print(f"  Reading chunk {i+1}/{len(temp_chunk_files)}...")
        
        df_chunk = pd.read_parquet(temp_file)
        df_chunks.append(df_chunk)
        
        # Clean up temp file
        os.remove(temp_file)
    
    # Combine using efficient concatenation
    if verbose:
        print("Final concatenation...")
    
    df_final = pd.concat(df_chunks, ignore_index=True, copy=False)
    
    # Build tensor in a memory-efficient way
    if verbose:
        print("Building final tensor...")
    
    T_final = np.zeros((len(all_event_lengths), n_pairs_max, 19), dtype=np.float32)
    T_final.fill(np.nan)
    
    # Reconstruct tensor from the final DataFrame (more memory efficient)
    current_event = 0
    event_ptr = 0
    
    for event_length in all_event_lengths:
        if event_length > 0:
            # Get the rows for this event from the final DataFrame
            event_data_slice = df_final.iloc[event_ptr:event_ptr + event_length]
            event_values = event_data_slice.values.astype(np.float32)
            T_final[current_event, :event_length, :] = event_values
            event_ptr += event_length
        current_event += 1

    if verbose:
        print(f"🎉 Conversion complete!")
        print(f"   Final DataFrame: {len(df_final):,} rows × {len(df_final.columns)} columns")
        print(f"   Final Tensor: {T_final.shape}")
        print(f"   Memory usage: {df_final.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"   Events processed: {len(all_event_lengths)}")
        print(f"   Total pairs: {len(df_final):,}")
    
    return df_final, T_final

def _process_single_event_comprehensive(event_result: Dict, model_name: str = None) -> pd.DataFrame:
    """
    Process a single event into a comprehensive DataFrame with derived features.
    Memory-optimized version with optional model name.
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

    # --- Derived features ---
    delta_eta = np.abs(eta_i - eta_j).astype(np.float32)
    delta_phi = np.abs(phi_i - phi_j).astype(np.float32)
    delta_phi = np.minimum(delta_phi, 2*np.pi - delta_phi).astype(np.float32)
    spatial_distance = np.sqrt(delta_eta**2 + delta_phi**2).astype(np.float32)
    snr_ratio = np.divide(snr_i, snr_j, out=np.full_like(snr_i, np.inf, dtype=np.float32), where=snr_j != 0)
    avg_snr = ((snr_i + snr_j) / 2).astype(np.float32)
    snr_sum = (snr_i + snr_j).astype(np.float32)
    snr_product = (snr_i * snr_j).astype(np.float32)

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

def build_querying_table_comprehensive(metrics: Dict, output_dir: str, model_name: str, chunk_size: int = 50) -> str:
    """
    Builds a comprehensive analysis table with derived features from inference results.
    Memory-efficient chunked version with model-specific naming.
    """
    os.makedirs(output_dir, exist_ok=True)
    events = metrics['final_results_per_event']
    total_events = len(events)

    print(f"📊 Building comprehensive query table for {model_name} from {total_events} events (chunked)...")
    print(f"   Chunk size: {chunk_size} events")
    print(f"   Estimated total rows: ~{sum(ev['preds'].shape[0] for ev in events):,}")

    # Process events in chunks and save incrementally
    temp_chunk_files = []
    
    for start_idx in range(0, total_events, chunk_size):
        end_idx = min(start_idx + chunk_size, total_events)
        chunk_events = events[start_idx:end_idx]
        
        print(f"   Processing chunk {start_idx}-{end_idx}...")

        chunk_dfs = []
        
        for event_result in chunk_events:
            df_event = _process_single_event_comprehensive(event_result)
            # Add model name to each event
            df_event['model_name'] = model_name
            chunk_dfs.append(df_event)

        # Combine this chunk and save to temporary file
        if chunk_dfs:
            df_chunk = pd.concat(chunk_dfs, ignore_index=True)
            
            # Optimize memory usage
            for col in df_chunk.columns:
                if df_chunk[col].dtype == 'float64':
                    df_chunk[col] = df_chunk[col].astype('float32')
                elif df_chunk[col].dtype == 'int64':
                    # Check if we can use smaller integer types
                    max_val = df_chunk[col].max()
                    min_val = df_chunk[col].min()
                    if min_val >= 0:
                        if max_val < 256:
                            df_chunk[col] = df_chunk[col].astype('uint8')
                        elif max_val < 65536:
                            df_chunk[col] = df_chunk[col].astype('uint16')
                        else:
                            df_chunk[col] = df_chunk[col].astype('uint32')
                    else:
                        if min_val > -128 and max_val < 127:
                            df_chunk[col] = df_chunk[col].astype('int8')
                        elif min_val > -32768 and max_val < 32767:
                            df_chunk[col] = df_chunk[col].astype('int16')
                        else:
                            df_chunk[col] = df_chunk[col].astype('int32')
            
            temp_file = os.path.join(output_dir, f"temp_{model_name}_chunk_{start_idx:06d}.parquet")
            df_chunk.to_parquet(temp_file, index=False)
            temp_chunk_files.append(temp_file)
            
            print(f"     ✅ Saved chunk: {len(df_chunk):,} rows, memory: {df_chunk.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Clean up
            del df_chunk, chunk_dfs
            import gc
            gc.collect()

    # Now combine all chunks efficiently
    print("   Combining all chunks...")
    
    # Read chunks back in sequence
    final_dfs = []
    for i, temp_file in enumerate(temp_chunk_files):
        if i % 10 == 0:
            print(f"     Reading chunk {i+1}/{len(temp_chunk_files)}...")
        
        df_chunk = pd.read_parquet(temp_file)
        final_dfs.append(df_chunk)
        
        # Clean up temp file
        os.remove(temp_file)
    
    # Final combination
    print("   Final concatenation...")
    df_comprehensive = pd.concat(final_dfs, ignore_index=True)
    
    # Save final comprehensive file with model-specific name
    output_path = os.path.join(output_dir, f"comprehensive_query_table_{model_name}.parquet")
    df_comprehensive.to_parquet(output_path, index=False)
    
    print(f"💾 Saved comprehensive table: {output_path}")
    print(f"   - Total rows: {len(df_comprehensive):,}")
    print(f"   - Total events: {total_events}")
    print(f"   - Columns: {len(df_comprehensive.columns)}")
    print(f"   - Final memory usage: {df_comprehensive.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return output_path

def save_conversion_results(
    pkl_filepath: str,
    save_df: bool = True,
    save_npy: bool = True,
    save_comprehensive: bool = True,
    max_events: Optional[int] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    comprehensive_chunk_size: int = 50
) -> Dict[str, str]:
    """
    Enhanced conversion function with model-specific comprehensive table naming.
    Handles both absolute and relative paths correctly.
    """
    # FIX: Handle relative paths properly
    if output_dir is None:
        # If pkl_filepath is absolute, use its directory
        if os.path.isabs(pkl_filepath):
            output_dir = os.path.dirname(pkl_filepath)
        else:
            # If relative, use current working directory
            output_dir = os.getcwd()
    
    # FIX: Ensure output_dir is never empty
    if not output_dir:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # FIX: Extract base_name safely
    if os.path.isabs(pkl_filepath):
        base_name = os.path.splitext(os.path.basename(pkl_filepath))[0]
    else:
        base_name = os.path.splitext(pkl_filepath)[0]
    
    # Just use the entire filename (without .pkl)
    model_name = base_name  # "12_layer_model", "1_layer_model", etc.
    
    saved_paths = {}
    
    try:
        # Load the metrics data
        with open(pkl_filepath, "rb") as f:
            metrics_data = pickle.load(f)

        # 1. Save raw format (original conversion)
        if save_df or save_npy:
            # FIX: Pass output_dir to convert_results_to_dataframe
            df_raw, T_pairs = convert_results_to_dataframe(
                pkl_filepath, 
                max_events, 
                output_dir=output_dir,  # Pass the output_dir
                verbose=verbose
            )
            
            if save_df:
                parquet_path = os.path.join(output_dir, f"{base_name}_raw_pairs.parquet")
                df_raw.to_parquet(parquet_path, index=False)
                saved_paths['raw_dataframe'] = parquet_path
                if verbose:
                    print(f"✓ Saved raw DataFrame: {parquet_path}")
            
            if save_npy:
                npy_path = os.path.join(output_dir, f"{base_name}_T_pairs.npy")
                np.save(npy_path, T_pairs)
                saved_paths['tensor'] = npy_path
                if verbose:
                    print(f"✓ Saved NumPy tensor: {npy_path}")

        # 2. Save comprehensive analysis format (MODEL-SPECIFIC)
        if save_comprehensive and 'final_results_per_event' in metrics_data:
            comprehensive_path = build_querying_table_comprehensive(
                metrics=metrics_data,
                output_dir=output_dir,
                model_name=model_name,  # Pass model name
                chunk_size=comprehensive_chunk_size
            )
            saved_paths['comprehensive_table'] = comprehensive_path
            if verbose:
                print(f"✓ Saved comprehensive table: {comprehensive_path}")
                
    except Exception as e:
        print(f"✗ Conversion failed for {pkl_filepath}: {e}")
        raise
    
    return saved_paths

def convert_existing_results():
    """
    Standalone function to convert existing PKL files.
    Can be called separately if needed.
    """
    print("DEBUG: Starting convert_existing_results()")
    print(f"DEBUG: sys.argv = {sys.argv}")
    print(f"DEBUG: Current directory: {os.getcwd()}")
    
    parser = argparse.ArgumentParser(description="Convert training results PKL to DataFrame and NumPy formats")
    parser.add_argument("pkl_file", help="Path to input .pkl file")
    parser.add_argument("--max-events", type=int, default=None, help="Max events to process")
    parser.add_argument("--output-dir", help="Output directory (default: current directory)")
    parser.add_argument("--no-df", action="store_true", help="Skip DataFrame output")
    parser.add_argument("--no-npy", action="store_true", help="Skip NumPy tensor output")
    parser.add_argument("--no-comprehensive", action="store_true", help="Skip comprehensive table output")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    
    args = parser.parse_args()

    print(f"DEBUG: args.pkl_file = {args.pkl_file}")
    print(f"DEBUG: args.output_dir = {args.output_dir}")
    print(f"DEBUG: File exists: {os.path.exists(args.pkl_file)}")

    # FIX: Provide explicit output directory if not provided
    if args.output_dir is None:
        args.output_dir = os.getcwd()  # Use current directory
        print(f"DEBUG: Using output directory: {args.output_dir}")

    # FIX: Check if file exists
    if not os.path.exists(args.pkl_file):
        print(f"❌ Error: File not found: {args.pkl_file}")
        print(f"❌ Current directory: {os.getcwd()}")
        sys.exit(1)

    save_df = not args.no_df
    save_npy = not args.no_npy
    save_comprehensive = not args.no_comprehensive

    if not save_df and not save_npy and not save_comprehensive:
        print("Warning: All output formats disabled, nothing to save!")
        return

    try:
        saved_paths = save_conversion_results(
            pkl_filepath=args.pkl_file,
            save_df=save_df,
            save_npy=save_npy,
            save_comprehensive=save_comprehensive,
            max_events=args.max_events,
            output_dir=args.output_dir,  # Now explicitly provided
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n🎉 Conversion completed successfully!")
            for file_type, path in saved_paths.items():
                print(f"   {file_type}: {path}")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Dataset: Generates balanced graph-edge batches for multi-class classification
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

#     # ----- Helpers -------------------------------------------------------------

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
#         for idx, event_id in enumerate(self.event_indices):
#             # Try to get the key - handle both integer and string formats
#             event_key = event_id
#             if event_key not in self.features_dict:
#                 # Try string format
#                 event_key = f"data_{event_id}"
#                 if event_key not in self.features_dict:
#                     raise KeyError(f"Event ID {event_id} not found in features_dict")
            
#             # Use all neighbor pairs for this event
#             edge_idx = torch.arange(self.neighbor_pairs.shape[0])
#             out_labels = self.labels[idx][edge_idx]  # Use idx instead of event_id for labels
    
#             pairs = self.neighbor_pairs[edge_idx].T
#             x_scaled = self.features_dict[event_key]
#             x_unscaled = (self.unscaled_features_dict.get(event_key)
#                           if self.unscaled_features_dict else None)
    
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

class MultiClassBatchGenerator(IterableDataset):
    """
    Optimized IterableDataset for large events (~1.2M edges):
      • Uses all neighbor pairs per event
      • Supports unscaled features for analysis
      • Simplified, memory-efficient design
    """

    def __init__(
        self,
        features_dict: Dict[int, np.ndarray],
        neighbor_pairs: np.ndarray,
        labels: np.ndarray,
        mode: str = "train",
        is_bi_directional: bool = True,
        batch_size: int = 1,
        train_ratio: float = 0.7,
        debug: bool = False,
        unscaled_data_dict: Optional[Dict[int, np.ndarray]] = None,
    ):
        # Config
        self.debug = debug
        self.is_bi_directional = is_bi_directional
        self.batch_size = batch_size

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

        # Train/test split
        num_events = len(features_dict)
        split_idx = int(num_events * train_ratio)
        self.event_indices = (list(range(split_idx)) if mode == "train"
                              else list(range(split_idx, num_events)))

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
        Handle both integer and string keys.
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
            
            # Use all neighbor pairs for this event with random shuffling
            num_pairs = self.neighbor_pairs.shape[0]
            shuffled_indices = torch.randperm(num_pairs)  # Random permutation like old generator
            
            pairs = self.neighbor_pairs[shuffled_indices].T
            x_scaled = self.features_dict[event_key]
            x_unscaled = (self.unscaled_features_dict.get(event_key)
                          if self.unscaled_features_dict else None)
            
            # FIX: Use event_id for label indexing like old generator
            out_labels = self.labels[event_id][shuffled_indices]  # Use event_id instead of idx

            samples.append((x_scaled, pairs, pairs.clone(), out_labels.unsqueeze(1), x_unscaled))
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
              delta: float = 0.0001, debug: bool = False,
              # Enhanced conversion parameters
              auto_convert_results: bool = True,
              save_comprehensive_table: bool = True) -> Tuple[Dict, nn.Module, str]:
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

        # Save final metrics to disk
        save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)
        total_min, total_sec = divmod(metrics['total_time'], 60)
        print(f"\nTraining complete in {int(total_min)}m {total_sec:.1f}s")
        print(f"Best model at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")

    # --- AFTER TRAINING COMPLETES: Auto-convert results ---
    if not debug and auto_convert_results:
        try:
            print(f"\n{'='*50}")
            print("AUTO-CONVERTING RESULTS...")
            print(f"{'='*50}")
            
            # The metrics file contains the final results
            if os.path.exists(metrics_path):
                saved_paths = save_conversion_results(
                    pkl_filepath=metrics_path,
                    save_df=True,
                    save_npy=True,
                    save_comprehensive=save_comprehensive_table,
                    output_dir=save_dir,  # Save in same directory as model
                    verbose=True
                )
                
                # Add conversion info to metrics
                final_metrics['conversion_paths'] = saved_paths
                print("✅ All results converted successfully!")
                
                # Display summary
                for file_type, path in saved_paths.items():
                    print(f"   📁 {file_type}: {path}")
                    
            else:
                print("⚠ Metrics file not found for conversion")
                
        except Exception as e:
            print(f"⚠ Results conversion failed: {e}")
            # Don't crash the whole training if conversion fails
    
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
            delta=config.get('delta', 0.0001),
            auto_convert_results=config.get('auto_convert_results', True),
            save_comprehensive_table=config.get('save_comprehensive_table', True)
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
        'auto_convert_results': False,
        'save_comprehensive_table': True,
        'generator_flags': {  # defaults for the data generator
            'is_bi_directional': True,
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
        # GPU 0
        {**base_config,
        'model_name': "fixed_generator_bs2_model.pt",
        'description': "Model with two events per batch",
        'batch_size': 2}
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
        print(f"      → Auto-convert: {config['auto_convert_results']}")

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
        print("\nReceived shutdown signal. Exiting gracefully...")
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if we're running in conversion mode or training mode
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        # Remove 'convert' argument and call conversion function
        sys.argv.pop(1)  # Remove the 'convert' argument
        convert_existing_results()
    else:
        # Kick off the multi-GPU training workflow
        main()
