# MultiClassBatchGenerator

The `MultiClassBatchGenerator` is a custom PyTorch `IterableDataset` designed to generate edge samples from event-based graph data. It is primarily used in edge classification tasks, where the goal is to predict labels on edges between nodes (e.g., physical connections, relationships, interactions).

## How It Works

### Step-by-Step Pipeline

1. Initialization
- Converts input node features and labels to PyTorch tensors.
- Partitions events into train and test sets using `train_ratio`.
- Builds per-event index maps for each class to enable fast sampling.

2. Sampling Edges per Event

- For each event:
    - Randomly samples a specified number of edges per class from the label matrix.
    - Ensures equal representation for each class as defined in `class_counts`.
    - If padding is enabled, fills under-sampled classes with dummy edges.
    - Optionally flags padded edges to ignore their loss contribution.

3. Edge Augmentation
- Adds reverse edges to ensure the graph is bi-directional, if enabled.
- Adds self-loops for each node using `add_self_loops()`.

4. Output Format
- Returns a PyTorch Geometric (PyG) `Data` object per event containing:
    - `x`: node features for that event.
    - `edge_index`: full augmented edge list.
    - `edge_index_out`: original forward edges before augmentation.
    - `y`: labels for each edge (including padding if applicable).

## Arguments and their Explinations

```python
MultiClassBatchGenerator(
    features_dict,        # dict of node features for each event
    neighbor_pairs,       # global edge template (e.g., (i, j) pairs)
    labels,               # edge labels for each event
    class_counts,         # how many edges to sample per class
    mode='train',         # 'train' or 'test'
    is_bi_directional=True,
    batch_size=1,         # reserved for future use
    train_ratio=0.7,      # split events into training/testing sets
    padding=False,        # pad to match total class sample size
    with_labels=False     # mark padded edges with dummy label (e.g., 4)
)
```

| Argument            | Description                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `features_dict`     | Dict mapping event IDs (`"data_0"`, `"data_1"`, ...) to node features (shape `[num_nodes, num_features]`) |
| `neighbor_pairs`    | Shared list of candidate edges (shape `[num_edges, 2]`)                                                   |
| `labels`            | Matrix of shape `[num_events, num_edges]` with integer labels                                             |
| `class_counts`      | Dict like `{0: 50, 1: 50, 2: 50}` defining how many edges to sample from each class                       |
| `mode`              | Defines if the generator samples from train or test events                                                |
| `train_ratio`       | Fraction of events used for training (default = 0.7)                                                      |
| `padding`           | Adds dummy edges if an event lacks enough samples for a class                                             |
| `with_labels`       | If `False`, assigns label `4` to padded edges to avoid loss impact                                        |
| `is_bi_directional` | Adds reversed edges and self-loops if `True`                                                              |


## Key Methods

### `__iter__(self)`

Main method that yields PyG `Data` objects event-by-event.

### `_sample_edges(self, event_id)`

Samples a fixed number of edges from each class for a given event. Handles padding if needed.

### `_build_class_indices(self)`

Builds a lookup dictionary for each event that maps class labels to edge indices.

### `collate_data(batch)`

Static method for combining multiple `Data` objects into batched tensors.

## Output Example

A single iteration yields a PyG `Data` object:

```python
Data(
  x: Tensor [num_nodes, num_features],
  edge_index: Tensor [2, num_edges * 2 + num_nodes],  # if bi-directional + self-loops
  edge_index_out: Tensor [2, num_sampled_edges],
  y: Tensor [num_sampled_edges, 1]  # labels
)
```

## Note on Reproducibility

The class sets seeds for `random`, `numpy`, and `torch` (CPU + CUDA) for deterministic sampling:

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

## Collating Batches

To prepare multiple events for processing in one step, use:

```python
batch = [data1, data2, ...]
X_list, edge_index_list, edge_raw_list, labels = MultiClassBatchGenerator.collate_data(batch)
```

This returns:

- A list of node features per event
- Processed edge indices (with augmentation)
- Original sampled edge indices (before flip/self-loops)
- Concatenated edge labels
