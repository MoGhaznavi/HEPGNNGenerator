# MultiEdgeClassifier

The `MultiEdgeClassifier` is a deep graph-based model built using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), designed specifically for **edge classification** tasks across multiple events (graphs). It leverages multi-layer GCNs and pairwise node embeddings to learn edge-level representations.

## Architecture Overview

```python
MultiEdgeClassifier(
    input_dim,      # Number of input node features
    hidden_dim,     # Hidden dimensionality for embeddings
    output_dim,     # Number of edge classes (e.g., 3 or 4)
    device,         # Target computation device (e.g., "cuda")
    num_layers=12,  # Number of GCNConv layers
    layer_weights=True  # Optional: learnable weights per layer
)
```

### Key Components

* **Linear Input Projection**
  Projects input node features to the hidden dimension:

  ```python
  self.node_embedding = nn.Linear(input_dim, hidden_dim)
  ```

* **Stacked GCNConv Layers**
  A sequence of Graph Convolutional layers with batch normalization and residual connections:

  ```python
  self.convs = nn.ModuleList([...])
  self.bns = nn.ModuleList([...])
  ```

* **Edge Representation Construction**
  For each edge, concatenate the final embeddings of the source and destination nodes:

  ```python
  edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
  ```

* **Final Classifier**
  A linear classifier outputs edge-level predictions:

  ```python
  self.fc = nn.Linear(2 * hidden_dim, output_dim)
  ```

## Forward Pass Logic

```python
output = model(x_list, edge_index_list, edge_index_out_list)
```

### Inputs

| Argument              | Type              | Description                                                      |
| --------------------- | ----------------- | ---------------------------------------------------------------- |
| `x_list`              | List\[Tensor]     | Node feature tensors (1 per event)                               |
| `edge_index_list`     | List\[Tensor]     | Processed edge indices (bi-directional, with self-loops)         |
| `edge_index_out_list` | List\[Tensor]     | Original (forward) edge pairs sampled from the generator         |
| `y_batch`             | Optional\[Tensor] | Ground-truth labels for loss calculation (not used in `forward`) |

### Flow

1. For each event:

   * Move input tensors to device.
   * Project node features using a linear layer.
   * Apply multiple GCN layers with residual connections.
   * Extract embeddings for each source-destination edge pair.
2. Concatenate all event-wise edge representations.
3. Pass them through a fully connected layer to produce logits.

### Output

```python
Tensor of shape [total_edges_across_events, output_dim]
```

## Model Features

| Feature                            | Description                                                            |
| ---------------------------------- | ---------------------------------------------------------------------- |
| Deep GCN Stack                     | 12 GCNConv layers with residual updates and batch normalization        |
| Event-wise Parallelism             | Efficient batching over multiple graphs (events)                       |
| Learnable Layer Weights (Optional) | Future-ready for weighted layer fusion (e.g., for deep supervision)    |
| Rich Edge Embeddings               | Builds edge representations from concatenated source/destination nodes |

## Design Decisions

* **Node-to-Edge Transformation**
  Edge classification relies on good node embeddings. This model learns them through GCNs, then combines node pairs with `concat`.

* **Residual Updates**
  Each GCN layer uses residual addition (`x_embed = x_embed + ...`) to combat oversmoothing and preserve early-layer features.

* **BatchNorm**
  Batch normalization is used per layer to stabilize training in deep GCNs.

* **Device-Aware**
  All tensors are explicitly moved to the assigned device for efficient CUDA training.


## Notes on Reproducibility

All random seeds are set globally in the module:

```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```

This ensures consistent behavior across training runs.
