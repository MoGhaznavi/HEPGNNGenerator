import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class MultiEdgeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, 
                 num_layers=12, layer_weights=True):
        super().__init__()
        self.device = device

        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

        self.layer_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ]) if layer_weights else None

    def forward(self, x_list, edge_index_list, edge_index_out_list, y_batch=None):
        all_edge_reprs = []
        for x, processed_edges, original_edges in zip(x_list, edge_index_list, edge_index_out_list):
            x = x.to(self.device, non_blocking=True)
            processed_edges = processed_edges.to(self.device, non_blocking=True)
            x_embed = self.node_embedding(x)
            for conv, bn in zip(self.convs, self.bns):
                x_embed = x_embed + torch.relu(bn(conv(x_embed, processed_edges)))
            src, dst = original_edges[0], original_edges[1]
            edge_repr = torch.cat([x_embed[src], x_embed[dst]], dim=-1)
            all_edge_reprs.append(edge_repr)
        return self.fc(torch.cat(all_edge_reprs, dim=0))