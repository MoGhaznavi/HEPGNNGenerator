import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class MultiClassBatchGenerator(IterableDataset):
    def __init__(self, features_dict, neighbor_pairs, labels, class_counts, 
                 mode='train', is_bi_directional=True, batch_size=1, train_ratio=0.7,
                 padding=False, with_labels=False):
        self.features_dict = {
            k: torch.as_tensor(v, dtype=torch.float32)
            for k, v in features_dict.items()
        }
        self.neighbor_pairs = torch.tensor(neighbor_pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.class_counts = class_counts
        self.is_bi_directional = is_bi_directional
        self.batch_size = batch_size
        self.padding = padding
        self.with_labels = with_labels

        num_events = len(features_dict)
        train_size = int(num_events * train_ratio)
        self.event_indices = (
            list(range(train_size)) if mode == 'train' 
            else list(range(train_size, num_events)))

        self._build_class_indices()

    def _build_class_indices(self):
        self.class_indices = {}
        for event_id in range(len(self.labels)):
            self.class_indices[event_id] = {
                cls: torch.where(self.labels[event_id] == cls)[0]
                for cls in self.class_counts.keys()
            }

    def _sample_edges(self, event_id):
        sampled_indices = []
        padded_mask = []

        for cls, count in self.class_counts.items():
            indices = self.class_indices[event_id][cls]
            if len(indices) == 0:
                continue
            selected = indices[torch.randperm(len(indices))[:min(count, len(indices))]]
            sampled_indices.append(selected)
            padded_mask.extend([True] * len(selected))

        sampled = torch.cat(sampled_indices) if sampled_indices else torch.tensor([], dtype=torch.long)

        if self.padding and len(sampled) < sum(self.class_counts.values()):
            pad_size = sum(self.class_counts.values()) - len(sampled)
            pad_indices = self.class_indices[event_id][0][:pad_size]
            sampled = torch.cat([sampled, pad_indices])
            padded_mask.extend([False] * len(pad_indices))

        return sampled, torch.tensor(padded_mask)

    def __iter__(self):
        for event_id in self.event_indices:
            edge_sample_idx, padded_mask = self._sample_edges(event_id)
            selected_pairs = self.neighbor_pairs[edge_sample_idx].T
            selected_labels = self.labels[event_id, edge_sample_idx].clone()

            if self.padding and not self.with_labels:
                selected_labels[~padded_mask] = 4

            edge_index = selected_pairs.clone()
            if self.is_bi_directional:
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
                edge_index, _ = add_self_loops(edge_index)

            yield Data(
                x=self.features_dict[f"data_{event_id}"],
                edge_index=edge_index,
                edge_index_out=selected_pairs,
                y=selected_labels.unsqueeze(1)
            )

    @staticmethod
    def collate_data(batch):
        node_features_list = [torch.as_tensor(data.x, dtype=torch.float32) for data in batch]
        processed_edge_index_list = [torch.as_tensor(data.edge_index, dtype=torch.long) for data in batch]
        original_edge_index_list = [torch.as_tensor(data.edge_index_out, dtype=torch.long) for data in batch]
        concatenated_edge_labels = torch.cat([torch.as_tensor(data.y, dtype=torch.long) for data in batch], dim=0)
        
        return node_features_list, processed_edge_index_list, original_edge_index_list, concatenated_edge_labels