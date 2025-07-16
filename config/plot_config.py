import torch

plot_config = {
    # === Paths ===
    'load_path': "/storage/mxg1065/all_pairs_models",

    # === Analysis Parameters ===
    'target_tpr': 0.99,
    'layer_val': 6,

    # === Class Labels and Colors ===
    'class_names': ['Lone-Lone', 'True-True', 'Cluster-Lone', 'Lone-Cluster', 'Cluster-Cluster'],
    'colors': ['blue', 'orange', 'green', 'red', 'purple'],

    # === Model Groups for Comparison ===
    'model_groups': {
        'No Weights': [f"all_pairs_seed{i}.pkl" for i in range(6)],
        'With Weights': [f"all_pairs_with_weights_seed{i}.pkl" for i in range(6)],
        # One can add more groups here
    }
}
