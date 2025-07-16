import matplotlib.pyplot as plt
from metrics.analysis import load_metrics, plot_seed_and_aggregate, plot_individual_class_overlayed
from config.plot_config import plot_config

# === Unpack config ===
load_path = plot_config['load_path']
target_tpr = plot_config['target_tpr']
layer_val = plot_config['layer_val']
model_groups = plot_config['model_groups']
class_names = plot_config['class_names']
colors = plot_config['colors']

# === Store Results ===
individual_dfs = {}
aggregate_dfs = {}

# === Loop through model groups and plot ===
for label, file_list in model_groups.items():
    print(f"\n--- Processing: {label} ---")
    df_indiv, df_agg = load_metrics(file_list, layer_val, load_path, target_tpr)
    individual_dfs[label] = df_indiv
    aggregate_dfs[label] = df_agg

    plot_seed_and_aggregate(df_indiv, df_agg, f"({label})")

# === Overlay if exactly 2 model groups ===
if len(model_groups) == 2:
    labels = list(model_groups.keys())
    plot_individual_class_overlayed(
        individual_dfs[labels[0]], individual_dfs[labels[1]],
        aggregate_dfs[labels[0]], aggregate_dfs[labels[1]]
    )
else:
    print("Skipping overlay: only supports exactly 2 groups.")