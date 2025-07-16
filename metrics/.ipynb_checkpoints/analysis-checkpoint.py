import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from config.plot_config import plot_config

class_names = plot_config['class_names']
colors = plot_config['colors']
num_classes = len(class_names)


def to_softmax(logits):
    logits = np.asarray(logits)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def load_data_pickle(file_name, load_path):
    full_path = os.path.join(load_path, file_name)
    with open(full_path, 'rb') as file:
        data_dict = pickle.load(file)
    print(f"Data successfully loaded from {full_path}")
    return data_dict

def load_metrics(file_list, layer_val, load_path, target_tpr=0.99):
    all_stats = defaultdict(list)

    for file_name in file_list:
        metrics = load_data_pickle(file_name, load_path)
        y_true = np.ravel(metrics['final_test_labels'])
        y_score = metrics['final_test_scores']
        y_score = np.vstack(y_score) if isinstance(y_score, list) else y_score

        true_true_mask = (y_true == 1)
        y_true_tt = y_true[true_true_mask]
        y_score_tt = y_score[true_true_mask, 1]

        fpr, tpr, thresholds = roc_curve(y_true_tt == 1, y_score_tt)
        try:
            threshold_99 = thresholds[np.where(tpr >= target_tpr)[0][0]]
        except IndexError:
            threshold_99 = 0.5

        print(f"[{file_name}] Threshold for {target_tpr*100:.1f}% TPR (True-True): {threshold_99:.4f}")

        pred = np.full_like(y_true, fill_value=-1)
        true_true_high = y_score[:, 1] >= threshold_99
        pred[true_true_high] = 1

        for i in range(len(y_true)):
            if pred[i] == -1:
                other_class = np.argmax(np.delete(y_score[i], 1))
                pred[i] = other_class if other_class < 1 else other_class + 1

        cm = confusion_matrix(y_true, pred, labels=range(num_classes))
        total_samples = cm.sum(axis=1)
        recalls = np.diag(cm) / total_samples
        recall_errors = np.sqrt(recalls * (1 - recalls) / total_samples)

        fpr_vals, fpr_errors = [], []
        for cls in range(num_classes):
            FP = cm[:, cls].sum() - cm[cls, cls]
            TP = cm[cls, cls]
            FN = cm[cls].sum() - TP
            TN = cm.sum() - TP - FP - FN
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
            fpr_err = np.sqrt(fpr * (1 - fpr) / (FP + TN)) if (FP + TN) > 0 else 0
            fpr_vals.append(fpr)
            fpr_errors.append(fpr_err)

        seed = int(file_name.split('seed')[-1].split('.')[0])
        for cls in range(num_classes):
            all_stats['Seed'].append(seed)
            all_stats['Layer'].append(layer_val)
            all_stats['Class'].append(class_names[cls])
            all_stats['Recall (%)'].append(recalls[cls] * 100)
            all_stats['Recall Adj Std Error (%)'].append(recall_errors[cls] * 100)
            all_stats['FPR (%)'].append(fpr_vals[cls] * 100)
            all_stats['FPR Adj Std Error (%)'].append(fpr_errors[cls] * 100)

    df_indiv = pd.DataFrame(all_stats)
    grouped = df_indiv.groupby(['Layer', 'Class'])
    df_agg = grouped.mean().reset_index()
    df_std = grouped.std().reset_index()
    df_agg['Recall Std (%)'] = df_std['Recall (%)']
    df_agg['FPR Std (%)'] = df_std['FPR (%)']
    return df_indiv, df_agg

def aggregate_across_seeds(df_indiv):
    grouped = df_indiv.groupby(['Layer', 'Class'])
    df_mean = grouped.mean().reset_index()
    df_std = grouped.std().reset_index()
    df_mean['Recall Std (%)'] = df_std['Recall (%)']
    df_mean['FPR Std (%)'] = df_std['FPR (%)']
    return df_mean

def plot_seed_and_aggregate(df_indiv, df_agg, title_suffix):
    fig, (ax_recall, ax_fpr) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    x_seeds = sorted(df_indiv['Seed'].unique())

    for i, class_name in enumerate(class_names):
        class_df_indiv = df_indiv[df_indiv['Class'] == class_name].sort_values('Seed')
        class_df_agg = df_agg[df_agg['Class'] == class_name]

        # Plot individual seed points with connecting lines and error bars
        ax_recall.plot(
            class_df_indiv['Seed'], class_df_indiv['Recall (%)'],
            marker='o', color=colors[i], alpha=0.7,
            label=f'{class_name} individual seeds'
        )
        ax_recall.errorbar(
            class_df_indiv['Seed'], class_df_indiv['Recall (%)'],
            yerr=class_df_indiv['Recall Adj Std Error (%)'],
            fmt='none', ecolor=colors[i], alpha=0.7
        )

        ax_fpr.plot(
            class_df_indiv['Seed'], class_df_indiv['FPR (%)'],
            marker='o', color=colors[i], alpha=0.7,
            label=f'{class_name} individual seeds'
        )
        ax_fpr.errorbar(
            class_df_indiv['Seed'], class_df_indiv['FPR (%)'],
            yerr=class_df_indiv['FPR Adj Std Error (%)'],
            fmt='none', ecolor=colors[i], alpha=0.7
        )

        # Plot aggregate mean as star, with std as shaded band
        mean_rec = class_df_agg['Recall (%)'].values[0]
        std_rec = class_df_agg['Recall Std (%)'].values[0]
        mean_fpr = class_df_agg['FPR (%)'].values[0]
        std_fpr = class_df_agg['FPR Std (%)'].values[0]

        x_agg = max(x_seeds) + 0.5

        ax_recall.errorbar(
            x_agg, mean_rec, yerr=std_rec,
            fmt='*', color=colors[i], markersize=15, markeredgecolor='black',
            label=f'{class_name} mean ± std'
        )
        ax_recall.fill_between([x_agg - 0.2, x_agg + 0.2], mean_rec - std_rec, mean_rec + std_rec, color=colors[i], alpha=0.15)

        ax_fpr.errorbar(
            x_agg, mean_fpr, yerr=std_fpr,
            fmt='*', color=colors[i], markersize=15, markeredgecolor='black',
            label=f'{class_name} mean ± std'
        )
        ax_fpr.fill_between([x_agg - 0.2, x_agg + 0.2], mean_fpr - std_fpr, mean_fpr + std_fpr, color=colors[i], alpha=0.15)

    ax_recall.axhline(95, color='red', linestyle='--', linewidth=1.5, label='Target Recall 95%')

    ax_recall.set_ylabel('Recall (%)')
    ax_recall.set_title(f'Recall vs Seeds {title_suffix}')
    ax_recall.set_ylim(0, 110)
    ax_recall.set_xticks(list(x_seeds) + [x_agg])
    ax_recall.set_xticklabels([f"Seed {i}" for i in x_seeds] + ['Mean ± Std'])
    ax_recall.legend(loc='best', fontsize='small', ncol=2)
    ax_recall.grid(True)

    ax_fpr.set_ylabel('False Positive Rate (%)')
    ax_fpr.set_title(f'FPR vs Seeds {title_suffix}')
    ax_fpr.set_ylim(0, 15)
    ax_fpr.set_xlabel('Seed')
    ax_fpr.set_xticks(list(x_seeds) + [x_agg])
    ax_fpr.set_xticklabels([f"Seed {i}" for i in x_seeds] + ['Mean ± Std'])
    ax_fpr.legend(loc='best', fontsize='small', ncol=2)
    ax_fpr.grid(True)

    plt.tight_layout()
    plt.show()

def plot_individual_class_overlayed(df_no, df_yes, agg_no, agg_yes):
    x_seeds = sorted(df_no['Seed'].unique())
    x_agg = max(x_seeds) + 0.5

    for i, class_name in enumerate(class_names):
        no = df_no[df_no['Class'] == class_name].sort_values('Seed')
        yes = df_yes[df_yes['Class'] == class_name].sort_values('Seed')
        agg_n = agg_no[agg_no['Class'] == class_name]
        agg_y = agg_yes[agg_yes['Class'] == class_name]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # --- Recall ---
        for label, df_indiv, df_agg, color, style in zip(['No Weights', 'With Weights'], [no, yes], [agg_n, agg_y], ['gray', colors[i]], ['-', '--']):
            ax1.plot(df_indiv['Seed'], df_indiv['Recall (%)'], marker='o', color=color, label=f'{label}', linestyle=style)
            ax1.errorbar(df_indiv['Seed'], df_indiv['Recall (%)'], yerr=df_indiv['Recall Adj Std Error (%)'],
                         fmt='none', ecolor=color, capsize=4)
            mean = df_agg['Recall (%)'].values[0]
            std = df_agg['Recall Std (%)'].values[0]
            ax1.errorbar(x_agg, mean, yerr=std, fmt='*', color=color, markersize=15, markeredgecolor='black')

        ax1.axhline(95, color='red', linestyle='--', linewidth=1.5, label='Target Recall 95%')
        ymin, ymax = ax1.get_ylim()
        ypad = (ymax - ymin) * 0.1
        ax1.set_ylim(max(0, ymin - ypad), min(110, ymax + ypad))
        ax1.set_title(f'{class_name} - Recall vs Seeds')
        ax1.set_ylabel('Recall (%)')
        ax1.legend()
        ax1.grid(True)

        # --- FPR ---
        for label, df_indiv, df_agg, color, style in zip(['No Weights', 'With Weights'], [no, yes], [agg_n, agg_y], ['gray', colors[i]], ['-', '--']):
            ax2.plot(df_indiv['Seed'], df_indiv['FPR (%)'], marker='o', color=color, label=f'{label}', linestyle=style)
            ax2.errorbar(df_indiv['Seed'], df_indiv['FPR (%)'], yerr=df_indiv['FPR Adj Std Error (%)'],
                         fmt='none', ecolor=color, capsize=4)
            mean = df_agg['FPR (%)'].values[0]
            std = df_agg['FPR Std (%)'].values[0]
            ax2.errorbar(x_agg, mean, yerr=std, fmt='*', color=color, markersize=15, markeredgecolor='black')

        ymin, ymax = ax2.get_ylim()
        ypad = (ymax - ymin) * 0.1
        ax2.set_ylim(max(0, ymin - ypad), min(100, ymax + ypad))
        ax2.set_title(f'{class_name} - FPR vs Seeds')
        ax2.set_ylabel('FPR (%)')
        ax2.set_xlabel('Seed')
        ax2.legend()
        ax2.grid(True)

        ax2.set_xticks(list(x_seeds) + [x_agg])
        ax2.set_xticklabels([f"Seed {s}" for s in x_seeds] + ['Mean ± Std'])

        plt.tight_layout()
        plt.show()