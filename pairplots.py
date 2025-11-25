
import numpy as np
import matplotlib.pyplot as plt

def pair_plot(X, labels=None, figsize=(8, 8), hist_bins=20, alpha=0.6, s=10):
    """
    Create a pair plot (scatterplot matrix) using matplotlib and numpy only.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features).
    labels : np.ndarray or list, optional
        Array of shape (n_samples,), used for coloring points.
    figsize : tuple
        Size of the matplotlib figure.
    hist_bins : int
        Number of bins for histograms.
    alpha : float
        Transparency for scatter points.
    s : int
        Marker size for scatter plots.
    """

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    n_samples, n_features = X.shape

    fig, axes = plt.subplots(n_features, n_features, figsize=figsize)

    # Choose colors
    if labels is None:
        colors = np.full(n_samples, 'tab:blue')
    else:
        unique_labels = np.unique(labels)
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        label_to_color = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
        colors = np.array([label_to_color[lab] for lab in labels])

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                # Histogram on the diagonal
                ax.hist(X[:, j], bins=hist_bins, color='gray', alpha=0.7)
            else:
                # Scatter plot
                ax.scatter(X[:, j], X[:, i], c=colors, alpha=alpha, s=s)

            # Remove ticks except on left and bottom
            if i < n_features - 1:
                ax.set_xticks([])
            if j > 0:
                ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    
    
    import numpy as np
import matplotlib.pyplot as plt

def pair_plot_multi(
    tables,
    table_labels=None,
    feature_names=None,
    colors=None,
    markers=None,
    figsize=None,
    alpha=0.6,
    s=10,
    bins=24,
    density=True,
    share_axes=True,           # now only shares X by column; Y is not shared
    clip_quantiles=(0.0, 1.0),
    max_points=8000,
    random_state=0,
):
    """
    Pair plot (scatterplot matrix) comparing multiple tables using only numpy + matplotlib.

    Parameters are the same as before. Key change: we do NOT share y across a row,
    so diagonal histograms get their own y-scale and no longer flatten to lines.
    """
    # Normalize inputs
    tables = [np.asarray(t) for t in tables]
    if len(tables) == 0:
        raise ValueError("`tables` must be a non-empty list of 2D arrays.")
    n_features = tables[0].shape[1]
    if any(t.ndim != 2 for t in tables) or any(t.shape[1] != n_features for t in tables):
        raise ValueError("All tables must be 2D with the same number of features.")
    if table_labels is None:
        table_labels = [f"table_{i}" for i in range(len(tables))]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError("`feature_names` length must match n_features.")

    # Colors/markers
    if colors is None:
        base = plt.cm.tab10.colors
        colors = [base[i % len(base)] for i in range(len(tables))]
    if markers is None:
        base_markers = ['o', 's', '^', 'x', 'D', 'v', 'P', '*']
        markers = [base_markers[i % len(base_markers)] for i in range(len(tables))]

    # Optional proportional downsampling for scatter
    rng = np.random.default_rng(random_state)
    totals = np.array([len(t) for t in tables], dtype=float)
    total_points = totals.sum()
    if total_points > max_points:
        frac = max_points / total_points
        keep_counts = np.maximum(1, np.floor(totals * frac)).astype(int)
        scatter_tables = [t[rng.choice(len(t), size=k, replace=False)] for t, k in zip(tables, keep_counts)]
    else:
        scatter_tables = tables

    # Global feature ranges with optional quantile clipping
    all_concat = np.vstack(tables)
    q_low, q_high = clip_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("clip_quantiles must satisfy 0.0 <= q_low < q_high <= 1.0")

    xlims = []
    for j in range(n_features):
        col = all_concat[:, j]
        lo = np.quantile(col, q_low) if q_low > 0 else np.min(col)
        hi = np.quantile(col, q_high) if q_high < 1 else np.max(col)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        xlims.append((lo, hi))

    # Figure/axes: share ONLY X by column; do not share Y so histograms keep their own scale
    if figsize is None:
        figsize = (3 * n_features, 3 * n_features)
    fig, axes = plt.subplots(
        n_features, n_features, figsize=figsize,
        sharex=('col' if share_axes else False),
        sharey=False
    )
    if n_features == 1:
        axes = np.array([[axes]])

    # Bin edges per feature for consistent overlay
    bin_edges = [np.linspace(lo, hi, bins + 1) for (lo, hi) in xlims]

    # Plot
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                # Diagonal: overlay step histograms per table (independent y-scale)
                edges = bin_edges[j]
                for t, label, color in zip(tables, table_labels, colors):
                    vals = t[:, j]
                    vals = vals[(vals >= edges[0]) & (vals <= edges[-1])]
                    ax.hist(vals, bins=edges, histtype='step',
                            density=density, label=label, color=color, linewidth=1.5)
                ax.set_xlim(xlims[j])
                # Let y autoscale naturally for histograms
            else:
                # Off-diagonal: overlay scatter per table (set both x/y limits)
                for t, label, color, marker in zip(scatter_tables, table_labels, colors, markers):
                    ax.scatter(t[:, j], t[:, i], s=s, alpha=alpha,
                               c=[color], marker=marker, linewidths=0)
                ax.set_xlim(xlims[j])
                ax.set_ylim(xlims[i])

            # Labels only on outer edges
            if i == n_features - 1:
                ax.set_xlabel(feature_names[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(feature_names[i])
            else:
                ax.set_yticklabels([])

            ax.grid(False)

    # Legend on top-right subplot
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle='None', label=lbl)
               for lbl, c, m in zip(table_labels, colors, markers)]
    axes[0, -1].legend(handles=handles, loc='upper right', frameon=False)

    plt.tight_layout()
    return fig, axes