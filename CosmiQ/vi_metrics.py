import numpy as np
from dataset import dataset, PairedDatasetImagePath
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def _build_data(image, mask=None, ignore=None, mode='pair', sky=None, n_mask=1, seed=1):
    """
    Build a dataset object exactly like evaluate.roc does.
    - image: either (N, W, H) numpy array, or list of .npy paths
    - mask, ignore: same semantics as in your training/evaluate code
    """
    if mode == 'pair':
        if isinstance(image, np.ndarray) and image.ndim == 3:
            data = dataset(image, mask, ignore)
        elif isinstance(image[0], str) or isinstance(image[0], np.str_):
            data = PairedDatasetImagePath(image, seed=seed)
        else:
            raise TypeError('Input must be numpy arrays or list of file paths!')
    elif mode == 'simulate':
        raise NotImplementedError("simulate mode not wired here yet")
    else:
        raise ValueError('Mode must be either pair or simulate')
    return data


def collect_vi_outputs(model, image, mask=None, ignore=None,
                       mode='pair', sky=None, n_mask=1, seed=1,
                       max_images=None):
    """
    Loop over a dataset and collect:
      - p: mean CR probability per pixel (flattened)
      - y: ground truth label per pixel (flattened)
      - H: predictive entropy per pixel (flattened)

    Assumes:
      prob = model.clean(img, inpaint=False, binary=False)  # (H, W)
      entropy map is stored in model.last_entropy with same shape.
    """
    data = _build_data(image, mask=mask, ignore=ignore,
                       mode=mode, sky=sky, n_mask=n_mask, seed=seed)

    all_p = []
    all_y = []
    all_H = []

    n_items = len(data) if max_images is None else min(len(data), max_images)

    for idx in range(n_items):
        img, msk, ign = data[idx]      # each is (H, W)

        # run VI model
        prob = model.clean(img, inpaint=False, binary=False)
        H = model.last_entropy

        # sanity check
        if prob.shape != msk.shape or prob.shape != ign.shape:
            raise ValueError(f"Shape mismatch at idx={idx}: "
                             f"prob{prob.shape}, mask{msk.shape}, ignore{ign.shape}")
        if H.shape != prob.shape:
            raise ValueError(f"Entropy shape mismatch at idx={idx}: H{H.shape}, prob{prob.shape}")

        # ignore mask: 1 = bad pixel, so keep where (1 - ignore) == 1
        valid = (1 - ign).astype(bool)

        all_p.append(prob[valid].ravel())
        all_y.append(msk[valid].ravel())
        all_H.append(H[valid].ravel())

    p = np.concatenate(all_p)
    y = np.concatenate(all_y).astype(np.int32)
    H = np.concatenate(all_H)

    return p, y, H


# ---------------------------------------------------------------------
# Scalar + calibration metrics
# ---------------------------------------------------------------------
def calibration_and_scores(p, y, n_bins=15, eps=1e-7):
    """
    p: 1D array of mean predicted CR probabilities
    y: 1D array of ground truth labels (0/1)
    Returns:
      bins_edges, bin_confs, bin_accs, bin_counts, ECE, brier, nll
    """
    # Brier & NLL
    brier = np.mean((p - y) ** 2)
    nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    # Reliability / ECE
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1  # 0..n_bins-1

    bin_confs = np.zeros(n_bins)
    bin_accs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask_b = (bin_ids == b)
        if not np.any(mask_b):
            bin_confs[b] = np.nan
            bin_accs[b] = np.nan
            bin_counts[b] = 0
            continue
        bin_p = p[mask_b]
        bin_y = y[mask_b]
        bin_confs[b] = np.mean(bin_p)
        bin_accs[b]  = np.mean(bin_y)
        bin_counts[b] = mask_b.sum()

    N = bin_counts.sum()
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] == 0:
            continue
        w = bin_counts[b] / N
        ece += w * abs(bin_accs[b] - bin_confs[b])

    return bins, bin_confs, bin_accs, bin_counts, ece, brier, nll


def error_vs_entropy(H, p, y, threshold=0.5, n_bins=10):
    """
    H: entropy array (1D)
    p: probs (1D)
    y: labels (1D)
    Returns: centers, error_rate_per_bin, counts
    """
    y_pred = (p >= threshold).astype(np.int32)
    err = (y_pred != y).astype(np.int32)

    H_min, H_max = H.min(), H.max()
    bins = np.linspace(H_min, H_max, n_bins + 1)
    bin_ids = np.digitize(H, bins) - 1

    centers = []
    err_rates = []
    counts = []

    for b in range(n_bins):
        m = (bin_ids == b)
        if not np.any(m):
            centers.append(0.5 * (bins[b] + bins[b+1]))
            err_rates.append(np.nan)
            counts.append(0)
            continue
        centers.append(0.5 * (bins[b] + bins[b+1]))
        err_rates.append(err[m].mean())
        counts.append(m.sum())

    return np.array(centers), np.array(err_rates), np.array(counts)


def risk_coverage(H, p, y, threshold=0.5):
    """
    Compute risk–coverage curve.
    H: entropy (1D)
    p: mean probabilities (1D)
    y: labels (1D)
    Returns: coverage (1D), risk (1D)
    """
    y_pred = (p >= threshold).astype(np.int32)
    correct = (y_pred == y).astype(np.int32)

    # most confident first -> sort by H ascending
    order = np.argsort(H)
    correct_sorted = correct[order]

    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    accuracy = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - accuracy

    return coverage, risk


# ---------------------------------------------------------------------
# Plotting helpers (ax-aware)
# ---------------------------------------------------------------------
def plot_reliability(bins, bin_confs, bin_accs, ece,
                     title='', ax=None, label='VI U-Net'):
    """
    Reliability diagram. If ax is None, creates its own figure.
    Otherwise, plots into the provided axes (for grid layouts).
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        own_fig = True
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(bin_confs, bin_accs, 'o-', label=label)
    ax.set_xlabel('Predicted CR probability')
    ax.set_ylabel('Empirical CR frequency')
    ax.set_title(f'{title}\nECE={ece:.4f}')
    # ECE={ece:.4f}
    ax.grid(True)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()

    return fig, ax


def plot_error_vs_entropy(centers, err_rates,
                          label='VI U-Net',
                          title='Error rate vs entropy',
                          ax=None):
    """
    Plot error rate vs predictive entropy. If ax is None, creates its own figure.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        own_fig = True
    else:
        fig = ax.figure

    ax.plot(centers, err_rates, 'o-', label=label)
    ax.set_xlabel('Predictive entropy')
    ax.set_ylabel('Error rate')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()

    return fig, ax


def plot_risk_coverage(coverage, risk,
                       label='VI U-Net',
                       title='Risk–coverage curve',
                       ax=None):
    """
    Plot risk–coverage curve. If ax is None, creates its own figure.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        own_fig = True
    else:
        fig = ax.figure

    ax.plot(coverage, risk, label=label)
    ax.set_xlabel('Coverage (fraction of pixels kept)')
    ax.set_ylabel('Risk (error rate)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()

    return fig, ax


# ---------------------------------------------------------------------
# 3×3 grid plots for VI model
# ---------------------------------------------------------------------
def plot_vi_reliability_grid(vi_model,
                             f435_test_field_dirs,
                             f606_test_field_dirs,
                             f814_test_field_dirs,
                             field_label_to_key=None,
                             n_bins=15,
                             seed=1):
    """
    Make a 3x3 grid of VI reliability diagrams:
      rows    = F435W, F606W, F814W
      columns = Globular Clusters, Extragalactic Fields, Resolved Galaxies
    """
    if field_label_to_key is None:
        field_label_to_key = {
            'Globular Clusters':    'GC',
            'Extragalactic Fields': 'EX',
            'Resolved Galaxies':    'GAL',
        }

    filter_dicts = {
        'F435W': f435_test_field_dirs,
        'F606W': f606_test_field_dirs,
        'F814W': f814_test_field_dirs,
    }

    row_labels = ['F435W', 'F606W', 'F814W']
    col_labels = ['Globular Clusters', 'Extragalactic Fields', 'Resolved Galaxies']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)

    for i, filt in enumerate(row_labels):
        for j, field_label in enumerate(col_labels):
            ax = axes[i, j]

            dirs_dict = filter_dicts[filt]
            field_key = field_label_to_key[field_label]
            subset_dirs = dirs_dict[field_key]

            # VI outputs
            p_vi, y_vi, H_vi = collect_vi_outputs(
                vi_model,
                subset_dirs,
                mode='pair',
                seed=seed
            )

            (bins_vi,
             bin_confs_vi,
             bin_accs_vi,
             bin_counts_vi,
             ece_vi,
             brier_vi,
             nll_vi) = calibration_and_scores(p_vi, y_vi, n_bins=n_bins)

            print(f"[VI] {filt} {field_label} – "
                  f"Brier: {brier_vi:.4f}, "
                  f"NLL: {nll_vi:.4f}, "
                  f"ECE: {ece_vi:.4f}")

            title = f'{filt} – {field_label}'
            plot_reliability(
                bins_vi, bin_confs_vi, bin_accs_vi, ece_vi,
                title=title,
                ax=axes[i, j],
                label='VI U-Net'
            )

    plt.show()
    return fig, axes


def plot_vi_error_entropy_grid(vi_model,
                               f435_test_field_dirs,
                               f606_test_field_dirs,
                               f814_test_field_dirs,
                               field_label_to_key=None,
                               threshold=0.5,
                               n_bins=10,
                               seed=1):
    """
    Make a 3x3 grid of VI error-rate vs entropy plots:
      rows    = F435W, F606W, F814W
      columns = Globular Clusters, Extragalactic Fields, Resolved Galaxies
    """
    if field_label_to_key is None:
        field_label_to_key = {
            'Globular Clusters':    'GC',
            'Extragalactic Fields': 'EX',
            'Resolved Galaxies':    'GAL',
        }

    filter_dicts = {
        'F435W': f435_test_field_dirs,
        'F606W': f606_test_field_dirs,
        'F814W': f814_test_field_dirs,
    }

    row_labels = ['F435W', 'F606W', 'F814W']
    col_labels = ['Globular Clusters', 'Extragalactic Fields', 'Resolved Galaxies']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)

    for i, filt in enumerate(row_labels):
        for j, field_label in enumerate(col_labels):
            ax = axes[i, j]

            dirs_dict = filter_dicts[filt]
            field_key = field_label_to_key[field_label]
            subset_dirs = dirs_dict[field_key]

            p_vi, y_vi, H_vi = collect_vi_outputs(
                vi_model,
                subset_dirs,
                mode='pair',
                seed=seed
            )

            centers_vi, err_rates_vi, _ = error_vs_entropy(
                H_vi, p_vi, y_vi,
                threshold=threshold,
                n_bins=n_bins
            )

            title = f'{filt} – {field_label}'
            plot_error_vs_entropy(
                centers_vi, err_rates_vi,
                label='VI U-Net',
                title=title,
                ax=ax
            )

    plt.show()
    return fig, axes


def plot_vi_risk_coverage_grid(vi_model,
                               f435_test_field_dirs,
                               f606_test_field_dirs,
                               f814_test_field_dirs,
                               field_label_to_key=None,
                               threshold=0.5,
                               seed=1):
    """
    Make a 3x3 grid of VI risk–coverage curves:
      rows    = F435W, F606W, F814W
      columns = Globular Clusters, Extragalactic Fields, Resolved Galaxies
    """
    if field_label_to_key is None:
        field_label_to_key = {
            'Globular Clusters':    'GC',
            'Extragalactic Fields': 'EX',
            'Resolved Galaxies':    'GAL',
        }

    filter_dicts = {
        'F435W': f435_test_field_dirs,
        'F606W': f606_test_field_dirs,
        'F814W': f814_test_field_dirs,
    }

    row_labels = ['F435W', 'F606W', 'F814W']
    col_labels = ['Globular Clusters', 'Extragalactic Fields', 'Resolved Galaxies']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)

    for i, filt in enumerate(row_labels):
        for j, field_label in enumerate(col_labels):
            ax = axes[i, j]

            dirs_dict = filter_dicts[filt]
            field_key = field_label_to_key[field_label]
            subset_dirs = dirs_dict[field_key]

            p_vi, y_vi, H_vi = collect_vi_outputs(
                vi_model,
                subset_dirs,
                mode='pair',
                seed=seed
            )

            coverage_vi, risk_vi = risk_coverage(
                H_vi, p_vi, y_vi,
                threshold=threshold
            )

            title = f'{filt} – {field_label}'
            plot_risk_coverage(
                coverage_vi, risk_vi,
                label='VI U-Net',
                title=title,
                ax=ax
            )

    plt.show()
    return fig, axes
