import numpy as np
from dataset import dataset, PairedDatasetImagePath
import matplotlib.pyplot as plt

def _build_data(image, mask=None, ignore=None, mode='pair', sky=None, n_mask=1, seed=1):
    """
    Same logic as evaluate.roc: builds a dataset object.
    """
    if mode == 'pair':
        if isinstance(image, np.ndarray) and image.ndim == 3:
            data = dataset(image, mask, ignore)
        elif isinstance(image[0], str) or isinstance(image[0], np.str_):
            data = PairedDatasetImagePath(image, seed=seed)
        else:
            raise TypeError('Input must be numpy data arrays or list of file paths!')
    elif mode == 'simulate':
        raise NotImplementedError("simulate mode not wired here yet")
    else:
        raise ValueError('Mode must be either pair or simulate')
    return data


def collect_baseline_outputs(model, image, mask=None, ignore=None,
                             mode='pair', sky=None, n_mask=1, seed=1,
                             max_images=None):
    """
    For the *baseline* deepCR/BN2 model:
      - p: mean CR probability per pixel (flattened)
      - y: ground truth label per pixel (flattened)
      - H: pseudo-uncertainty = p*(1-p) per pixel (flattened)
    """
    data = _build_data(image, mask=mask, ignore=ignore,
                       mode=mode, sky=sky, n_mask=n_mask, seed=seed)

    all_p = []
    all_y = []
    all_H = []

    n_items = len(data) if max_images is None else min(len(data), max_images)

    for idx in range(n_items):
        img, msk, ign = data[idx]      # each is (H, W)

        # deterministic baseline prob map
        prob = model.clean(img, inpaint=False, binary=False)  # (H, W)

        # pseudo-uncertainty: highest near 0.5, lowest near 0 or 1
        H = prob * (1.0 - prob)

        if prob.shape != msk.shape or prob.shape != ign.shape:
            raise ValueError(f"Shape mismatch at idx={idx}: "
                             f"prob{prob.shape}, mask{msk.shape}, ignore{ign.shape}")

        # ignore mask: 1 = bad pixel
        valid = (1 - ign).astype(bool)

        all_p.append(prob[valid].ravel())
        all_y.append(msk[valid].ravel())
        all_H.append(H[valid].ravel())

    p = np.concatenate(all_p)
    y = np.concatenate(all_y).astype(np.int32)
    H = np.concatenate(all_H)

    return p, y, H


def calibration_and_scores(p, y, n_bins=15, eps=1e-7):
    brier = np.mean((p - y) ** 2)
    nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1

    bin_confs = np.zeros(n_bins)
    bin_accs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask_b = (bin_ids == b)
        if not np.any(mask_b):
            bin_confs[b] = np.nan
            bin_accs[b] = np.nan
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


def plot_reliability(bins, bin_confs, bin_accs, ece,
                     label='Model', title='', ax=None):
    """
    Reliability diagram. If ax is None, creates its own figure.
    Otherwise, plots into the provided axes (for grid layouts).
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        own_fig = True

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(bin_confs, bin_accs, 'o-', label=label)
    ax.set_xlabel('Predicted CR probability')
    ax.set_ylabel('Empirical CR frequency')
    ax.set_title(f'{title}\nECE={ece:.4f}')
    ax.grid(True)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()


def error_vs_entropy(H, p, y, threshold=0.5, n_bins=10):
    y_pred = (p >= threshold).astype(np.int32)
    err = (y_pred != y).astype(np.int32)

    H_min, H_max = H.min(), H.max()
    bins = np.linspace(H_min, H_max, n_bins + 1)
    bin_ids = np.digitize(H, bins) - 1

    centers = []
    err_rates = []

    for b in range(n_bins):
        m = (bin_ids == b)
        if not np.any(m):
            centers.append(0.5 * (bins[b] + bins[b+1]))
            err_rates.append(np.nan)
            continue
        centers.append(0.5 * (bins[b] + bins[b+1]))
        err_rates.append(err[m].mean())

    return np.array(centers), np.array(err_rates)


def risk_coverage(H, p, y, threshold=0.5):
    y_pred = (p >= threshold).astype(np.int32)
    correct = (y_pred == y).astype(np.int32)

    order = np.argsort(H)  # most confident (low H) -> least
    correct_sorted = correct[order]

    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    accuracy = cum_correct / np.arange(1, n + 1)
    risk = 1.0 - accuracy

    return coverage, risk


def plot_error_vs_entropy(centers, err_rates, label='Model',
                          title='Error rate vs uncertainty', ax=None):
    """
    Plot error rate vs uncertainty. If ax is None, creates its own figure.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        own_fig = True

    ax.plot(centers, err_rates, 'o-', label=label)
    ax.set_xlabel('Uncertainty (proxy)')
    ax.set_ylabel('Error rate')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()


def plot_risk_coverage(coverage, risk, label='Model',
                       title='Risk–coverage curve', ax=None):
    """
    Plot risk–coverage curve. If ax is None, creates its own figure.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        own_fig = True

    ax.plot(coverage, risk, label=label)
    ax.set_xlabel('Coverage (fraction of pixels kept)')
    ax.set_ylabel('Risk (error rate)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()

def plot_reliability_grid(baseline_model,
                          f435_test_field_dirs,
                          f606_test_field_dirs,
                          f814_test_field_dirs,
                          field_label_to_key=None,
                          n_bins=15,
                          seed=1):
    """
    Make a 3x3 grid of reliability diagrams (9 plots):
      rows    = F435W, F606W, F814W
      columns = Globular Clusters, Extragalactic Fields, Resolved Galaxies

    Parameters
    ----------
    baseline_model : model with .clean(...)
    f435_test_field_dirs, f606_test_field_dirs, f814_test_field_dirs : dicts
        Mapping from field key (e.g. 'GC', 'EXF', 'GAL') to list of image paths.
    field_label_to_key : dict or None
        Mapping from human-readable label to dict key.
        If None, uses a default mapping.
    n_bins : int
        Number of bins for calibration_and_scores.
    seed : int
        Random seed passed to collect_baseline_outputs.
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

            # Collect probabilities and pseudo-uncertainty
            p_b, y_b, H_b = collect_baseline_outputs(
                baseline_model,
                subset_dirs,
                mode='pair',
                seed=seed
            )

            # Calibration metrics
            (bins_b,
             bin_confs_b,
             bin_accs_b,
             bin_counts_b,
             ece_b,
             brier_b,
             nll_b) = calibration_and_scores(p_b, y_b, n_bins=n_bins)

            print(f"{filt} {field_label} – "
                  f"Brier: {brier_b:.4f}, "
                  f"NLL: {nll_b:.4f}, "
                  f"ECE: {ece_b:.4f}")

            # Plot reliability into the correct subplot
            title = f'{filt} – {field_label}'
            plot_reliability(
                bins_b, bin_confs_b, bin_accs_b, ece_b,
                label='Baseline U-Net',
                title=title,
                ax=ax
            )

    plt.show()
    return fig, axes

def plot_error_entropy_grid(baseline_model,
                            f435_test_field_dirs,
                            f606_test_field_dirs,
                            f814_test_field_dirs,
                            field_label_to_key=None,
                            threshold=0.5,
                            n_bins=10,
                            seed=1):
    """
    Make a 3x3 grid of error-rate vs entropy plots:
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

            # baseline outputs
            p_b, y_b, H_b = collect_baseline_outputs(
                baseline_model,
                subset_dirs,
                mode='pair',
                seed=seed
            )

            centers_b, err_rates_b = error_vs_entropy(
                H_b, p_b, y_b,
                threshold=threshold,
                n_bins=n_bins
            )

            title = f'{filt} – {field_label}'
            plot_error_vs_entropy(
                centers_b, err_rates_b,
                label='Baseline U-Net',
                title=title,
                ax=ax
            )

    plt.show()
    return fig, axes

def plot_risk_coverage_grid(baseline_model,
                            f435_test_field_dirs,
                            f606_test_field_dirs,
                            f814_test_field_dirs,
                            field_label_to_key=None,
                            threshold=0.5,
                            seed=1):
    """
    Make a 3x3 grid of risk–coverage curves:
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

            # baseline outputs
            p_b, y_b, H_b = collect_baseline_outputs(
                baseline_model,
                subset_dirs,
                mode='pair',
                seed=seed
            )

            coverage_b, risk_b = risk_coverage(
                H_b, p_b, y_b,
                threshold=threshold
            )

            title = f'{filt} – {field_label}'
            plot_risk_coverage(
                coverage_b, risk_b,
                label='Baseline U-Net',
                title=title,
                ax=ax
            )

    plt.show()
    return fig, axes
