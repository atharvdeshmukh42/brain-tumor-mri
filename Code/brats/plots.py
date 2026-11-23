"""
Every chart helper used in the notebooks, in one visual style.

Each figure is saved as PNG (300 dpi, for slides) and PDF (vector, for the report),
together with a CSV of the numbers behind it, so it can be redrawn without rerunning anything.

Image orientation: arrays are indexed (X, Y, Z) as in the NIfTI files; view2d / view_vertical turn a slice
into the usual radiological display (anterior / superior up).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# -----------------------------------------------------------------------------
# Style (one look for every graph)
# -----------------------------------------------------------------------------
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
GOOD, CRITICAL = '#0ca30c', '#d03b3b'

REGION_COLORS = {'WT': PALETTE[0], 'TC': PALETTE[1], 'ET': PALETTE[2]}
LABEL_COLORS = {1: PALETTE[1], 2: PALETTE[0], 3: PALETTE[2]}       # necrotic core, edema, enhancing
SPLIT_COLORS = {'train': PALETTE[0], 'val': PALETTE[1], 'test': PALETTE[2]}
MOD_COLORS = {'flair': PALETTE[0], 't1': PALETTE[1], 't1ce': PALETTE[2], 't2': PALETTE[3]}
MOD_TITLES = {'flair': 'FLAIR', 't1': 'T1', 't1ce': 'T1ce', 't2': 'T2'}
LABEL_TITLES = {1: 'Necrotic core', 2: 'Edema', 3: 'Enhancing'}

SEQ_CMAP = LinearSegmentedColormap.from_list('seq_blue', BLUE_RAMP)
DIV_CMAP = LinearSegmentedColormap.from_list('div_blue_red', ['#184f95', '#6da7ec', '#f0efec', '#ec8b8b', '#a52a2a'])
HEAT_CMAP = LinearSegmentedColormap.from_list('heat_orange', ['#fde6d8', '#f5a27a', '#eb6834', '#b3401a', '#6e2108'])

SHOW = True   # set to False to only save figures without displaying them


def set_style() -> None:
    plt.rcParams.update({
        'figure.dpi': 100, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.prop_cycle': cycler(color=PALETTE),
        'lines.linewidth': 1.6, 'lines.markersize': 4,
        'legend.frameon': False, 'legend.fontsize': 9,
        'image.cmap': 'gray',
    })


set_style()


def save_fig(fig: plt.Figure, out_dir: Path, name: str, data=None, show: bool | None = None) -> Path:
    """Save PNG + PDF (+ CSV of the plotted numbers), display it in the notebook, and close it.

    data: a DataFrame (saved as <name>.csv) or a dict {key: DataFrame} (saved as <name>_<key>.csv).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.png')
    fig.savefig(out_dir / f'{name}.pdf')
    if data is not None:
        tables = data.items() if isinstance(data, dict) else [('', data)]
        for key, table in tables:
            if table is None:
                continue
            if not isinstance(table, pd.DataFrame):
                table = pd.DataFrame(table)
            table.to_csv(out_dir / (f'{name}_{key}.csv' if key else f'{name}.csv'), index=False)
    if SHOW if show is None else show:
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            pass
    plt.close(fig)
    return out_dir / f'{name}.png'


# -----------------------------------------------------------------------------
# Image helpers
# -----------------------------------------------------------------------------
def view2d(slice_xy: np.ndarray) -> np.ndarray:
    """Orient an axial slice (X, Y) for display: anterior up, patient right on the left (radiological)."""
    return np.asarray(slice_xy).T


def view_vertical(slice_2d: np.ndarray) -> np.ndarray:
    """Orient a coronal (X, Z) or sagittal (Y, Z) slice for display: superior up."""
    return np.asarray(slice_2d).T[::-1]


def _gray_limits(image: np.ndarray) -> tuple[float, float]:
    brain = image[image != 0]
    if brain.size == 0:
        return 0, 1
    return np.percentile(brain, 0.5), np.percentile(brain, 99.5)


def show_gray(ax, image: np.ndarray, title: str | None = None) -> None:
    image = np.asarray(image, dtype=np.float32)
    low, high = _gray_limits(image)
    # Background (exactly 0) is drawn black, also for z-scored images where 0 is mid-grey.
    ax.imshow(np.where(image == 0, low, image), cmap='gray', vmin=low, vmax=high, interpolation='nearest')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


def label_rgba(labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Coloured, semi-transparent overlay of a label map (background transparent)."""
    rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
    for value, color in LABEL_COLORS.items():
        mask = labels == value
        rgba[mask, :3] = matplotlib.colors.to_rgb(color)
        rgba[mask, 3] = alpha
    return rgba


def show_labels(ax, image: np.ndarray, labels: np.ndarray, title: str | None = None, alpha: float = 0.45) -> None:
    show_gray(ax, image, title)
    ax.imshow(label_rgba(labels, alpha), interpolation='nearest')


def show_contours(ax, mask: np.ndarray | None, color: str, linewidth: float = 1.0) -> None:
    if mask is not None and np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def show_heat(ax, image: np.ndarray, heat: np.ndarray, title: str | None = None, alpha_max: float = 0.75) -> None:
    """Heatmap overlay: one orange ramp, transparency grows with the value."""
    show_gray(ax, image, title)
    heat = np.clip(np.asarray(heat, dtype=float), 0, 1)
    rgba = HEAT_CMAP(heat)
    rgba[..., 3] = heat * alpha_max
    ax.imshow(rgba, interpolation='bilinear')


def label_legend(fig: plt.Figure, loc: str = 'lower center', ncol: int = 3) -> None:
    handles = [matplotlib.patches.Patch(color=LABEL_COLORS[v], label=LABEL_TITLES[v]) for v in (1, 2, 3)]
    fig.legend(handles=handles, loc=loc, ncol=ncol, frameon=False)


def image_grid(images: list[np.ndarray], titles: list[str] | None = None, overlays=None, contours=None, heats=None,
               ncols: int = 6, cell: float = 2.1, suptitle: str | None = None, legend_labels: bool = False
               ) -> plt.Figure:
    """Grid of 2D images (already oriented).

    overlays: label maps; contours: per image a list of (mask, color); heats: heatmaps in [0, 1].
    """
    n = len(images)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, 'no examples', ha='center', va='center')
        ax.set_axis_off()
        return fig
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, (cell + 0.35) * nrows + 0.4), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.set_axis_off()
            continue
        title = titles[i] if titles else None
        if heats is not None and heats[i] is not None:
            show_heat(ax, images[i], heats[i], title)
        elif overlays is not None and overlays[i] is not None:
            show_labels(ax, images[i], overlays[i], title)
        else:
            show_gray(ax, images[i], title)
        if contours is not None and contours[i] is not None:
            for mask, color in contours[i]:
                show_contours(ax, mask, color)
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    if legend_labels:
        label_legend(fig)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Generic chart panels
# -----------------------------------------------------------------------------
def log_axis(ax, which: str = 'x') -> None:
    """Log scale without the crowded minor-tick labels."""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    (ax.set_xscale if which == 'x' else ax.set_yscale)('log')
    axis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def hist_panel(ax, values, bins=40, color: str = PALETTE[0], title: str = '', xlabel: str = '',
               logx: bool = False) -> None:
    """Histogram with the median marked."""
    values = pd.Series(values).dropna().values
    if logx:
        values = values[values > 0]
        if values.size:
            bins = np.logspace(np.log10(values.min()), np.log10(values.max() + 1e-9), bins)
            log_axis(ax, 'x')
    if values.size:
        ax.hist(values, bins=bins, color=color, edgecolor='white', linewidth=0.5)
        median = np.median(values)
        ax.axvline(median, color=INK2, linewidth=1, linestyle='--')
        ax.text(0.98, 0.95, f'median {median:.3g}\nn = {values.size}', transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color=INK2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')


def box_panel(ax, groups, labels, colors=None, ylabel: str = '', title: str = '', show_points: bool = True,
              rng_seed: int = 0) -> None:
    """Box plots with every individual value drawn as a jittered dot."""
    rng = np.random.default_rng(rng_seed)
    colors = colors or PALETTE
    data = [pd.Series(group).dropna().values for group in groups]
    boxes = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=not show_points,
                       medianprops=dict(color=INK, linewidth=1.2))
    for i, patch in enumerate(boxes['boxes']):
        color = colors[i % len(colors)]
        patch.set_facecolor(matplotlib.colors.to_rgba(color, 0.25))
        patch.set_edgecolor(color)
    if show_points:
        for i, values in enumerate(data):
            if values.size:
                x = i + 1 + rng.uniform(-0.18, 0.18, size=values.size)
                ax.scatter(x, values, s=6, color=colors[i % len(colors)], alpha=0.5, linewidths=0)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def bar_panel(ax, labels, values, colors=None, title: str = '', ylabel: str = '', fmt: str = '{:.0f}') -> None:
    """Bar chart with the value written on top of each bar."""
    colors = colors or [PALETTE[0]] * len(labels)
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='white', linewidth=1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value), ha='center', va='bottom',
                fontsize=8, color=INK2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def scatter_identity(ax, x, y, color: str = PALETTE[0], xlabel: str = '', ylabel: str = '', title: str = '') -> None:
    """Scatter plot with the y = x line (perfect agreement)."""
    ax.scatter(x, y, s=10, color=color, alpha=0.6, linewidths=0)
    limits = [0, max(np.nanmax(x) if len(x) else 1, np.nanmax(y) if len(y) else 1) * 1.05]
    ax.plot(limits, limits, color=AXIS, linestyle='--', linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def bland_altman(ax, predicted, truth, color: str = PALETTE[0], title: str = '', unit: str = 'cm3') -> None:
    """Bland-Altman plot: difference vs mean, with the bias and the 95% limits of agreement."""
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    mean, difference = (predicted + truth) / 2, predicted - truth
    bias, sd = np.nanmean(difference), np.nanstd(difference)
    ax.scatter(mean, difference, s=10, color=color, alpha=0.6, linewidths=0)
    for value, style in ((bias, '-'), (bias + 1.96 * sd, '--'), (bias - 1.96 * sd, '--')):
        ax.axhline(value, color=INK2, linestyle=style, linewidth=1)
    ax.text(0.99, 0.97, f'bias {bias:.2f}\n95% limits ±{1.96 * sd:.2f} {unit}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=INK2)
    ax.set_xlabel(f'Mean of predicted and true ({unit})')
    ax.set_ylabel(f'Predicted - true ({unit})')
    ax.set_title(title)


# -----------------------------------------------------------------------------
# Complete figures
# -----------------------------------------------------------------------------
def training_curves(history: list[dict], panels, title: str = 'Training curves') -> plt.Figure:
    """panels: list of (panel title, [(history column, legend label), ...], y-axis label)."""
    table = pd.DataFrame(history)
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (panel_title, series, ylabel) in zip(axes.ravel(), panels):
        for i, (column, label) in enumerate(series):
            if column in table and table[column].notna().any():
                points = table[['epoch', column]].dropna()
                ax.plot(points['epoch'], points[column], label=label, color=PALETTE[i % len(PALETTE)],
                        marker='o' if len(points) < 30 else None)
        ax.set_title(panel_title)
        ax.set_xlabel('epoch')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if len(series) > 1:
            ax.legend()
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def confusion_matrix_fig(counts, labels, title: str = 'Confusion matrix', normalize: bool = False) -> plt.Figure:
    """Confusion matrix (rows = true, columns = predicted), as counts or row percentages."""
    counts = np.asarray(counts, dtype=float)
    shown = counts / np.clip(counts.sum(1, keepdims=True), 1, None) if normalize else counts
    top = shown.max() if shown.max() > 0 else 1
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2.2, 1.1 * len(labels) + 1.6))
    ax.imshow(shown, cmap=SEQ_CMAP, vmin=0, vmax=top)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f'{shown[i, j]:.2%}' if normalize else f'{int(counts[i, j]):,}'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color='white' if shown[i, j] > 0.55 * top else INK)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title + (' (row-normalized)' if normalize else ''))
    fig.tight_layout()
    return fig


def roc_fig(curves: dict, title: str = 'ROC curve') -> tuple[plt.Figure, pd.DataFrame | None]:
    """curves: {split name: (labels, probabilities)}."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    rows = []
    for i, (name, (y, prob)) in enumerate(curves.items()):
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=SPLIT_COLORS.get(name, PALETTE[i]), label=f'{name} (AUC {roc_auc_score(y, prob):.3f})')
        rows.append(pd.DataFrame({'split': name, 'fpr': fpr, 'tpr': tpr}))
    ax.plot([0, 1], [0, 1], color=AXIS, linestyle='--', linewidth=1)
    ax.set_xlabel('False positive rate (1 - specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig, (pd.concat(rows) if rows else None)


def pr_fig(curves: dict, title: str = 'Precision-recall curve') -> tuple[plt.Figure, pd.DataFrame | None]:
    """curves: {split name: (labels, probabilities)}."""
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    rows = []
    for i, (name, (y, prob)) in enumerate(curves.items()):
        if len(np.unique(y)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y, prob)
        ax.plot(recall, precision, color=SPLIT_COLORS.get(name, PALETTE[i]),
                label=f'{name} (AP {average_precision_score(y, prob):.3f})')
        rows.append(pd.DataFrame({'split': name, 'recall': recall, 'precision': precision}))
    ax.set_xlabel('Recall (sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc='lower left')
    fig.tight_layout()
    return fig, (pd.concat(rows) if rows else None)


def threshold_sweep_fig(y, prob, chosen: float, title: str = 'Metrics vs decision threshold'
                        ) -> tuple[plt.Figure, pd.DataFrame]:
    """Sensitivity, specificity, precision and F1 for every threshold from 0.01 to 0.99."""
    y = np.asarray(y).astype(int)
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = prob >= threshold
        tp, fp = np.sum(pred & (y == 1)), np.sum(pred & (y == 0))
        fn, tn = np.sum(~pred & (y == 1)), np.sum(~pred & (y == 0))
        sensitivity, specificity = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        rows.append({'threshold': threshold, 'sensitivity': sensitivity, 'specificity': specificity,
                     'precision': precision, 'f1': f1, 'accuracy': (tp + tn) / max(len(y), 1)})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, column in enumerate(['sensitivity', 'specificity', 'precision', 'f1']):
        ax.plot(table['threshold'], table[column], label=column, color=PALETTE[i])
    ax.axvline(chosen, color=INK2, linestyle='--', linewidth=1)
    ax.text(chosen, 0.02, f' chosen {chosen:.2f}', color=INK2, fontsize=8)
    ax.set_xlabel('Decision threshold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc='lower center', ncol=4)
    fig.tight_layout()
    return fig, table


def calibration_fig(table: pd.DataFrame, ece: float, y, prob, title: str = 'Calibration') -> plt.Figure:
    """Reliability diagram and the distribution of predicted probabilities per true class."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axes[0]
    bins = table.dropna()
    ax.plot([0, 1], [0, 1], color=AXIS, linestyle='--', linewidth=1, label='perfect')
    ax.plot(bins['mean_prob'], bins['frac_pos'], marker='o', color=PALETTE[0], label=f'model (ECE {ece:.3f})')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction that really had tumor')
    ax.set_title('Reliability diagram')
    ax.legend(loc='upper left')
    ax = axes[1]
    y = np.asarray(y).astype(int)
    edges = np.linspace(0, 1, 41)
    ax.hist(prob[y == 0], bins=edges, color=PALETTE[0], alpha=0.75, label='no tumor', edgecolor='white', linewidth=0.4)
    ax.hist(prob[y == 1], bins=edges, color=PALETTE[1], alpha=0.75, label='tumor', edgecolor='white', linewidth=0.4)
    ax.set_yscale('log')
    ax.set_xlabel('Predicted tumor probability')
    ax.set_ylabel('slices (log scale)')
    ax.set_title('Confidence by true class')
    ax.legend()
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig


def run_comparison_fig(table: pd.DataFrame, metrics: list[str], name_col: str = 'run',
                       title: str = 'Run comparison') -> plt.Figure:
    """One bar panel per metric, one bar per run."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(max(3.2 * n, 4), 3.6), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        if metric not in table:
            ax.set_visible(False)
            continue
        bar_panel(ax, table[name_col].tolist(), table[metric].tolist(),
                  colors=[PALETTE[i % len(PALETTE)] for i in range(len(table))], title=metric, fmt='{:.3g}')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig
