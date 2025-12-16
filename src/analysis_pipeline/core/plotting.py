"""Plotting utilities for gradient analysis and visualization."""

from typing import List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.stats import norm
import tifffile as tiff

from .metrics import compute_kl_matrix, normalize_histogram
from ..utils.file_utils import load_prediction


def plot_multiple_hist(
    ax: plt.Axes,
    histograms: List[np.ndarray],
    bin_edges: np.ndarray,
    labels: List[str],
    colors: List[str],
    title: str,
    legend: bool = False,
) -> None:
    """
    Plot multiple precomputed histograms with fitted normal distributions.

    Args:
        ax: Matplotlib axis to plot on
        histograms: List of histogram count arrays
        bin_edges: Shared bin edges for all histograms
        labels: Labels for each histogram
        colors: Colors for each histogram
        title: Plot title
        legend: Whether to display legend
    """
    if not (len(histograms) == len(labels) == len(colors)):
        raise ValueError("histograms, labels, and colors must have the same length")

    bin_edges = np.asarray(bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)

    data_handles = []
    fit_handles = []

    for counts, label, color in zip(histograms, labels, colors):
        if np.sum(counts) == 0:
            continue

        # Normalize histogram to density
        area = np.trapz(counts, bin_centers)
        density = counts / area if area > 0 else counts

        # Weighted mean & std
        mu = np.average(bin_centers, weights=density)
        var = np.average((bin_centers - mu) ** 2, weights=density)
        std = np.sqrt(var)

        # Plot histogram bars
        bars = ax.bar(
            bin_centers,
            density,
            width=np.diff(bin_edges),
            alpha=0.5,
            color=color,
            label=label,
            edgecolor="none",
        )
        data_handles.append(bars[0])

        # Plot normal fit line
        (line,) = ax.plot(
            x,
            norm.pdf(x, mu, std),
            color=color,
            lw=1.8,
            label=f"{label} fit: μ={mu:.2f}, σ={std:.2f}",
        )
        fit_handles.append(line)

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.set_tick_params(labelleft=True)

    if legend:
        # Left legend: fit info
        leg_fit = ax.legend(
            handles=fit_handles,
            loc="upper left",
            fontsize=8,
            frameon=True,
            title="Normal Fit",
        )
        ax.add_artist(leg_fit)

        # Right legend: histogram names
        ax.legend(
            handles=data_handles,
            labels=labels,
            loc="upper right",
            fontsize=8,
            frameon=True,
            title="Histograms",
        )


def plot_multiple_boxplots(
    axs: List[plt.Axes],
    arrays_list: List[List[np.ndarray]],
    labels_list: List[List[str]],
    colors_list: List[List[str]],
    titles_list: List[str],
    legend: bool = False,
) -> None:
    """
    Plot multiple boxplots dynamically on given axes.

    Args:
        axs: List of matplotlib axes
        arrays_list: List of lists of arrays to plot
        labels_list: Labels for each array in each subplot
        colors_list: Colors for each box in each subplot
        titles_list: Titles for each subplot
        legend: Whether to add legend
    """
    if not (
        len(axs)
        == len(arrays_list)
        == len(labels_list)
        == len(colors_list)
        == len(titles_list)
    ):
        raise ValueError(
            "Length of axs, arrays_list, labels_list, colors_list, "
            "titles_list must match"
        )

    for ax, arrays, labels, colors, title in zip(
        axs, arrays_list, labels_list, colors_list, titles_list
    ):
        bp = ax.boxplot(arrays, patch_artist=True, labels=labels)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.grid(True)
        if legend:
            ax.legend(labels)


def plot_kl_heatmaps_for_range(
    grad_utils_list: List,
    bin_edges: np.ndarray,
    start: int = 29,
    end: int = 34,
    channels: int = 1,
    labels: Optional[List[str]] = None,
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    Generate KL divergence heatmaps for a range of tile positions.

    Args:
        grad_utils_list: List of gradient utility objects
        bin_edges: Bin edges for histograms
        start: Start index for position range
        end: End index for position range
        channels: Channel to analyze
        labels: Method labels
        cmap: Colormap for heatmap

    Returns:
        Matplotlib figure
    """
    from .gradient_analysis import GradientUtils

    n_utils = len(grad_utils_list)
    if labels is None:
        labels = [f"Model{i}" for i in range(n_utils)]

    middle_hists = []
    for gu in grad_utils_list:
        grad_mid = gu.get_gradients_at("middle", channels=channels)
        middle_hists.append(GradientUtils.compute_histograms(grad_mid, bin_edges))

    n_plots = end - start + 1
    fig, axes = plt.subplots(
        1, n_plots, figsize=(10 * n_plots, 7.5), constrained_layout=False
    )
    if n_plots == 1:
        axes = [axes]

    kl_mats = []
    for index in range(start, end + 1):
        histograms = []
        for gu, mid_hist in zip(grad_utils_list, middle_hists):
            grad_at_idx = gu.get_gradients_at(index, channels=channels)
            hist_at_idx = GradientUtils.compute_histograms(grad_at_idx, bin_edges)
            histograms.extend([hist_at_idx, mid_hist])
        kl_mats.append(compute_kl_matrix(histograms))

    vmin = min(np.min(mat) for mat in kl_mats)
    vmax = max(np.max(mat) for mat in kl_mats)

    for ax, index, kl_mat in zip(axes, range(start, end + 1), kl_mats):
        hist_labels = []
        for label in labels:
            hist_labels.extend([f"{label}-Edge", f"{label}-Mid"])
        sns.heatmap(
            kl_mat,
            annot=True,
            fmt=".3f",
            xticklabels=hist_labels,
            yticklabels=hist_labels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"Index {index}")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        ax=axes,
        location="right",
        shrink=0.8,
        label="KL Divergence",
    )
    fig.suptitle("KL Divergence Between Gradient Distributions", fontsize=16)
    return fig


def save_figure(fig: plt.Figure, save_path: Path, dpi: int = 300) -> None:
    """
    Save figure to file and close it.

    Args:
        fig: Matplotlib figure
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path.name}")


def plot_prediction_comparison(
    prediction_paths: List[Union[str, Path]],
    target_dir: Union[str, Path],
    target_channel_paths: List[str],
    method_names: List[str],
    frame_idx: int = 0,
    save_path: Optional[Union[str, Path]] = None,
    crop_factor: float = 4.0,
    vmin_percentile: float = 1.0,
    vmax_percentile: float = 99.0,
    figsize: Tuple[int, int] = (20, 12),
    dpi: int = 300,
) -> plt.Figure:
    """
    Create a comprehensive comparison plot of predictions vs ground truth.

    This function creates a grid layout showing:
    - Full ground truth images for each channel
    - Full input image (concatenated channels)
    - Cropped/zoomed regions with yellow rectangle indicators
    - Predictions from multiple methods for each channel
    - Difference maps between predictions

    Args:
        prediction_paths: List of paths to prediction files (.tiff, .pkl, .dill)
        target_dir: Base directory containing target channel subdirectories
        target_channel_paths: List of relative paths to channel files
            (e.g., ["channel_1/channel_1.tiff", "channel_2/channel_2.tiff"])
        method_names: Names of prediction methods (must match prediction_paths length)
        frame_idx: Frame index to visualize (for multi-frame data)
        save_path: Optional path to save the figure
        crop_factor: Factor for center crop (1/crop_factor of image size)
        vmin_percentile: Lower percentile for normalization
        vmax_percentile: Upper percentile for normalization
        figsize: Figure size (width, height)
        dpi: DPI for saved figure

    Returns:
        Matplotlib figure object

    Example:
        >>> plot_prediction_comparison(
        ...     prediction_paths=["pred1.tiff", "pred2.tiff", "pred3.tiff"],
        ...     target_dir="/group/jug/aman/Datasets/PAVIA_ATN/data",
        ...     target_channel_paths=["channel_1/channel_1.tiff", "channel_2/channel_2.tiff"],
        ...     method_names=["OuterTiling", "InnerTiling", "SWT"],
        ...     frame_idx=0,
        ...     save_path="./comparison.png"
        ... )
    """
    target_dir = Path(target_dir)
    n_methods = len(prediction_paths)
    n_channels = len(target_channel_paths)

    if len(method_names) != n_methods:
        raise ValueError(
            f"Number of method names ({len(method_names)}) must match "
            f"number of predictions ({n_methods})"
        )

    # Load ground truth targets
    targets = []
    for channel_path in target_channel_paths:
        full_path = target_dir / channel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Target file not found: {full_path}")
        target = tiff.imread(full_path)
        targets.append(target)

    # Load predictions
    predictions = []
    for pred_path in prediction_paths:
        pred = load_prediction(pred_path)
        predictions.append(pred)

    # Extract frame if needed
    targets_frame = [t[frame_idx] if t.ndim > 2 else t for t in targets]
    predictions_frame = [
        p[frame_idx] if p.ndim > 2 else p for p in predictions
    ]

    # Handle channel dimension - assume last dimension is channel for predictions
    # Targets are typically (H, W) or (C, H, W)
    targets_frame_channels = []
    for t in targets_frame:
        if t.ndim == 2:
            targets_frame_channels.append([t])
        elif t.ndim == 3 and t.shape[0] <= 3:  # (C, H, W)
            targets_frame_channels.append([t[i] for i in range(t.shape[0])])
        else:
            targets_frame_channels.append([t])

    # Flatten targets to individual channels
    all_gt_channels = []
    for t_channels in targets_frame_channels:
        all_gt_channels.extend(t_channels)

    # Create input image (concatenation of all GT channels)
    input_img = np.concatenate([c for c in all_gt_channels], axis=-1) if len(all_gt_channels) > 1 else all_gt_channels[0]

    # Compute normalization from GT
    vmin = np.percentile(all_gt_channels[0], vmin_percentile)
    vmax = np.percentile(all_gt_channels[0], vmax_percentile)

    # Compute crop region (center crop)
    h, w = all_gt_channels[0].shape
    crop_h, crop_w = int(h / crop_factor), int(w / crop_factor)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2

    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows=n_channels + 1,
        ncols=n_methods + 3,
        figure=fig,
        hspace=0.3,
        wspace=0.3
    )

    # Row 0: Full images (GT channels + Input)
    for ch_idx, gt_channel in enumerate(all_gt_channels[:n_channels]):
        ax = fig.add_subplot(gs[0, ch_idx])
        ax.imshow(gt_channel, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT Channel {ch_idx + 1}", fontsize=10, fontweight="bold")
        ax.axis("off")
        # Add crop indicator
        rect = Rectangle(
            (start_w, start_h), crop_w, crop_h,
            linewidth=2, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)

    # Input image
    ax_input = fig.add_subplot(gs[0, n_channels])
    if input_img.ndim == 2:
        ax_input.imshow(input_img, cmap="gray", vmin=vmin, vmax=vmax)
    else:
        # For multi-channel, show first channel
        ax_input.imshow(input_img if input_img.ndim == 2 else input_img[..., 0], cmap="gray", vmin=vmin, vmax=vmax)
    ax_input.set_title("Input (GT)", fontsize=10, fontweight="bold")
    ax_input.axis("off")
    rect = Rectangle(
        (start_w, start_h), crop_w, crop_h,
        linewidth=2, edgecolor="yellow", facecolor="none"
    )
    ax_input.add_patch(rect)

    # Row 0: Cropped GT
    for ch_idx, gt_channel in enumerate(all_gt_channels[:n_channels]):
        ax = fig.add_subplot(gs[0, n_channels + 1 + ch_idx])
        cropped = gt_channel[start_h:start_h + crop_h, start_w:start_w + crop_w]
        ax.imshow(cropped, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT Ch{ch_idx + 1} (Crop)", fontsize=9)
        ax.axis("off")

    # Rows 1+: Predictions for each channel
    for ch_idx in range(n_channels):
        for method_idx, (pred, method_name) in enumerate(zip(predictions_frame, method_names)):
            # Extract channel from prediction
            if pred.ndim == 2:
                pred_channel = pred
            elif pred.ndim == 3:
                # Assume last dimension is channel
                if pred.shape[-1] <= n_channels:
                    pred_channel = pred[..., ch_idx] if ch_idx < pred.shape[-1] else pred[..., 0]
                else:
                    # Assume first dimension is channel
                    pred_channel = pred[ch_idx] if ch_idx < pred.shape[0] else pred[0]
            else:
                pred_channel = pred

            # Full prediction
            ax = fig.add_subplot(gs[ch_idx + 1, method_idx])
            ax.imshow(pred_channel, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{method_name}\nCh{ch_idx + 1}", fontsize=9)
            ax.axis("off")
            rect = Rectangle(
                (start_w, start_h), crop_w, crop_h,
                linewidth=2, edgecolor="yellow", facecolor="none"
            )
            ax.add_patch(rect)

            # Cropped prediction
            ax = fig.add_subplot(gs[ch_idx + 1, n_methods + method_idx])
            cropped_pred = pred_channel[start_h:start_h + crop_h, start_w:start_w + crop_w]
            ax.imshow(cropped_pred, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{method_name} (Crop)", fontsize=9)
            ax.axis("off")

    # Add difference maps between first and other predictions (last column)
    if n_methods > 1:
        for ch_idx in range(n_channels):
            for method_idx in range(1, n_methods):
                pred_ref = predictions_frame[0]
                pred_comp = predictions_frame[method_idx]

                # Extract channels
                if pred_ref.ndim == 2:
                    ref_channel = pred_ref
                elif pred_ref.ndim == 3:
                    ref_channel = pred_ref[..., ch_idx] if ch_idx < pred_ref.shape[-1] else pred_ref[..., 0]
                else:
                    ref_channel = pred_ref

                if pred_comp.ndim == 2:
                    comp_channel = pred_comp
                elif pred_comp.ndim == 3:
                    comp_channel = pred_comp[..., ch_idx] if ch_idx < pred_comp.shape[-1] else pred_comp[..., 0]
                else:
                    comp_channel = pred_comp

                diff = comp_channel - ref_channel
                diff_cropped = diff[start_h:start_h + crop_h, start_w:start_w + crop_w]

                ax = fig.add_subplot(gs[ch_idx + 1, -1])
                im = ax.imshow(diff_cropped, cmap="seismic", vmin=-np.abs(diff_cropped).max(), vmax=np.abs(diff_cropped).max())
                ax.set_title(f"Diff: {method_names[method_idx]}\n- {method_names[0]}", fontsize=8)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Prediction Comparison (Frame {frame_idx})",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    if save_path:
        save_path = Path(save_path)
        save_figure(fig, save_path, dpi=dpi)

    return fig
