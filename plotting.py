# /project/anomaly_detection/plotting.py
# Comments: English only
from typing import Optional, Sequence, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_timeseries_with_anomalies(
    ts: pd.Series,
    anomalies_idx: Optional[Sequence[pd.Timestamp]] = None,
    title: str = "Transaction amount (with anomalies)",
    savepath: Optional[str] = None,
) -> None:
    """
    Plot a time series and optionally highlight anomaly timestamps.
    Note: uses default matplotlib colors (no explicit color selection).
    """
    fig = plt.figure()
    plt.plot(ts.index, ts.values, linewidth=1.5)
    if anomalies_idx is not None and len(anomalies_idx) > 0:
        # plot anomalies as points; default color/marker
        plt.scatter(anomalies_idx, ts.loc[anomalies_idx].values, s=25)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_residuals_hist(
    residuals: np.ndarray,
    bins: int = 40,
    title: str = "Residuals histogram",
    savepath: Optional[str] = None,
) -> None:
    """Histogram of model residuals."""
    fig = plt.figure()
    plt.hist(residuals, bins=bins)
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(
    precision: Sequence[float],
    recall: Sequence[float],
    title: str = "Precision–Recall curve",
    savepath: Optional[str] = None,
) -> None:
    """Plot Precision-Recall curve from arrays (already computed)."""
    fig = plt.figure()
    plt.plot(recall, precision, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    title: str = "ROC curve",
    savepath: Optional[str] = None,
) -> None:
    """Plot ROC curve from arrays (already computed)."""
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=1.5)
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_cusum(
    residuals: np.ndarray,
    title: str = "CUSUM of standardized residuals",
    savepath: Optional[str] = None,
) -> None:
    """Simple CUSUM chart over standardized residuals."""
    std = np.std(residuals) if np.std(residuals) > 0 else 1.0
    z = residuals / std
    cusum = np.cumsum(z)

    fig = plt.figure()
    plt.plot(np.arange(len(cusum)), cusum, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("CUSUM")
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_graph_network(
    edges: Sequence[Tuple[str, str]],
    node_scores: Optional[dict] = None,
    title: str = "Account–Merchant network",
    savepath: Optional[str] = None,
) -> None:
    """
    Draw a simple network using networkx.
    node_scores: optional dict {node: score} to scale node size.
    No explicit colors are set (defaults only).
    """
    G = nx.Graph()
    G.add_edges_from(edges)

    # basic layout
    pos = nx.spring_layout(G, seed=42)

    # default sizes; scale by score if provided
    sizes = []
    for n in G.nodes():
        if node_scores and n in node_scores:
            # scale size; bounded to avoid extreme bubbles
            s = 300 + 1200 * float(node_scores[n])
            s = max(100, min(2000, s))
            sizes.append(s)
        else:
            sizes.append(300)

    fig = plt.figure()
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=sizes,
        width=0.5,
    )
    plt.title(title)
    plt.tight_layout()
    if savepath:
        _ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=300)
    plt.close(fig)


def save_all_charts(
    out_dir: str,
    ts: pd.Series,
    anomalies_idx: Optional[Sequence[pd.Timestamp]],
    residuals: np.ndarray,
    pr_curve: Tuple[Sequence[float], Sequence[float]],
    roc_curve: Tuple[Sequence[float], Sequence[float]],
    edges: Sequence[Tuple[str, str]],
    node_scores: Optional[dict] = None,
) -> None:
    """
    Convenience wrapper: generate and save all standard charts to out_dir.
    """
    _ensure_dir(out_dir)
    plot_timeseries_with_anomalies(
        ts, anomalies_idx, savepath=os.path.join(out_dir, "timeseries_anomalies.png")
    )
    plot_residuals_hist(
        residuals, savepath=os.path.join(out_dir, "residuals_hist.png")
    )
    precision, recall = pr_curve
    plot_precision_recall_curve(
        precision, recall, savepath=os.path.join(out_dir, "pr_curve.png")
    )
    fpr, tpr = roc_curve
    plot_roc_curve(fpr, tpr, savepath=os.path.join(out_dir, "roc_curve.png"))
    plot_cusum(
        residuals, savepath=os.path.join(out_dir, "cusum.png")
    )
    plot_graph_network(
        edges, node_scores=node_scores, savepath=os.path.join(out_dir, "network.png")
    )
