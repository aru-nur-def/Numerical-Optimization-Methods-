# /project/anomaly_detection/pipeline_example.py
# Comments: English only
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from datetime import datetime, timedelta

from plotting import save_all_charts


def make_synthetic_timeseries(n: int = 500, seed: int = 7) -> pd.Series:
    """Create a synthetic univariate time series with a few injected spikes."""
    rng = np.random.default_rng(seed)
    idx0 = datetime(2025, 1, 1, 0, 0, 0)
    index = [idx0 + timedelta(minutes=15) * i for i in range(n)]

    base = np.sin(np.linspace(0, 10, n)) * 3.0 + 20.0
    noise = rng.normal(0, 1.2, size=n)
    y = base + noise

    # inject a few anomalies
    spikes_idx = rng.choice(np.arange(30, n - 30), size=6, replace=False)
    y[spikes_idx] += rng.normal(10, 3, size=len(spikes_idx))

    return pd.Series(y, index=pd.to_datetime(index))


def simple_zscore_anomaly(ts: pd.Series, z_thr: float = 2.75) -> np.ndarray:
    """Compute z-scores and return boolean mask of anomalies."""
    mu = ts.rolling(24, min_periods=12).mean()
    sd = ts.rolling(24, min_periods=12).std().fillna(1.0)
    z = (ts - mu) / sd
    return (z.abs() > z_thr).astype(int).values  # 1 if anomaly else 0


def compute_residuals(ts: pd.Series, window: int = 24) -> np.ndarray:
    """Residuals vs. a rolling mean baseline."""
    mu = ts.rolling(window, min_periods=window // 2).mean()
    mu = mu.fillna(method="bfill").fillna(method="ffill")
    resid = (ts - mu).values
    return resid


def fake_binary_labels(ts: pd.Series, anomaly_mask: np.ndarray) -> np.ndarray:
    """
    In a real setup, labels come from chargebacks or analyst decisions.
    For the demo, we treat detected spikes as positive labels with noise.
    """
    labels = anomaly_mask.copy()
    # randomly flip a few labels to simulate imperfect ground truth
    rng = np.random.default_rng(42)
    flip_idx = rng.choice(np.arange(len(labels)), size=10, replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    return labels


def edges_synthetic(num_accounts: int = 30, num_merchants: int = 15, seed: int = 3):
    """Build a small synthetic bipartite edge list for a network plot."""
    rng = np.random.default_rng(seed)
    accounts = [f"a_{i}" for i in range(num_accounts)]
    merchants = [f"m_{j}" for j in range(num_merchants)]
    edges = []
    for a in accounts:
        # connect each account to a few merchants
        m_ids = rng.choice(merchants, size=rng.integers(1, 5), replace=False)
        for m in m_ids:
            edges.append((a, m))
    # optional scores to size nodes (e.g., anomaly score per node)
    node_scores = {n: float(rng.random()) for n in (accounts + merchants)}
    return edges, node_scores


def main() -> None:
    # 1) synthetic time series
    ts = make_synthetic_timeseries(n=520)
    # 2) naive anomaly mask
    anomaly_mask = simple_zscore_anomaly(ts, z_thr=2.5)
    anomalies_idx = ts.index[anomaly_mask == 1]

    # 3) residuals vs. rolling baseline
    residuals = compute_residuals(ts, window=24)
    std = residuals.std() if residuals.std() > 0 else 1.0
    std_resid = residuals / std  # standardized for CUSUM chart

    # 4) emulate scores & labels (for PR/ROC)
    #    Here we pretend |std_resid| is a score: larger -> more anomalous
    scores = np.abs(std_resid)
    y_true = fake_binary_labels(ts, anomaly_mask)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)

    # 5) toy network
    edges, node_scores = edges_synthetic()

    # 6) save all charts
    out_dir = "figs"
    save_all_charts( 
        out_dir=out_dir,
        ts=ts,
        anomalies_idx=anomalies_idx,
        residuals=std_resid,
        pr_curve=(precision, recall),
        roc_curve=(fpr, tpr),
        edges=edges,
        node_scores=node_scores,
    )
    print(f"Charts saved to: {out_dir}")


if __name__ == "__main__":
    main()
