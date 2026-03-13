# make_heatmaps_per_algorithm.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def compute_speedup(df_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      speedup_vs_offline_mean per (dataset,n,d,k) relative to OfflineKMeans runtime_sec_mean.
    """
    key_cols = ["sweep_id", "dataset", "n", "d", "k"]
    offline = df_agg[df_agg["algorithm"] == "KMeans(sk)"][key_cols + ["runtime_sec_mean"]].copy()
    offline = offline.rename(columns={"runtime_sec_mean": "offline_runtime_sec_mean"})

    merged = df_agg.merge(offline, on=key_cols, how="left")
    merged["speedup_vs_offline_mean"] = merged["offline_runtime_sec_mean"] / merged["runtime_sec_mean"]

    # Ensure quality loss pct mean exists
    if "quality_loss_pct_vs_offline_mean" not in merged.columns:
        merged["quality_loss_pct_vs_offline_mean"] = (merged["cost_ratio_vs_offline_mean"] - 1.0) * 100.0

    return merged

def save_tradeoff_heatmap_per_algorithm(
    df_agg_tradeoff: pd.DataFrame,
    out_dir: str,
    bins_x: int = 30,
    bins_y: int = 30,
    x_clip=(-5.0, 30.0),
    y_clip=(0.2, 5.0),
) -> None:
    """
    One PNG per algorithm:
      X = quality_loss_pct_vs_offline_mean
      Y = speedup_vs_offline_mean
      Value = count density
    """
    ensure_dir(out_dir)

    algorithms = sorted(df_agg_tradeoff["algorithm"].dropna().unique().tolist())
    for algo in algorithms:
        sub = df_agg_tradeoff[df_agg_tradeoff["algorithm"] == algo].copy()
        xs = sub["quality_loss_pct_vs_offline_mean"].astype(float).to_numpy()
        ys = sub["speedup_vs_offline_mean"].astype(float).to_numpy()

        mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size == 0:
            continue

        xs = np.clip(xs, x_clip[0], x_clip[1])
        ys = np.clip(ys, y_clip[0], y_clip[1])

        H, xedges, yedges = np.histogram2d(xs, ys, bins=[bins_x, bins_y], range=[list(x_clip), list(y_clip)])

        plt.figure()
        plt.imshow(H.T, aspect="auto", origin="lower")
        plt.colorbar()
        plt.title(f"Tradeoff density heatmap — {algo}")
        plt.xlabel("Quality loss (%) vs Offline (binned)")
        plt.ylabel("Speedup vs Offline (binned)")

        xt = np.linspace(0, bins_x - 1, 6).astype(int)
        yt = np.linspace(0, bins_y - 1, 6).astype(int)
        plt.xticks(xt, [f"{xedges[i]:.1f}" for i in xt], rotation=45, ha="right")
        plt.yticks(yt, [f"{yedges[i]:.2f}" for i in yt])

        plt.tight_layout()
        safe_name = algo.replace("/", "_").replace(" ", "_")
        out_path = os.path.join(out_dir, f"heatmap_tradeoff_{safe_name}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

def main():
    # Update these two lines if you change names
    input_agg_csv = os.path.join("results_out", "FAST_sweep_v2_agg.csv")
    out_dir = os.path.join("results_out", "heatmaps_per_algorithm")

    df = pd.read_csv(input_agg_csv)
    df = compute_speedup(df)
    save_tradeoff_heatmap_per_algorithm(df, out_dir=out_dir)

    print("Done.")
    print("Heatmaps written to:", out_dir)

if __name__ == "__main__":
    main()