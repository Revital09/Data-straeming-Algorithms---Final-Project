from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = os.path.join("results_Heavy", "Kmeans_sweep_raw.csv")
OUT_DIR = os.path.join("results_Heavy", "plots_minimal")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ensure_dir(OUT_DIR)
    df = pd.read_csv(INPUT_CSV)

    # Basic numeric
    df["quality_loss_pct"] = pd.to_numeric(df["quality_loss_pct_vs_kmeans"], errors="coerce")
    df["runtime_sec"] = pd.to_numeric(df["runtime_sec"], errors="coerce")
    df["mem_bytes"] = pd.to_numeric(df.get("memory", np.nan), errors="coerce")
    df["peak_mem_bytes"] = pd.to_numeric(df.get("extra__peak_mem_bytes", np.nan), errors="coerce")
    df["throughput"] = pd.to_numeric(df.get("extra__throughput_pts_per_sec", np.nan), errors="coerce")
    df["eps"] = pd.to_numeric(df.get("extra__eps", np.nan), errors="coerce")

    # Choose memory measure (Result.memory preferred; fallback to peak)
    df["memory_bytes"] = df["mem_bytes"]
    df.loc[~np.isfinite(df["memory_bytes"]), "memory_bytes"] = df["peak_mem_bytes"]

    # Filter finite rows
    base = df[np.isfinite(df["quality_loss_pct"]) & np.isfinite(df["runtime_sec"])].copy()

    # 1) Quality vs Time
    plt.figure()
    for algo, sub in base.groupby("algorithm"):
        plt.scatter(sub["runtime_sec"], sub["quality_loss_pct"], label=algo, alpha=0.6)
    plt.xlabel("Runtime (sec)")
    plt.ylabel("Quality loss (%) vs KMeans")
    plt.title("Quality vs Time (all runs)")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "quality_vs_time.png"), dpi=200)
    plt.close()

    # 2) Quality vs Memory
    base2 = base[np.isfinite(base["memory_bytes"])].copy()
    plt.figure()
    for algo, sub in base2.groupby("algorithm"):
        plt.scatter(sub["memory_bytes"], sub["quality_loss_pct"], label=algo, alpha=0.6)
    plt.xlabel("Memory (bytes) [Result.memory preferred]")
    plt.ylabel("Quality loss (%) vs KMeans")
    plt.title("Quality vs Memory (all runs)")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "quality_vs_memory.png"), dpi=200)
    plt.close()

    # 3) Time vs Memory
    plt.figure()
    for algo, sub in base2.groupby("algorithm"):
        plt.scatter(sub["memory_bytes"], sub["runtime_sec"], label=algo, alpha=0.6)
    plt.xlabel("Memory (bytes) [Result.memory preferred]")
    plt.ylabel("Runtime (sec)")
    plt.title("Time vs Memory (all runs)")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "time_vs_memory.png"), dpi=200)
    plt.close()

    # 4) Minimal ablation plot: Boutsidis eps → tradeoff
    bout = base2[base2["algorithm"].str.contains("Boutsidis", na=False) & np.isfinite(base2["eps"])].copy()
    if len(bout) > 0:
        # aggregate per eps
        agg = bout.groupby("eps").agg(
            quality_loss_mean=("quality_loss_pct", "mean"),
            runtime_mean=("runtime_sec", "mean"),
            memory_mean=("memory_bytes", "mean"),
            throughput_mean=("throughput", "mean"),
            count=("quality_loss_pct", "count")
        ).reset_index()

        # tradeoff curve: eps vs quality loss
        plt.figure()
        plt.plot(agg["eps"], agg["quality_loss_mean"], marker="o")
        plt.xlabel("eps (Boutsidis)")
        plt.ylabel("Mean quality loss (%)")
        plt.title("Ablation: eps vs quality loss (Boutsidis)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ablation_eps_vs_quality.png"), dpi=200)
        plt.close()

        # tradeoff curve: eps vs runtime
        plt.figure()
        plt.plot(agg["eps"], agg["runtime_mean"], marker="o")
        plt.xlabel("eps (Boutsidis)")
        plt.ylabel("Mean runtime (sec)")
        plt.title("Ablation: eps vs runtime (Boutsidis)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ablation_eps_vs_runtime.png"), dpi=200)
        plt.close()

        # scatter: quality loss vs memory, annotated by eps
        plt.figure()
        plt.scatter(agg["memory_mean"], agg["quality_loss_mean"])
        for _, r in agg.iterrows():
            plt.annotate(f"eps={r['eps']:.2f}", (r["memory_mean"], r["quality_loss_mean"]))
        plt.xlabel("Mean memory (bytes)")
        plt.ylabel("Mean quality loss (%)")
        plt.title("Ablation: tradeoff by eps (Boutsidis)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ablation_eps_tradeoff.png"), dpi=200)
        plt.close()

    print("Plots saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
