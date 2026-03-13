from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from charikar_updated import Charikar_Facility
from tuned_utils import extract_quality, pick_best_overall


def tune_charikar_parameters(
    samples: np.ndarray,
    k: int,
    output_dir: str,
    labels=None,
    init_facility_values=(0.25, 0.5, 1.0),
    max_centers_factor_values=(4.0, 8.0, 12.0),
    seeds=(42, 77, 211),
    quality_weight: float = 0.5,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
) -> dict:
    """
    Run Charikar_Facility on a parameter grid, aggregate across seeds,
    rank combinations by tradeoff score, save outputs to output_dir,
    and return ONLY the best parameter combination as a dict.
    """

    os.makedirs(output_dir, exist_ok=True)
    rows = []

    for init_facility in init_facility_values:
        for max_centers_factor in max_centers_factor_values:
            for seed in seeds:
                rng = np.random.default_rng(seed)

                algo = Charikar_Facility(
                    init_facility=init_facility,
                    max_centers_factor=max_centers_factor,
                )

                result = algo.fit(
                    samples,
                    k,
                    rng,
                    labels,
                )

                quality = extract_quality(result)
                extra = result.extra or {}

                rows.append(
                    {
                        "seed": int(seed),
                        "init_facility": float(init_facility),
                        "max_centers_factor": float(max_centers_factor),
                        "runtime_sec": float(result.runtime_sec),
                        "memory": float(result.memory),
                        "memory_mb": float(result.memory / (1024.0 ** 2)),
                        "cost_sse": float(result.cost_sse),
                        "ari": None if result.ari is None else float(result.ari),
                        "nmi": None if result.nmi is None else float(result.nmi),
                        "quality": float(quality),
                        "opened_centers": int(extra.get("opened_centers", -1)),
                        "facility_final": float(extra.get("facility_final", np.nan)),
                        "points_seen": int(extra.get("points_seen", -1)),
                        "avg_update_ms": float(extra.get("avg_update_ms", np.nan)),
                    }
                )

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, "charikar_all_results.csv"), index=False)

    agg = (
        df_all.groupby(["init_facility", "max_centers_factor"], as_index=False)
        .agg(
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_std=("runtime_sec", "std"),
            memory_mean=("memory", "mean"),
            memory_std=("memory", "std"),
            cost_sse_mean=("cost_sse", "mean"),
            cost_sse_std=("cost_sse", "std"),
            quality_mean=("quality", "mean"),
            quality_std=("quality", "std"),
            ari_mean=("ari", "mean"),
            nmi_mean=("nmi", "mean"),
            opened_centers_mean=("opened_centers", "mean"),
            facility_final_mean=("facility_final", "mean"),
        )
        .reset_index(drop=True)
    )
    agg["memory_mb_mean"] = agg["memory_mean"] / (1024.0 ** 2)

    agg.to_csv(os.path.join(output_dir, "charikar_aggregated_results.csv"), index=False)

    scored_df, best_one_df = pick_best_overall(
        agg=agg,
        quality_col="quality_mean",
        runtime_col="runtime_sec_mean",
        memory_col="memory_mean",
        quality_weight=quality_weight,
        runtime_weight=runtime_weight,
        memory_weight=memory_weight,
    )

    scored_df.to_csv(os.path.join(output_dir, "charikar_scored_results.csv"), index=False)
    best_one_df.to_csv(os.path.join(output_dir, "charikar_best_overall.csv"), index=False)

    best_row = best_one_df.iloc[0]

    best_result = {
        "init_facility": float(best_row["init_facility"]),
        "max_centers_factor": float(best_row["max_centers_factor"]),
        "tradeoff_score": float(best_row["tradeoff_score"]),
        "quality_mean": float(best_row["quality_mean"]),
        "runtime_sec_mean": float(best_row["runtime_sec_mean"]),
        "memory_mean": float(best_row["memory_mean"]),
        "memory_mb_mean": float(best_row["memory_mb_mean"]),
        "cost_sse_mean": float(best_row["cost_sse_mean"]),
        "ari_mean": None if pd.isna(best_row["ari_mean"]) else float(best_row["ari_mean"]),
        "nmi_mean": None if pd.isna(best_row["nmi_mean"]) else float(best_row["nmi_mean"]),
        "opened_centers_mean": float(best_row["opened_centers_mean"]),
        "facility_final_mean": float(best_row["facility_final_mean"]),
    }

    with open(os.path.join(output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    # Memory vs Quality
    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["memory_mb_mean"], scored_df["quality_mean"])
    plt.scatter(
        [best_row["memory_mb_mean"]],
        [best_row["quality_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        f"BEST f0={best_row['init_facility']}, mcf={best_row['max_centers_factor']}",
        (best_row["memory_mb_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Memory usage (MB, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Charikar tuning: Memory usage vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_vs_quality.png"), dpi=150)
    plt.close()

    # Runtime vs Quality
    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["runtime_sec_mean"], scored_df["quality_mean"])
    plt.scatter(
        [best_row["runtime_sec_mean"]],
        [best_row["quality_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        f"BEST f0={best_row['init_facility']}, mcf={best_row['max_centers_factor']}",
        (best_row["runtime_sec_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Runtime (sec, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Charikar tuning: Runtime vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_quality.png"), dpi=150)
    plt.close()

    return best_result


def main():
    X, y = make_blobs(
        n_samples=10000,
        centers=8,
        n_features=10,
        cluster_std=2.0,
        random_state=42,
    )

    X = X.astype(np.float32)

    best_result = tune_charikar_parameters(
        samples=X,
        k=8,
        output_dir="output/charikar_blobs",
        labels=y,
        init_facility_values=(0.25, 0.5, 1.0, 2.0, 4.0),
        max_centers_factor_values=(4.0, 8.0, 12.0, 16.0, 24.0),
        seeds=(42, 77, 211),
        quality_weight=0.5,
        runtime_weight=0.25,
        memory_weight=0.25,
    )

    print("Best overall parameter combination:")
    print(best_result)


if __name__ == "__main__":
    main()