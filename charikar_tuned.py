from __future__ import annotations
import os
import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from charikar_updated import Charikar_Facility_PhasedKMeans
from tuned_utils import extract_quality, pick_best_overall

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def tune_charikar_parameters(
    samples: np.ndarray,
    k: int,
    output_dir: str,
    labels=None,
    init_facility_values=(0.25, 0.5, 1.0, 2.0),
    phase_centers_factor_values=(8.0, 12.0, 16.0),
    compress_to_factor_values=(3.0, 4.0, 6.0),
    growth_factor_values=(1.5, 2.0, 3.0),
    seeds=(42, 77, 211),
    quality_weight: float = 0.5,
    runtime_weight: float = 0.25,
    memory_weight: float = 0.25,
):
    """
    Runs Charikar_Facility_PhasedKMeans for every parameter combination,
    aggregates metrics across seeds, ranks them by one overall score,
    saves CSVs and graphs, and returns the single best overall combination.

    Saved files in output_dir:
      - charikar_all_results.csv
      - charikar_aggregated_results.csv
      - charikar_scored_results.csv
      - charikar_best_overall.csv
      - best_overall.json
      - memory_vs_quality.png
      - runtime_vs_quality.png
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for init_facility in init_facility_values:
        for phase_centers_factor in phase_centers_factor_values:
            for compress_to_factor in compress_to_factor_values:
                for growth_factor in growth_factor_values:
                    for seed in seeds:
                        rng = np.random.default_rng(seed)

                        algo = Charikar_Facility_PhasedKMeans(
                            init_facility=init_facility,
                            phase_centers_factor=phase_centers_factor,
                            compress_to_factor=compress_to_factor,
                            growth_factor=growth_factor,
                        )

                        result = algo.fit(X=samples, k=k, rng=rng, y=labels)
                        quality = extract_quality(result)
                        extra = result.extra or {}

                        rows.append(
                            {
                                "seed": int(seed),
                                "init_facility": float(init_facility),
                                "phase_centers_factor": float(phase_centers_factor),
                                "compress_to_factor": float(compress_to_factor),
                                "growth_factor": float(growth_factor),
                                "runtime_sec": float(result.runtime_sec),
                                "memory": float(result.memory),
                                "cost_sse": float(result.cost_sse),
                                "ari": None if result.ari is None else float(result.ari),
                                "nmi": None if result.nmi is None else float(result.nmi),
                                "quality": float(quality),
                                "points_seen": (
                                    int(result.points_seen)
                                    if result.points_seen is not None
                                    else int(extra.get("points_seen", -1))
                                ),
                                "opened_centers_final_state": int(
                                    extra.get("opened_centers_final_state", -1)
                                ),
                                "facility_final": float(extra.get("facility_final", np.nan)),
                                "avg_update_ms": float(extra.get("avg_update_ms", np.nan)),
                                "num_phases": int(extra.get("num_phases", -1)),
                                "num_compressions": int(extra.get("num_compressions", -1)),
                                "total_opened_events": int(extra.get("total_opened_events", -1)),
                                "phase_max_centers": int(extra.get("phase_max_centers", -1)),
                                "compress_target": int(extra.get("compress_target", -1)),
                            }
                        )

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(output_dir, "charikar_all_results.csv"), index=False)

    agg = (
        df_all.groupby(
            [
                "init_facility",
                "phase_centers_factor",
                "compress_to_factor",
                "growth_factor",
            ],
            as_index=False,
        )
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
            points_seen_mean=("points_seen", "mean"),
            opened_centers_final_state_mean=("opened_centers_final_state", "mean"),
            facility_final_mean=("facility_final", "mean"),
            avg_update_ms_mean=("avg_update_ms", "mean"),
            num_phases_mean=("num_phases", "mean"),
            num_compressions_mean=("num_compressions", "mean"),
            total_opened_events_mean=("total_opened_events", "mean"),
            phase_max_centers_mean=("phase_max_centers", "mean"),
            compress_target_mean=("compress_target", "mean"),
        )
        .reset_index(drop=True)
    )

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

    with open(os.path.join(output_dir, "best_overall.json"), "w", encoding="utf-8") as f:
        json.dump(best_one_df.to_dict(orient="records"), f, indent=2)

    best_row = best_one_df.iloc[0]

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["memory_mean"], scored_df["quality_mean"])
    plt.scatter(
        [best_row["memory_mean"]],
        [best_row["quality_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        (
            "BEST "
            f"f0={best_row['init_facility']}, "
            f"phase={best_row['phase_centers_factor']}, "
            f"cmp={best_row['compress_to_factor']}, "
            f"grow={best_row['growth_factor']}"
        ),
        (best_row["memory_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Memory usage (bytes, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Charikar tuning: Memory usage vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_vs_quality.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(scored_df["runtime_sec_mean"], scored_df["quality_mean"])
    plt.scatter(
        [best_row["runtime_sec_mean"]],
        [best_row["quality_mean"]],
        marker="x",
        s=120,
    )
    plt.annotate(
        (
            "BEST "
            f"f0={best_row['init_facility']}, "
            f"phase={best_row['phase_centers_factor']}, "
            f"cmp={best_row['compress_to_factor']}, "
            f"grow={best_row['growth_factor']}"
        ),
        (best_row["runtime_sec_mean"], best_row["quality_mean"]),
    )
    plt.xlabel("Runtime (sec, mean)")
    plt.ylabel("Quality (mean)")
    plt.title("Charikar tuning: Runtime vs Quality")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_quality.png"), dpi=150)
    plt.close()

    return best_one_df


def main():
    X, y = make_blobs(
        n_samples=10000,
        centers=8,
        n_features=20,
        cluster_std=2.0,
        random_state=42,
    )

    X = X.astype("float32")

    best_df = tune_charikar_parameters(
        samples=X,
        k=8,
        output_dir="output/charikar_blobs",
        labels=y,
        init_facility_values=(0.5, 1.0, 2.0),
        phase_centers_factor_values=(8.0, 12.0),
        compress_to_factor_values=(4.0, 6.0),
        growth_factor_values=(1.5, 2.0),
        seeds=(42, 77, 211),
        quality_weight=0.5,
        runtime_weight=0.25,
        memory_weight=0.25,
    )

    print("Best overall parameter combination:")
    print(best_df)


if __name__ == "__main__":
    main()