from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import weighted_kmeans_centers, assign_labels, kmeans_cost_sse


class Charikar_Facility_PhasedKMeans(Algo):
    """
    Phased streaming facility-opening heuristic inspired by Charikar,
    adapted to Euclidean k-means with weighted k-means compression.

    Notes:
    - This is a practical phased k-means adaptation.
    - It is not a faithful implementation of Charikar et al. (2003),
      which is for streaming k-median.
    """

    name = "[16]Charikar2003_PhasedFacilityKMeans"

    def __init__(
        self,
        init_facility: float = 1.0,
        phase_centers_factor: float = 12.0,
        compress_to_factor: float = 6.0,
        growth_factor: float = 2.0,
        n_init_compress: int = 3,
        max_iter_compress: int = 80,
        n_init_final: int = 5,
        max_iter_final: int = 200,
    ):
        self.init_facility = float(init_facility)
        self.phase_centers_factor = float(phase_centers_factor)
        self.compress_to_factor = float(compress_to_factor)
        self.growth_factor = float(growth_factor)
        self.n_init_compress = int(n_init_compress)
        self.max_iter_compress = int(max_iter_compress)
        self.n_init_final = int(n_init_final)
        self.max_iter_final = int(max_iter_final)

    def _compress_weighted_centers(
        self,
        centers: list[np.ndarray],
        counts: list[float],
        target: int,
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], list[float]]:
        if len(centers) <= target:
            return centers, counts

        Cmat = np.vstack(centers).astype(np.float64, copy=False)
        w = np.asarray(counts, dtype=np.float64)

        Cnew = weighted_kmeans_centers(
            Cmat,
            w,
            k=target,
            rng=rng,
            n_init=self.n_init_compress,
            max_iter=self.max_iter_compress,
        )

        lab = assign_labels(Cmat, Cnew)
        wnew = np.zeros(Cnew.shape[0], dtype=np.float64)
        np.add.at(wnew, lab, w)

        centers_new = [c.copy() for c in Cnew]
        counts_new = [float(v) for v in wnew]
        return centers_new, counts_new

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        n, d = X.shape
        if n == 0:
            raise ValueError("X must contain at least one sample.")
        if k <= 0:
            raise ValueError("k must be positive.")

        phase_max_centers = max(int(self.phase_centers_factor * k), k + 5)
        compress_target = max(k, int(self.compress_to_factor * k))

        idx0 = int(rng.integers(0, n))
        centers: list[np.ndarray] = [X[idx0].copy()]
        counts: list[float] = [1.0]

        facility_cost = float(self.init_facility)

        phase_id = 1
        phase_start_idx = 0
        phase_points = 0
        total_opened_events = 1
        num_compressions = 0
        phase_summaries: list[dict] = []

        for i in range(n):
            x = X[i]
            phase_points += 1

            C = np.vstack(centers)
            sq_dists = np.sum((C - x) ** 2, axis=1)
            j_near = int(np.argmin(sq_dists))
            d2 = float(sq_dists[j_near])

            p_open = min(1.0, d2 / (facility_cost + 1e-12))

            if rng.random() < p_open:
                centers.append(x.copy())
                counts.append(1.0)
                total_opened_events += 1
            else:
                counts[j_near] += 1.0

            if len(centers) > phase_max_centers:
                before = len(centers)

                centers, counts = self._compress_weighted_centers(
                    centers=centers,
                    counts=counts,
                    target=compress_target,
                    rng=rng,
                )

                after = len(centers)
                num_compressions += 1

                phase_summaries.append(
                    {
                        "phase": phase_id,
                        "start_index": phase_start_idx,
                        "end_index": i,
                        "points_in_phase": phase_points,
                        "facility_cost": float(facility_cost),
                        "centers_before_compress": int(before),
                        "centers_after_compress": int(after),
                    }
                )

                facility_cost *= self.growth_factor
                phase_id += 1
                phase_start_idx = i + 1
                phase_points = 0

        phase_summaries.append(
            {
                "phase": phase_id,
                "start_index": phase_start_idx,
                "end_index": n - 1,
                "points_in_phase": phase_points,
                "facility_cost": float(facility_cost),
                "centers_before_compress": int(len(centers)),
                "centers_after_compress": int(len(centers)),
            }
        )

        Cmat = np.vstack(centers).astype(np.float64, copy=False)
        w = np.asarray(counts, dtype=np.float64)

        if Cmat.shape[0] > k:
            centers_final = weighted_kmeans_centers(
                Cmat,
                w,
                k=k,
                rng=rng,
                n_init=self.n_init_final,
                max_iter=self.max_iter_final,
            )
        elif Cmat.shape[0] == k:
            centers_final = Cmat.copy()
        else:
            extra_idx = rng.integers(0, Cmat.shape[0], size=k - Cmat.shape[0])
            centers_final = np.vstack([Cmat, Cmat[extra_idx]])

        t1 = time.perf_counter()

        cost = float(kmeans_cost_sse(X, centers_final))
        pred = assign_labels(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        avg_update_ms = float((t1 - t0) * 1000.0 / max(1, n))
        memory = float(Cmat.nbytes + w.nbytes)

        return Result(
            centers=centers_final,
            runtime_sec=float(t1 - t0),
            memory=memory,
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            points_seen=int(n),
            extra={
                "opened_centers_final_state": int(Cmat.shape[0]),
                "facility_final": float(facility_cost),
                "avg_update_ms": float(avg_update_ms),
                "num_phases": int(phase_id),
                "num_compressions": int(num_compressions),
                "total_opened_events": int(total_opened_events),
                "phase_max_centers": int(phase_max_centers),
                "compress_target": int(compress_target),
                "phase_summaries": phase_summaries,
            },
        )