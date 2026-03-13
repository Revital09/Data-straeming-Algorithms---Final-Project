from __future__ import annotations
import time
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import weighted_kmeans_centers, assign_labels, kmeans_cost_sse

class Charikar_Facility(Algo):
    """
    Facility-location style online opening:
      - keep opened centers C with weights (counts)
      - each point opens with prob min(1, d^2 / f)
      - if too many centers: f*=2 and compress via weighted kmeans
      - final weighted kmeans to k
    """
    name = "[16]Charikar2003_FacilityOnline"

    def __init__(self, init_facility: float = 1.0, max_centers_factor: float = 12.0):
        self.init_facility = init_facility
        self.max_centers_factor = max_centers_factor

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        n = X.shape[0]
        idx0 = int(rng.integers(0, n))
        centers = [X[idx0].copy()]
        counts = [1.0]

        f = float(self.init_facility)
        max_centers = max(int(self.max_centers_factor * k), k + 5)

        for i in range(n):
            x = X[i]
            C = np.vstack(centers)
            d2 = float(np.min(np.sum((C - x) ** 2, axis=1)))

            p_open = min(1.0, d2 / (f + 1e-12))
            if rng.random() < p_open:
                centers.append(x.copy())
                counts.append(1.0)
            else:
                j = int(np.argmin(np.sum((C - x) ** 2, axis=1)))
                counts[j] += 1.0

            if len(centers) > max_centers:
                f *= 2.0
                Cmat = np.vstack(centers)
                w = np.array(counts, dtype=np.float64)
                target = max(k, max_centers // 2)

                Cnew = weighted_kmeans_centers(Cmat, w, k=target, rng=rng, n_init=3, max_iter=80)
                lab = assign_labels(Cmat, Cnew)
                wnew = np.zeros(Cnew.shape[0], dtype=np.float64)
                np.add.at(wnew, lab, w)

                centers = [c for c in Cnew]
                counts = [float(v) for v in wnew]

        Cmat = np.vstack(centers)
        w = np.array(counts, dtype=np.float64)

        centers_final = weighted_kmeans_centers(Cmat, w, k=k, rng=rng, n_init=5, max_iter=200)

        t1 = time.perf_counter()
        cost = kmeans_cost_sse(X, centers_final)
        pred = assign_labels(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        points_seen = int(n)
        avg_update_ms = float((t1 - t0) * 1000.0 / max(1, n))
        state_bytes = int(Cmat.nbytes + w.nbytes)

        return Result(
            centers=centers_final,
            runtime_sec=t1 - t0,
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "opened_centers": int(Cmat.shape[0]),
                "facility_final": float(f),
                "points_seen": int(n),
                "avg_update_ms": float(avg_update_ms),
                "state_bytes": int(state_bytes),
            },
        )