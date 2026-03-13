from __future__ import annotations
import time
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import weighted_kmeans_centers, assign_labels, kmeans_cost_sse


@dataclass
class _WeightedPoint:
    x: np.ndarray
    w: float
    is_raw: bool  # True iff this item comes from unread original stream X


class _OnlineFLKMeansState:
    """
    One ONLINE-FL-like invocation, adapted from k-median/facility-location
    to k-means/SSE:
      - opening probability uses squared distance
      - service cost accumulates weighted squared distance
      - centers carry weights
    """
    def __init__(self, facility_cost: float):
        self.facility_cost = float(facility_cost)
        self.centers: List[np.ndarray] = []
        self.counts: List[float] = []
        self.total_cost: float = 0.0  # weighted SSE on modified input
        self.num_opened: int = 0

        # how far this invocation got before stopping
        self.processed_items: int = 0
        self.raw_points_read: int = 0

        self.stopped: bool = False
        self.stop_reason: str | None = None

    def _nearest_center(self, x: np.ndarray) -> Tuple[int, float]:
        C = np.vstack(self.centers)
        sq_dists = np.sum((C - x) ** 2, axis=1)
        j = int(np.argmin(sq_dists))
        d2 = float(sq_dists[j])
        return j, d2

    def process_point(self, item: _WeightedPoint, rng: np.random.Generator) -> None:
        """
        Process one weighted point.
        """
        x = item.x
        w = float(item.w)

        # first center always opens
        if not self.centers:
            self.centers.append(x.copy())
            self.counts.append(w)
            self.num_opened += 1
            self.processed_items += 1
            if item.is_raw:
                self.raw_points_read += 1
            return

        j_near, d2 = self._nearest_center(x)

        # k-means adaptation of Meyerson-style opening rule:
        # min(1, w * d^2 / f)
        p_open = min(1.0, (w * d2) / (self.facility_cost + 1e-12))

        if rng.random() < p_open:
            self.centers.append(x.copy())
            self.counts.append(w)
            self.num_opened += 1
        else:
            self.counts[j_near] += w
            self.total_cost += w * d2

        self.processed_items += 1
        if item.is_raw:
            self.raw_points_read += 1

    def snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.centers:
            raise RuntimeError("Invocation has no centers.")
        C = np.vstack(self.centers).astype(np.float64, copy=False)
        w = np.asarray(self.counts, dtype=np.float64)
        return C, w


class CharikarInspired_PLS_KMeans(Algo):
    """
    Charikar-inspired phased streaming k-means using the PLS skeleton:

      PLS(X):
        L1 <- SET-LB(X)/beta
        X1 <- X
        while unread raw points remain:
            Mi, consumed <- PARA_CLUSTER(Li, Xi)
            Xi+1 <- Mi || unread_suffix_of_X
            Li+1 <- beta * Li
        final weighted-kmeans to exactly k centers

    Important:
    - This preserves the *structure* of Charikar et al. PLS,
      but adapts it to k-means/SSE.
    - Therefore it is heuristic, not theory-faithful to the
      original k-median guarantees.
    """

    name = "[16]CharikarInspired_PLS_KMeans"

    def __init__(
        self,
        beta: float = 25.0,
        gamma: float = 100.0,
        parallel_runs_factor: float = 2.0,   # 2 * log n
        summary_reduce_factor: float = 1.0,  # optional extra compression of Mi
        n_init_phase: int = 3,
        max_iter_phase: int = 80,
        n_init_final: int = 5,
        max_iter_final: int = 200,
    ):
        if beta <= 1.0:
            raise ValueError("beta must be > 1.")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0.")

        self.beta = float(beta)
        self.gamma = float(gamma)
        self.parallel_runs_factor = float(parallel_runs_factor)
        self.summary_reduce_factor = float(summary_reduce_factor)
        self.n_init_phase = int(n_init_phase)
        self.max_iter_phase = int(max_iter_phase)
        self.n_init_final = int(n_init_final)
        self.max_iter_final = int(max_iter_final)

    # -------------------------
    # Lower bound initialization
    # -------------------------
    def _set_lb_kmeans(self, X: np.ndarray, k: int) -> float:
        """
        Charikar PLS uses SET-LB = closest pair distance among first k+1 points.
        For k-means adaptation, use closest pair *squared* distance among first k+1.
        """
        n = X.shape[0]
        m = min(n, k + 1)
        if m <= 1:
            return 1.0

        Y = X[:m].astype(np.float64, copy=False)
        diff = Y[:, None, :] - Y[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        d2 += np.eye(m) * np.inf
        lb = float(np.min(d2))
        return max(lb, 1e-12)

    def _build_modified_stream(
        self,
        summary_points: List[_WeightedPoint],
        X: np.ndarray,
        raw_start_idx: int,
    ) -> List[_WeightedPoint]:
        """
        Xi = summary_points || unread raw suffix
        """
        stream = list(summary_points)
        for i in range(raw_start_idx, X.shape[0]):
            stream.append(_WeightedPoint(x=X[i], w=1.0, is_raw=True))
        return stream

    def _compress_summary_if_needed(
        self,
        centers: np.ndarray,
        weights: np.ndarray,
        k: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optional extra reduction of Mi after a phase.
        This is not in the original PLS pseudocode, but useful in the k-means adaptation.
        """
        target = max(k, int(math.ceil(self.summary_reduce_factor * k)))
        if centers.shape[0] <= target:
            return centers, weights

        Cnew = weighted_kmeans_centers(
            centers,
            weights,
            k=target,
            rng=rng,
            n_init=self.n_init_phase,
            max_iter=self.max_iter_phase,
        )
        lab = assign_labels(centers, Cnew)
        wnew = np.zeros(Cnew.shape[0], dtype=np.float64)
        np.add.at(wnew, lab, weights)
        return Cnew, wnew

    def _para_cluster(
        self,
        modified_stream: List[_WeightedPoint],
        Li: float,
        k: int,
        n: int,
        rng: np.random.Generator,
    ) -> Tuple[List[_WeightedPoint], int, dict]:
        """
        PLS-style PARA-CLUSTER:
          - run ~2 log n parallel ONLINE-FL invocations
          - facility cost f = Li / (k (1 + log n))
          - stop an invocation when median-count or cost threshold is exceeded
          - return the summary from the invocation that survives longest
          - raw points consumed = all raw points seen by that invocation
                                except the last point if it caused overflow
        """
        logn = max(1.0, math.log(max(2, n)))
        num_runs = max(1, int(math.ceil(self.parallel_runs_factor * logn)))

        facility_cost = Li / (k * (1.0 + logn) + 1e-12)

        # Paper-inspired thresholds; used structurally, not as a theorem for k-means
        median_limit = int(math.ceil(4.0 * k * (1.0 + logn) * (1.0 + 4.0 * (self.gamma + self.beta))))
        cost_limit = 4.0 * Li * (1.0 + 4.0 * (self.gamma + self.beta))

        states = [_OnlineFLKMeansState(facility_cost=facility_cost) for _ in range(num_runs)]
        run_seeds = rng.integers(0, 2**32 - 1, size=num_runs, dtype=np.uint64)
        run_rngs = [np.random.default_rng(int(s)) for s in run_seeds]

        for item_idx, item in enumerate(modified_stream):
            active = False
            for st, rrng in zip(states, run_rngs):
                if st.stopped:
                    continue
                active = True

                # process point tentatively
                prev_opened = st.num_opened
                prev_cost = st.total_cost
                prev_processed = st.processed_items
                prev_raw_read = st.raw_points_read
                prev_centers = [c.copy() for c in st.centers]
                prev_counts = list(st.counts)

                st.process_point(item, rrng)

                overflow = (st.num_opened > median_limit) or (st.total_cost > cost_limit)
                if overflow:
                    # revert the last point: paper says mark/read all seen points except last overflow point
                    st.num_opened = prev_opened
                    st.total_cost = prev_cost
                    st.processed_items = prev_processed
                    st.raw_points_read = prev_raw_read
                    st.centers = prev_centers
                    st.counts = prev_counts
                    st.stopped = True
                    st.stop_reason = "threshold_exceeded"

            if not active:
                break

        # states that never stopped are considered to have finished the stream
        for st in states:
            if not st.stopped:
                st.stop_reason = "end_of_stream"

        # last invocation to finish = one that processed the most items
        winner = max(states, key=lambda s: s.processed_items)

        Cw, Ww = winner.snapshot()
        Cw, Ww = self._compress_summary_if_needed(Cw, Ww, k=k, rng=rng)

        summary = [_WeightedPoint(x=Cw[i].copy(), w=float(Ww[i]), is_raw=False) for i in range(Cw.shape[0])]

        stats = {
            "facility_cost": float(facility_cost),
            "median_limit": int(median_limit),
            "cost_limit": float(cost_limit),
            "winner_processed_items": int(winner.processed_items),
            "winner_raw_points_read": int(winner.raw_points_read),
            "winner_num_centers": int(Cw.shape[0]),
            "winner_cost": float(winner.total_cost),
            "num_parallel_runs": int(num_runs),
        }
        return summary, int(winner.raw_points_read), stats

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        n, d = X.shape
        if n == 0:
            raise ValueError("X must contain at least one sample.")
        if k <= 0:
            raise ValueError("k must be positive.")

        # PLS initialization
        L = self._set_lb_kmeans(X, k) / self.beta
        raw_start_idx = 0
        phase_id = 1
        phase_summaries: list[dict] = []

        # M_i from previous phase; initially empty, so X1 = X
        summary_points: List[_WeightedPoint] = []

        while raw_start_idx < n:
            Xi = self._build_modified_stream(summary_points, X, raw_start_idx)

            Mi, raw_consumed, phase_stats = self._para_cluster(
                modified_stream=Xi,
                Li=L,
                k=k,
                n=n,
                rng=rng,
            )

            phase_summaries.append(
                {
                    "phase": int(phase_id),
                    "Li": float(L),
                    "raw_start_index": int(raw_start_idx),
                    "raw_consumed": int(raw_consumed),
                    "summary_in_size": int(len(summary_points)),
                    "summary_out_size": int(len(Mi)),
                    **phase_stats,
                }
            )

            # Xi+1 = Mi || unread suffix of original X
            summary_points = Mi
            raw_start_idx += raw_consumed

            # Safety: ensure progress
            if raw_consumed <= 0:
                # force one raw point to move, to avoid deadlock on overly tight thresholds
                summary_points.append(_WeightedPoint(x=X[raw_start_idx].copy(), w=1.0, is_raw=False))
                raw_start_idx += 1

            L *= self.beta
            phase_id += 1

        # final reduction to exactly k centers
        Csum = np.vstack([p.x for p in summary_points]).astype(np.float64, copy=False)
        Wsum = np.asarray([p.w for p in summary_points], dtype=np.float64)

        if Csum.shape[0] > k:
            centers_final = weighted_kmeans_centers(
                Csum,
                Wsum,
                k=k,
                rng=rng,
                n_init=self.n_init_final,
                max_iter=self.max_iter_final,
            )
        elif Csum.shape[0] == k:
            centers_final = Csum.copy()
        else:
            # pad only for compatibility with downstream code
            extra_idx = rng.integers(0, Csum.shape[0], size=k - Csum.shape[0])
            centers_final = np.vstack([Csum, Csum[extra_idx]])

        t1 = time.perf_counter()

        cost = float(kmeans_cost_sse(X, centers_final))
        pred = assign_labels(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        state_bytes = int(Csum.nbytes + Wsum.nbytes)

        return Result(
            centers=centers_final,
            runtime_sec=float(t1 - t0),
            cost_sse=cost,
            cost_ratio_vs_kmeans=float("nan"),
            ari=ari,
            nmi=nmi,
            extra={
                "points_seen": int(n),
                "dimension": int(d),
                "num_phases": int(phase_id - 1),
                "final_summary_size": int(Csum.shape[0]),
                "final_lower_bound": float(L / self.beta),
                "state_bytes": int(state_bytes),
                "avg_update_ms": float((t1 - t0) * 1000.0 / max(1, n)),
                "phase_summaries": phase_summaries,
                "beta": float(self.beta),
                "gamma": float(self.gamma),
            },
        )