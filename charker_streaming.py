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
    is_raw: bool


class _OnlineFLKMeansState:
    """
    One ONLINE-FL-style run, adapted to k-means:
    - opening probability uses squared distance
    - service cost accumulates weighted squared distance
    """
    def __init__(self, facility_cost: float):
        self.facility_cost = float(facility_cost)
        self.centers: List[np.ndarray] = []
        self.counts: List[float] = []
        self.total_cost: float = 0.0
        self.num_opened: int = 0
        self.processed_items: int = 0
        self.raw_points_read: int = 0
        self.stopped: bool = False
        self.stop_reason: str | None = None

    def process_point(self, item: _WeightedPoint, rng: np.random.Generator) -> None:
        x = item.x
        w = float(item.w)

        if not self.centers:
            self.centers.append(x.copy())
            self.counts.append(w)
            self.num_opened += 1
            self.processed_items += 1
            if item.is_raw:
                self.raw_points_read += 1
            return

        C = np.vstack(self.centers)
        sq_dists = np.sum((C - x) ** 2, axis=1)
        j = int(np.argmin(sq_dists))
        d2 = float(sq_dists[j])

        # k-means adaptation of the paper's facility-opening idea
        p_open = min(1.0, (w * d2) / (self.facility_cost + 1e-12))

        if rng.random() < p_open:
            self.centers.append(x.copy())
            self.counts.append(w)
            self.num_opened += 1
        else:
            self.counts[j] += w
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


class Charikar_KMeans(Algo):
    """
    Charikar-inspired PLS skeleton adapted to k-means.

    Compared to the earlier version:
    - no extra summary reduction after each phase
    - Mi is exactly the winner's weighted centers
    - number of parallel runs is fixed to ceil(2 * log n), following the paper's structure
    """

    name = "[16]CharikarInspired_PLS_KMeans_ChunkLoop"

    def __init__(
        self,
        beta: float = 25.0,
        gamma: float = 100.0,
        chunk_size: int = 1000,
        n_init_final: int = 5,
        max_iter_final: int = 300,
    ):
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.chunk_size = int(chunk_size)
        self.n_init_final = int(n_init_final)
        self.max_iter_final = int(max_iter_final)

    def _set_lb_kmeans(self, X: np.ndarray, k: int) -> float:
        """
        PLS-style lower bound initialization adapted to k-means:
        use the minimum squared pairwise distance among the first k+1 points.
        """
        m = min(X.shape[0], k + 1)
        if m <= 1:
            return 1.0

        Y = X[:m].astype(np.float64, copy=False)
        diff = Y[:, None, :] - Y[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        lb = float(np.min(d2))
        return max(lb, 1e-12)

    def _init_phase_states(
        self,
        Li: float,
        k: int,
        n: int,
        rng: np.random.Generator,
    ):
        logn = max(1.0, math.log(max(2, n)))

        # Paper-inspired: 2 log n parallel runs
        num_runs = max(1, int(math.ceil(2.0 * logn)))

        facility_cost = Li / (k * (1.0 + logn) + 1e-12)

        median_limit = int(math.ceil(
            4.0 * k * (1.0 + logn) * (1.0 + 4.0 * (self.gamma + self.beta))
        ))
        cost_limit = 4.0 * Li * (1.0 + 4.0 * (self.gamma + self.beta))

        states = [_OnlineFLKMeansState(facility_cost=facility_cost) for _ in range(num_runs)]
        run_seeds = rng.integers(0, 2**32 - 1, size=num_runs, dtype=np.uint64)
        run_rngs = [np.random.default_rng(int(s)) for s in run_seeds]

        return states, run_rngs, facility_cost, median_limit, cost_limit, num_runs

    def _feed_items_to_states(
        self,
        states: list[_OnlineFLKMeansState],
        run_rngs: list[np.random.Generator],
        items: list[_WeightedPoint],
        median_limit: int,
        cost_limit: float,
    ) -> None:
        for item in items:
            any_active = False

            for st, rrng in zip(states, run_rngs):
                if st.stopped:
                    continue
                any_active = True

                # Save old state so we can revert the last overflowing point
                prev_opened = st.num_opened
                prev_cost = st.total_cost
                prev_processed = st.processed_items
                prev_raw_read = st.raw_points_read
                prev_centers = [c.copy() for c in st.centers]
                prev_counts = list(st.counts)

                st.process_point(item, rrng)

                overflow = (st.num_opened > median_limit) or (st.total_cost > cost_limit)
                if overflow:
                    st.num_opened = prev_opened
                    st.total_cost = prev_cost
                    st.processed_items = prev_processed
                    st.raw_points_read = prev_raw_read
                    st.centers = prev_centers
                    st.counts = prev_counts
                    st.stopped = True
                    st.stop_reason = "threshold_exceeded"

            if not any_active:
                break

    def _run_one_phase_chunked(
        self,
        X: np.ndarray,
        raw_start_idx: int,
        summary_points: list[_WeightedPoint],
        Li: float,
        k: int,
        rng: np.random.Generator,
    ) -> tuple[list[_WeightedPoint], int, dict]:
        n = X.shape[0]

        states, run_rngs, facility_cost, median_limit, cost_limit, num_runs = self._init_phase_states(
            Li=Li, k=k, n=n, rng=rng
        )

        # Feed the carried summary Mi from the previous phase
        if summary_points:
            self._feed_items_to_states(
                states=states,
                run_rngs=run_rngs,
                items=summary_points,
                median_limit=median_limit,
                cost_limit=cost_limit,
            )

        # Feed unread raw points chunk by chunk
        for start in range(raw_start_idx, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)

            chunk_items = [
                _WeightedPoint(x=X[i], w=1.0, is_raw=True)
                for i in range(start, stop)
            ]

            self._feed_items_to_states(
                states=states,
                run_rngs=run_rngs,
                items=chunk_items,
                median_limit=median_limit,
                cost_limit=cost_limit,
            )

            if all(st.stopped for st in states):
                break

        for st in states:
            if not st.stopped:
                st.stop_reason = "end_of_stream"

        # Winner = run that progressed the farthest
        winner = max(states, key=lambda s: s.processed_items)
        Cw, Ww = winner.snapshot()

        # Paper-closer behavior: carry Mi directly, no extra compression
        Mi = [
            _WeightedPoint(x=Cw[i].copy(), w=float(Ww[i]), is_raw=False)
            for i in range(Cw.shape[0])
        ]

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

        return Mi, int(winner.raw_points_read), stats

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        n, d = X.shape
        if n == 0:
            raise ValueError("X must contain at least one sample.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        # PLS-style lower bound initialization
        L = self._set_lb_kmeans(X, k) / self.beta

        raw_start_idx = 0
        phase_id = 1
        summary_points: list[_WeightedPoint] = []
        phase_summaries: list[dict] = []

        while raw_start_idx < n:
            Mi, raw_consumed, phase_stats = self._run_one_phase_chunked(
                X=X,
                raw_start_idx=raw_start_idx,
                summary_points=summary_points,
                Li=L,
                k=k,
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
                    "chunk_size": int(self.chunk_size),
                    **phase_stats,
                }
            )

            summary_points = Mi

            if raw_consumed <= 0:
                # Safety against deadlock
                summary_points.append(
                    _WeightedPoint(x=X[raw_start_idx].copy(), w=1.0, is_raw=False)
                )
                raw_start_idx += 1
            else:
                raw_start_idx += raw_consumed

            L *= self.beta
            phase_id += 1

        Csum = np.vstack([p.x for p in summary_points]).astype(np.float64, copy=False)
        Wsum = np.asarray([p.w for p in summary_points], dtype=np.float64)

        # Final weighted k-means reduction to exactly k centers
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
            # Padding only for downstream compatibility
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
            memory=state_bytes,
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
                "avg_update_ms": float((t1 - t0) * 1000.0 / max(1, n)),
                "chunk_size": int(self.chunk_size),
                "phase_summaries": phase_summaries,
                "beta": float(self.beta),
                "gamma": float(self.gamma),
            },
        )