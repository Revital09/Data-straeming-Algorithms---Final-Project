from __future__ import annotations

import math
import time

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result


def _weighted_kmeans_centers(
    X: np.ndarray,
    w: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 5,
    max_iter: int = 300,
) -> np.ndarray:
    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=int(rng.integers(1, 1_000_000)),
    )
    km.fit(X, sample_weight=w)
    return km.cluster_centers_


class _OnlineFLKMeansState:
    __slots__ = (
        "facility_cost",
        "d",
        "max_centers",
        "centers",
        "center_sq_norms",
        "counts",
        "total_cost",
        "num_opened",
        "processed_items",
        "raw_points_read",
        "stopped",
        "stop_reason",
        "_undo_prev_total_cost",
        "_undo_prev_processed",
        "_undo_prev_raw_read",
        "_undo_action",
        "_undo_index",
        "_undo_prev_count",
    )

    def __init__(self, facility_cost: float, d: int, max_centers: int):
        self.facility_cost = float(facility_cost)
        self.d = int(d)
        self.max_centers = int(max_centers)

        self.centers = np.zeros((max_centers, d), dtype=np.float64)
        self.center_sq_norms = np.zeros(max_centers, dtype=np.float64)
        self.counts = np.zeros(max_centers, dtype=np.float64)

        self.total_cost = 0.0
        self.num_opened = 0
        self.processed_items = 0
        self.raw_points_read = 0

        self.stopped = False
        self.stop_reason: str | None = None

        self._undo_prev_total_cost = 0.0
        self._undo_prev_processed = 0
        self._undo_prev_raw_read = 0
        self._undo_action = 0
        self._undo_index = -1
        self._undo_prev_count = 0.0

    def process_point(
        self,
        x: np.ndarray,
        x_norm2: float,
        w: float,
        is_raw: bool,
        rng: np.random.Generator,
    ) -> None:
        w = float(w)

        self._undo_prev_total_cost = self.total_cost
        self._undo_prev_processed = self.processed_items
        self._undo_prev_raw_read = self.raw_points_read
        self._undo_action = 0
        self._undo_index = -1
        self._undo_prev_count = 0.0

        num_opened = self.num_opened
        if num_opened == 0:
            self.centers[0] = x
            self.center_sq_norms[0] = x_norm2
            self.counts[0] = w
            self.num_opened = 1
            self._undo_action = 1
            self._undo_index = 0
        else:
            active_centers = self.centers[:num_opened]
            sq_dists = self.center_sq_norms[:num_opened] + x_norm2 - 2.0 * (active_centers @ x)
            np.maximum(sq_dists, 0.0, out=sq_dists)

            j = int(np.argmin(sq_dists))
            d2 = float(sq_dists[j])
            p_open = (w * d2) / (self.facility_cost + 1e-12)

            if p_open >= 1.0 or rng.random() < p_open:
                idx = num_opened
                self.centers[idx] = x
                self.center_sq_norms[idx] = x_norm2
                self.counts[idx] = w
                self.num_opened = num_opened + 1
                self._undo_action = 1
                self._undo_index = idx
            else:
                self._undo_action = 2
                self._undo_index = j
                self._undo_prev_count = float(self.counts[j])
                self.counts[j] += w
                self.total_cost += w * d2

        self.processed_items += 1
        if is_raw:
            self.raw_points_read += 1

    def rollback_last(self) -> None:
        self.total_cost = self._undo_prev_total_cost
        self.processed_items = self._undo_prev_processed
        self.raw_points_read = self._undo_prev_raw_read

        if self._undo_action == 1:
            idx = self._undo_index
            self.counts[idx] = 0.0
            self.center_sq_norms[idx] = 0.0
            self.num_opened -= 1
        elif self._undo_action == 2:
            idx = self._undo_index
            self.counts[idx] = self._undo_prev_count

        self._undo_action = 0
        self._undo_index = -1
        self._undo_prev_count = 0.0

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        if self.num_opened == 0:
            raise RuntimeError("Invocation has no centers.")
        return (
            self.centers[:self.num_opened].copy(),
            self.counts[:self.num_opened].copy(),
        )


class Charikar_KMeans(Algo):
    name = "[Charikar2003] PLS KMeans"

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

    @staticmethod
    def _squared_norms(X: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", X, X, optimize=True)

    @staticmethod
    def _squared_distances(
        X: np.ndarray,
        X_sq_norms: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        center_sq_norms = np.einsum("ij,ij->i", centers, centers, optimize=True)
        dist2 = X_sq_norms[:, None] + center_sq_norms[None, :]
        dist2 -= 2.0 * (X @ centers.T)
        np.maximum(dist2, 0.0, out=dist2)
        return dist2

    def _assign_and_cost(self, X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, float]:
        X_sq_norms = self._squared_norms(X)
        dist2 = self._squared_distances(X, X_sq_norms, centers)
        pred = np.argmin(dist2, axis=1)
        cost = float(np.sum(dist2[np.arange(X.shape[0]), pred]))
        return pred, cost

    def _set_lb_kmeans(self, X: np.ndarray, k: int) -> float:
        m = min(X.shape[0], k + 1)
        if m <= 1:
            return 1.0

        Y = X[:m]
        y_sq = self._squared_norms(Y)
        d2 = y_sq[:, None] + y_sq[None, :] - 2.0 * (Y @ Y.T)
        np.maximum(d2, 0.0, out=d2)
        np.fill_diagonal(d2, np.inf)
        return max(float(np.min(d2)), 1e-12)

    def _init_phase_states(
        self,
        Li: float,
        k: int,
        n: int,
        d: int,
        rng: np.random.Generator,
    ):
        logn = max(1.0, math.log(max(2, n)))
        num_runs = max(1, int(math.ceil(2.0 * logn)))

        facility_cost = Li / (k * (1.0 + logn) + 1e-12)
        median_limit = int(math.ceil(
            4.0 * k * (1.0 + logn) * (1.0 + 4.0 * (self.gamma + self.beta))
        ))
        cost_limit = 4.0 * Li * (1.0 + 4.0 * (self.gamma + self.beta))
        max_centers = max(1, median_limit + 1)

        states = [
            _OnlineFLKMeansState(facility_cost=facility_cost, d=d, max_centers=max_centers)
            for _ in range(num_runs)
        ]
        run_seeds = rng.integers(0, 2**32 - 1, size=num_runs, dtype=np.uint64)
        run_rngs = [np.random.default_rng(int(seed)) for seed in run_seeds]

        return states, run_rngs, facility_cost, median_limit, cost_limit, num_runs

    def _feed_points_to_states(
        self,
        active_states: list[_OnlineFLKMeansState],
        active_rngs: list[np.random.Generator],
        points: np.ndarray,
        weights: np.ndarray,
        is_raw: bool,
        median_limit: int,
        cost_limit: float,
    ) -> tuple[list[_OnlineFLKMeansState], list[np.random.Generator]]:
        for i in range(points.shape[0]):
            if not active_states:
                break

            x = points[i]
            x_norm2 = float(np.dot(x, x))
            w = float(weights[i])

            next_states: list[_OnlineFLKMeansState] = []
            next_rngs: list[np.random.Generator] = []

            for st, rrng in zip(active_states, active_rngs):
                st.process_point(x=x, x_norm2=x_norm2, w=w, is_raw=is_raw, rng=rrng)
                overflow = (st.num_opened > median_limit) or (st.total_cost > cost_limit)
                if overflow:
                    st.rollback_last()
                    st.stopped = True
                    st.stop_reason = "threshold_exceeded"
                    continue

                next_states.append(st)
                next_rngs.append(rrng)

            active_states = next_states
            active_rngs = next_rngs

        return active_states, active_rngs

    def _run_one_phase_chunked(
        self,
        X: np.ndarray,
        raw_start_idx: int,
        summary_X: np.ndarray | None,
        summary_w: np.ndarray | None,
        Li: float,
        k: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, int, dict]:
        n, d = X.shape

        states, run_rngs, facility_cost, median_limit, cost_limit, num_runs = self._init_phase_states(
            Li=Li,
            k=k,
            n=n,
            d=d,
            rng=rng,
        )

        active_states = states[:]
        active_rngs = run_rngs[:]

        if summary_X is not None and summary_w is not None and summary_X.shape[0] > 0:
            active_states, active_rngs = self._feed_points_to_states(
                active_states=active_states,
                active_rngs=active_rngs,
                points=summary_X,
                weights=summary_w,
                is_raw=False,
                median_limit=median_limit,
                cost_limit=cost_limit,
            )

        for start in range(raw_start_idx, n, self.chunk_size):
            stop = min(start + self.chunk_size, n)
            chunk = X[start:stop]
            chunk_w = np.ones(chunk.shape[0], dtype=np.float64)

            active_states, active_rngs = self._feed_points_to_states(
                active_states=active_states,
                active_rngs=active_rngs,
                points=chunk,
                weights=chunk_w,
                is_raw=True,
                median_limit=median_limit,
                cost_limit=cost_limit,
            )

            if not active_states:
                break

        for st in active_states:
            st.stop_reason = "end_of_stream"

        winner = max(states, key=lambda s: s.processed_items)
        Mi_X, Mi_w = winner.snapshot()

        stats = {
            "facility_cost": float(facility_cost),
            "median_limit": int(median_limit),
            "cost_limit": float(cost_limit),
            "winner_processed_items": int(winner.processed_items),
            "winner_raw_points_read": int(winner.raw_points_read),
            "winner_num_centers": int(Mi_X.shape[0]),
            "winner_cost": float(winner.total_cost),
            "num_parallel_runs": int(num_runs),
        }
        return Mi_X, Mi_w, int(winner.raw_points_read), stats

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()

        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if n == 0:
            raise ValueError("X must contain at least one sample.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        L = self._set_lb_kmeans(X, k) / self.beta

        raw_start_idx = 0
        phase_id = 1
        summary_X: np.ndarray | None = None
        summary_w: np.ndarray | None = None
        phase_summaries: list[dict] = []

        while raw_start_idx < n:
            summary_in_size = 0 if summary_X is None else int(summary_X.shape[0])

            Mi_X, Mi_w, raw_consumed, phase_stats = self._run_one_phase_chunked(
                X=X,
                raw_start_idx=raw_start_idx,
                summary_X=summary_X,
                summary_w=summary_w,
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
                    "summary_in_size": int(summary_in_size),
                    "summary_out_size": int(Mi_X.shape[0]),
                    "chunk_size": int(self.chunk_size),
                    **phase_stats,
                }
            )

            summary_X = Mi_X
            summary_w = Mi_w

            if raw_consumed <= 0:
                x = X[raw_start_idx:raw_start_idx + 1]
                w = np.array([1.0], dtype=np.float64)

                if summary_X is None:
                    summary_X = x.copy()
                    summary_w = w
                else:
                    summary_X = np.vstack([summary_X, x])
                    summary_w = np.hstack([summary_w, w])

                raw_start_idx += 1
            else:
                raw_start_idx += raw_consumed

            L *= self.beta
            phase_id += 1

        assert summary_X is not None and summary_w is not None

        Csum = summary_X.astype(np.float64, copy=False)
        Wsum = summary_w.astype(np.float64, copy=False)

        if Csum.shape[0] > k:
            centers_final = _weighted_kmeans_centers(
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
            extra_idx = rng.integers(0, Csum.shape[0], size=k - Csum.shape[0])
            centers_final = np.vstack([Csum, Csum[extra_idx]])

        pred, cost = self._assign_and_cost(X, centers_final)

        ari = adjusted_rand_score(y, pred) if y is not None else None
        nmi = normalized_mutual_info_score(y, pred) if y is not None else None

        t1 = time.perf_counter()
        state_bytes = int(Csum.nbytes + Wsum.nbytes)

        return Result(
            centers=centers_final,
            runtime_sec=float(t1 - t0),
            memory=float(state_bytes),
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
