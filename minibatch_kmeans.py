from __future__ import annotations
import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from results import Algo, Result
from utils import kmeans_cost_sse, assign_labels

class MiniBatchKMeansAlgo(Algo):
    name = "MiniBatchKMeans(sk)"

    def __init__(self, batch_size: int = 2048, max_iter: int = 200):
        self.batch_size = batch_size
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, k: int, rng: np.random.Generator, y=None) -> Result:
        t0 = time.perf_counter()
        mbk = MiniBatchKMeans(
            n_clusters=k,
            batch_size=self.batch_size,
            random_state=int(rng.integers(1, 1_000_000)),
            max_iter=self.max_iter,
            n_init="auto"
        )
        mbk.fit(X)
        t1 = time.perf_counter()

        centers = mbk.cluster_centers_
        cost = kmeans_cost_sse(X, centers)
        labels = assign_labels(X, centers)

        ari = adjusted_rand_score(y, labels) if y is not None else None
        nmi = normalized_mutual_info_score(y, labels) if y is not None else None

        return Result(centers=centers, runtime_sec=t1 - t0, cost_sse=cost, cost_ratio_vs_kmeans=float("nan"), ari=ari, nmi=nmi)