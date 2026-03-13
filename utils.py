from __future__ import annotations
import math
import numpy as np
from sklearn.cluster import KMeans

def set_seed(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)

def assign_labels(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1)

def kmeans_cost_sse(X: np.ndarray, centers: np.ndarray) -> float:
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.sum(np.min(dist2, axis=1)))

def weighted_kmeans_centers(
    X: np.ndarray,
    w: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 5,
    max_iter: int = 200
) -> np.ndarray:
    km = KMeans(
        n_clusters=k,
        n_init=n_init,
        max_iter=max_iter,
        random_state=int(rng.integers(1, 1_000_000))
    )
    km.fit(X, sample_weight=w)
    return km.cluster_centers_

def kmeanspp_init_weighted(
    X: np.ndarray,
    w: np.ndarray,
    k: int,
    rng: np.random.Generator
) -> np.ndarray:
    n = X.shape[0]
    idx = []

    p0 = w / (w.sum() + 1e-12)
    idx.append(int(rng.choice(n, p=p0)))

    d2 = np.sum((X - X[idx[0]]) ** 2, axis=1)
    for _ in range(1, k):
        probs = w * d2
        s = probs.sum()
        if s <= 1e-12:
            idx.append(int(rng.integers(0, n)))
        else:
            probs = probs / s
            idx.append(int(rng.choice(n, p=probs)))
        new_d2 = np.sum((X - X[idx[-1]]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)

    return X[np.array(idx)]

def d2_sample(
    samples: np.ndarray,
    centers: np.ndarray | None,
    m: int,
    rng: np.random.Generator
) -> np.ndarray:
    if centers is None or len(centers) == 0:
        take = min(m, samples.shape[0])
        return samples[rng.choice(samples.shape[0], size=take, replace=False)]

    diff = samples[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    dmin = np.min(dist2, axis=1)
    s = dmin.sum()
    if s <= 1e-12:
        take = min(m, samples.shape[0])
        return samples[rng.choice(samples.shape[0], size=take, replace=False)]
    p = dmin / s
    take = min(m, samples.shape[0])
    return samples[rng.choice(samples.shape[0], size=take, replace=False, p=p)]

def compress_coreset(
    Xc: np.ndarray,
    wc: np.ndarray,
    target_size: int,
    k: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = Xc.shape[0]
    if n <= target_size:
        return Xc, wc

    init_k = min(k, max(2, int(math.sqrt(target_size))))
    init_centers = kmeanspp_init_weighted(Xc, wc, k=init_k, rng=rng)
    sampled = d2_sample(Xc, init_centers, m=target_size, rng=rng)

    total_w = float(np.sum(wc))
    w_new = np.full(sampled.shape[0], total_w / sampled.shape[0], dtype=np.float64)
    return sampled, w_new