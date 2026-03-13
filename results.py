from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class Result:
    centers: np.ndarray
    runtime_sec: float
    memory: float
    cost_sse: float
    cost_ratio_vs_kmeans: float
    ari: Optional[float] = None
    nmi: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

class Algo:
    name: str
    def fit(self, samples, k, rng, labels=None) -> Result:  # pragma: no cover
        raise NotImplementedError