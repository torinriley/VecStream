"""Validated in-memory vectors and vectorized exact cosine search."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import DimensionMismatchError, InvalidVectorError


def as_float32_vector(
    vector: Sequence[float] | np.ndarray, dimension: int | None = None
) -> np.ndarray:
    """Convert and validate a finite 1-D vector. Zero-vector cosine is 0.0."""
    try:
        value: np.ndarray = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidVectorError("vector must contain numeric values") from exc
    if value.ndim != 1 or value.size == 0:
        raise InvalidVectorError("vector must be a non-empty one-dimensional array")
    if dimension is not None and value.size != dimension:
        raise DimensionMismatchError(
            f"vector dimension {value.size} does not match expected dimension {dimension}"
        )
    if not np.all(np.isfinite(value)):
        raise InvalidVectorError("vector values must all be finite (no NaN or infinity)")
    return np.ascontiguousarray(value)


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized float32 copy; a zero vector remains all-zero."""
    # Accumulate in float64: squaring a finite float32 near its maximum can
    # overflow in float32, while subnormal values can underflow to zero.
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    return np.asarray(vector.astype(np.float64) / norm, dtype=np.float32)


class VectorStore:
    """An in-memory vector store with exact cosine-search ground truth."""

    def __init__(self) -> None:
        self.vectors: Dict[str, np.ndarray] = {}
        self.dimension: Optional[int] = None

    def add_vector(self, id: str, vector: Sequence[float] | np.ndarray) -> None:
        value = as_float32_vector(vector, self.dimension)
        if self.dimension is None:
            self.dimension = int(value.size)
        self.vectors[str(id)] = value.copy()

    def get_vector(self, id: str) -> List[float]:
        if id not in self.vectors:
            raise KeyError(f"Vector with ID {id} not found")
        return self.vectors[id].tolist()

    def remove_vector(self, id: str) -> None:
        if id not in self.vectors:
            raise KeyError(f"Vector with ID {id} not found")
        del self.vectors[id]
        if not self.vectors:
            self.dimension = None

    def search_similar(
        self, query: Sequence[float] | np.ndarray, k: int = 5, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Return exact top-k cosine neighbors using vectorized NumPy."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if not self.vectors:
            return []
        assert self.dimension is not None
        query_value = normalize(as_float32_vector(query, self.dimension))
        ids = list(self.vectors)
        matrix = np.stack([self.vectors[item_id] for item_id in ids])
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms != 0,
        )
        scores = normalized @ query_value
        eligible = np.flatnonzero(scores >= threshold)
        order = eligible[np.argsort(-scores[eligible], kind="stable")][:k]
        return [(ids[int(i)], float(scores[i])) for i in order]

    exact_search = search_similar
