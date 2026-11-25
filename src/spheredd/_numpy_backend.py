"""NumPy implementation of spherical-design generation and moment evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._designs import VALID_DESIGN_ORDERS, get_base_design


def _floating_dtype(value: Any, dtype: Any | None) -> np.dtype:
    if dtype is not None:
        result = np.dtype(dtype)
    else:
        source = np.asarray(value)
        result = source.dtype if np.issubdtype(source.dtype, np.floating) else np.dtype(np.float64)

    if not np.issubdtype(result, np.floating):
        raise TypeError("dtype must be a real floating-point dtype")
    if result.itemsize < np.dtype(np.float32).itemsize:
        result = np.dtype(np.float32)
    return result


def _effective_tolerance(dtype: np.dtype, tol: float) -> float:
    return max(float(tol), 100.0 * float(np.finfo(dtype).eps))


def _random_orthogonal(rank: int, rng: np.random.Generator, dtype: np.dtype) -> np.ndarray:
    matrix = rng.normal(size=(rank, rank)).astype(dtype, copy=False)
    rotation, upper = np.linalg.qr(matrix)
    signs = np.sign(np.diag(upper))
    signs[signs == 0.0] = 1.0
    return rotation * signs


def _spread_rotated_copies(
    base: np.ndarray,
    copies: int,
    rng: np.random.Generator,
    rotation_candidates: int,
) -> np.ndarray:
    if copies < 1:
        raise ValueError("copies must be positive")

    blocks = [base.copy()]
    current = base.copy()
    for _ in range(1, copies):
        best_block: np.ndarray | None = None
        best_distance = -np.inf
        for _ in range(rotation_candidates):
            rotation = _random_orthogonal(base.shape[1], rng, base.dtype)
            candidate = base @ rotation
            max_dot = float(np.max(candidate @ current.T))
            min_distance_squared = 2.0 - 2.0 * max_dot
            if min_distance_squared > best_distance:
                best_distance = min_distance_squared
                best_block = candidate

        if best_block is None:  # pragma: no cover - guarded by validation
            raise RuntimeError("failed to generate a rotated design block")
        blocks.append(best_block)
        current = np.vstack((current, best_block))

    return np.vstack(blocks)


def spherical_design_numpy(
    covariance: Any,
    n_points: int,
    *,
    mean: Any | None,
    radius: float,
    design_order: int,
    rotation_candidates: int,
    seed: int | None,
    tol: float,
    dtype: Any | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """NumPy backend for :func:`spheredd.spherical_design`."""
    if not isinstance(design_order, (int, np.integer)) or int(design_order) not in VALID_DESIGN_ORDERS:
        raise ValueError("design_order must be 2, 4, or 6")
    if not isinstance(n_points, (int, np.integer)) or int(n_points) < 1:
        raise ValueError("n_points must be a positive integer")
    if not isinstance(rotation_candidates, (int, np.integer)) or int(rotation_candidates) < 1:
        raise ValueError("rotation_candidates must be a positive integer")

    result_dtype = _floating_dtype(covariance, dtype)
    covariance = np.asarray(covariance, dtype=result_dtype)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must contain finite values")

    effective_tol = _effective_tolerance(result_dtype, tol)
    effective_rtol = max(1e-10, 100.0 * float(np.finfo(result_dtype).eps))
    if not np.allclose(covariance, covariance.T, rtol=effective_rtol, atol=effective_tol):
        raise ValueError("covariance must be symmetric")

    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]

    if mean is None:
        mean_array = np.zeros(dimension, dtype=result_dtype)
    else:
        mean_array = np.asarray(mean, dtype=result_dtype)
    if mean_array.shape != (dimension,):
        raise ValueError(f"mean must have shape ({dimension},)")
    if not np.all(np.isfinite(mean_array)):
        raise ValueError("mean must contain finite values")

    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -effective_tol * scale:
        raise ValueError("covariance must be positive semidefinite")

    active = eigenvalues > effective_tol * scale
    rank = int(np.sum(active))
    design_order = int(design_order)
    n_points = int(n_points)

    if rank == 0:
        points = np.repeat(mean_array[None, :], n_points, axis=0)
        weights = np.full(n_points, 1.0 / n_points, dtype=result_dtype)
        info = {
            "backend": "numpy",
            "rank": 0,
            "radius": radius,
            "design_order": design_order,
            "base_design_strength": design_order,
            "base_construction": "degenerate zero-covariance design",
            "base_design_exact": True,
            "base_cardinality_status": "degenerate",
            "base_validation_max_error": 0.0,
            "base_references": (),
            "base_size": 1,
            "copies": n_points,
            "requested_n_points": n_points,
            "actual_n_points": n_points,
            "directions": np.empty((n_points, 0), dtype=result_dtype),
            "dtype": str(result_dtype),
        }
        return points, weights, info

    base_design = get_base_design(rank, design_order)
    base = np.asarray(base_design.points, dtype=result_dtype)
    base_size = len(base)
    copies = int(np.ceil(n_points / base_size))
    rng = np.random.default_rng(seed)
    directions = _spread_rotated_copies(
        base=base,
        copies=copies,
        rng=rng,
        rotation_candidates=int(rotation_candidates),
    )

    factor = eigenvectors[:, active] * np.sqrt(eigenvalues[active])
    points = mean_array + result_dtype.type(radius) * directions @ factor.T
    weights = np.full(len(points), 1.0 / len(points), dtype=result_dtype)

    info = {
        "backend": "numpy",
        "rank": rank,
        "radius": radius,
        "design_order": design_order,
        "base_design_strength": base_design.strength,
        "base_construction": base_design.construction,
        "base_design_exact": base_design.exact,
        "base_cardinality_status": base_design.cardinality_status,
        "base_validation_max_error": base_design.validation_max_error,
        "base_references": base_design.references,
        "base_size": base_size,
        "copies": copies,
        "requested_n_points": n_points,
        "actual_n_points": len(points),
        "directions": directions,
        "dtype": str(result_dtype),
    }
    return points, weights, info


def evaluate_moments_numpy(
    points: Any,
    weights: Any | None,
    *,
    max_order: int,
    radius: float | None,
    source_rank: int | None,
    dtype: Any | None,
) -> dict[str, Any]:
    """NumPy backend for :func:`spheredd.evaluate_moments`."""
    result_dtype = _floating_dtype(points, dtype)
    points_array = np.asarray(points, dtype=result_dtype)
    if points_array.ndim != 2 or len(points_array) == 0:
        raise ValueError("points must have shape (n_points, dimension)")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("points must contain finite values")
    if not isinstance(max_order, (int, np.integer)) or not 2 <= int(max_order) <= 6:
        raise ValueError("max_order must be an integer from 2 to 6")

    n_points = len(points_array)
    if weights is None:
        weight_array = np.full(n_points, 1.0 / n_points, dtype=result_dtype)
    else:
        weight_array = np.asarray(weights, dtype=result_dtype)
        if weight_array.shape != (n_points,):
            raise ValueError("weights must have shape (n_points,)")
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must contain finite values")
        if np.any(weight_array < 0.0):
            raise ValueError("weights must be non-negative")
        if float(weight_array.sum()) <= 0.0:
            raise ValueError("weights must have a positive sum")
        weight_array = weight_array / weight_array.sum()

    centroid = weight_array @ points_array
    centered = points_array - centroid
    if radius is not None:
        if source_rank is None:
            raise ValueError("source_rank is required when radius is given")
        radius = float(radius)
        source_rank = int(source_rank)
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be positive and finite")
        if source_rank < 1:
            raise ValueError("source_rank must be a positive integer")
        centered = centered * (np.sqrt(source_rank) / radius)

    labels = "abcdefghijklmnopqrstuvwxyz"
    moments: dict[int, np.ndarray] = {}
    for order in range(1, int(max_order) + 1):
        output_labels = labels[:order]
        equation = "p," + ",".join(f"p{label}" for label in output_labels) + "->" + output_labels
        moments[order] = np.einsum(
            equation,
            weight_array,
            *([centered] * order),
            optimize=True,
        )

    return {"centroid": centroid, "covariance": moments[2], "moments": moments}
