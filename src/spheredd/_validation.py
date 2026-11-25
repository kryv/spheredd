"""Validation helpers for bundled spherical designs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def compositions(total: int, length: int, prefix: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    """Yield non-negative integer tuples of ``length`` summing to ``total``."""
    if length == 1:
        yield prefix + (total,)
        return

    for value in range(total + 1):
        yield from compositions(total - value, length - 1, prefix + (value,))


def unit_sphere_monomial_moment(exponents: tuple[int, ...], rank: int) -> float:
    """Return the normalized unit-sphere moment of one monomial."""
    if any(exponent % 2 for exponent in exponents):
        return 0.0

    order = sum(exponents)
    numerator = 1.0
    for exponent in exponents:
        for odd in range(1, exponent, 2):
            numerator *= odd

    denominator = 1.0
    for index in range(order // 2):
        denominator *= rank + 2 * index

    return numerator / denominator


def spherical_design_max_error(points: np.ndarray, max_order: int) -> float:
    """Return the largest norm or monomial residual through ``max_order``."""
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or len(points) == 0:
        raise ValueError("points must have shape (n_points, rank)")
    if not isinstance(max_order, (int, np.integer)) or max_order < 1:
        raise ValueError("max_order must be a positive integer")

    rank = points.shape[1]
    norms = np.einsum("ni,ni->n", points, points, optimize=True)
    maximum_error = float(np.max(np.abs(norms - 1.0)))

    for order in range(1, int(max_order) + 1):
        for exponents in compositions(order, rank):
            exponent_array = np.asarray(exponents, dtype=np.int64)
            observed = float(np.mean(np.prod(points**exponent_array, axis=1)))
            expected = unit_sphere_monomial_moment(exponents, rank)
            maximum_error = max(maximum_error, abs(observed - expected))

    return maximum_error


def validate_spherical_design(
    points: np.ndarray,
    *,
    max_order: int,
    name: str,
    atol: float,
) -> float:
    """Validate a bundled design and return its largest residual."""
    error = spherical_design_max_error(points, max_order=max_order)
    if not np.isfinite(error) or error > atol:
        raise RuntimeError(
            f"{name} failed spherical-design validation: maximum residual "
            f"{error:.3e} exceeds {atol:.3e}"
        )
    return error
