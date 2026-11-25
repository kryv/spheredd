"""Optional PyTorch implementation.

This module is imported lazily by :mod:`spheredd._api`; importing ``spheredd``
does not require PyTorch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._designs import VALID_DESIGN_ORDERS, get_base_design


def _floating_dtype(value: Any, dtype: torch.dtype | None) -> torch.dtype:
    if dtype is not None:
        result = dtype
    elif torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
        result = value.dtype
    else:
        result = torch.get_default_dtype()

    if result not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("dtype must be a real floating-point torch dtype")
    if result in (torch.float16, torch.bfloat16):
        result = torch.float32
    return result


def _effective_tolerance(dtype: torch.dtype, tol: float) -> float:
    return max(float(tol), 100.0 * float(torch.finfo(dtype).eps))


def _random_orthogonal(
    rank: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    matrix = torch.randn((rank, rank), dtype=dtype, device=device, generator=generator)
    rotation, upper = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(upper))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return rotation * signs


def _spread_rotated_copies(
    base: torch.Tensor,
    copies: int,
    rotation_candidates: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if copies < 1:
        raise ValueError("copies must be positive")

    blocks = [base.clone()]
    current = base.clone()
    for _ in range(1, copies):
        best_block: torch.Tensor | None = None
        best_distance = -float("inf")
        for _ in range(rotation_candidates):
            rotation = _random_orthogonal(
                base.shape[1],
                dtype=base.dtype,
                device=base.device,
                generator=generator,
            )
            candidate = base @ rotation
            max_dot = float(torch.max(candidate @ current.mT).detach().cpu())
            min_distance_squared = 2.0 - 2.0 * max_dot
            if min_distance_squared > best_distance:
                best_distance = min_distance_squared
                best_block = candidate

        if best_block is None:  # pragma: no cover - guarded by validation
            raise RuntimeError("failed to generate a rotated design block")
        blocks.append(best_block)
        current = torch.cat((current, best_block), dim=0)

    return torch.cat(blocks, dim=0)


def spherical_design_torch(
    covariance: Any,
    n_points: int,
    *,
    mean: Any | None,
    radius: float,
    design_order: int,
    rotation_candidates: int,
    seed: int | None,
    tol: float,
    dtype: torch.dtype | None,
    device: Any | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """PyTorch backend for :func:`spheredd.spherical_design`."""
    if not isinstance(design_order, (int, np.integer)) or int(design_order) not in VALID_DESIGN_ORDERS:
        raise ValueError("design_order must be 2, 4, or 6")
    if not isinstance(n_points, (int, np.integer)) or int(n_points) < 1:
        raise ValueError("n_points must be a positive integer")
    if not isinstance(rotation_candidates, (int, np.integer)) or int(rotation_candidates) < 1:
        raise ValueError("rotation_candidates must be a positive integer")

    if device is None and torch.is_tensor(covariance):
        result_device = covariance.device
    else:
        result_device = torch.device("cpu" if device is None else device)
    result_dtype = _floating_dtype(covariance, dtype)
    covariance_tensor = torch.as_tensor(covariance, dtype=result_dtype, device=result_device)

    if covariance_tensor.ndim != 2 or covariance_tensor.shape[0] != covariance_tensor.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not bool(torch.isfinite(covariance_tensor).all()):
        raise ValueError("covariance must contain finite values")

    effective_tol = _effective_tolerance(result_dtype, tol)
    effective_rtol = max(1e-10, 100.0 * float(torch.finfo(result_dtype).eps))
    if not torch.allclose(
        covariance_tensor,
        covariance_tensor.mT,
        rtol=effective_rtol,
        atol=effective_tol,
    ):
        raise ValueError("covariance must be symmetric")

    covariance_tensor = 0.5 * (covariance_tensor + covariance_tensor.mT)
    dimension = covariance_tensor.shape[0]

    if mean is None:
        mean_tensor = torch.zeros(dimension, dtype=result_dtype, device=result_device)
    else:
        mean_tensor = torch.as_tensor(mean, dtype=result_dtype, device=result_device)
    if mean_tensor.shape != (dimension,):
        raise ValueError(f"mean must have shape ({dimension},)")
    if not bool(torch.isfinite(mean_tensor).all()):
        raise ValueError("mean must contain finite values")

    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_tensor)
    scale = max(1.0, float(torch.max(torch.abs(eigenvalues)).detach().cpu()))
    if float(torch.min(eigenvalues).detach().cpu()) < -effective_tol * scale:
        raise ValueError("covariance must be positive semidefinite")

    active = eigenvalues > effective_tol * scale
    rank = int(torch.sum(active).detach().cpu())
    design_order = int(design_order)
    n_points = int(n_points)

    if rank == 0:
        points = mean_tensor.unsqueeze(0).repeat(n_points, 1)
        weights = torch.full(
            (n_points,),
            1.0 / n_points,
            dtype=result_dtype,
            device=result_device,
        )
        info = {
            "backend": "torch",
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
            "directions": torch.empty((n_points, 0), dtype=result_dtype, device=result_device),
            "dtype": str(result_dtype),
            "device": str(result_device),
        }
        return points, weights, info

    base_design = get_base_design(rank, design_order)
    base = torch.as_tensor(base_design.points.copy(), dtype=result_dtype, device=result_device)
    base_size = len(base)
    copies = int(np.ceil(n_points / base_size))

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=result_device)
        generator.manual_seed(int(seed))

    directions = _spread_rotated_copies(
        base=base,
        copies=copies,
        rotation_candidates=int(rotation_candidates),
        generator=generator,
    )
    factor = eigenvectors[:, active] * torch.sqrt(eigenvalues[active])
    points = mean_tensor + radius * directions @ factor.mT
    weights = torch.full(
        (len(points),),
        1.0 / len(points),
        dtype=result_dtype,
        device=result_device,
    )

    info = {
        "backend": "torch",
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
        "device": str(result_device),
    }
    return points, weights, info


def evaluate_moments_torch(
    points: Any,
    weights: Any | None,
    *,
    max_order: int,
    radius: float | None,
    source_rank: int | None,
    dtype: torch.dtype | None,
    device: Any | None,
) -> dict[str, Any]:
    """PyTorch backend for :func:`spheredd.evaluate_moments`."""
    if device is None and torch.is_tensor(points):
        result_device = points.device
    else:
        result_device = torch.device("cpu" if device is None else device)
    result_dtype = _floating_dtype(points, dtype)
    points_tensor = torch.as_tensor(points, dtype=result_dtype, device=result_device)

    if points_tensor.ndim != 2 or len(points_tensor) == 0:
        raise ValueError("points must have shape (n_points, dimension)")
    if not bool(torch.isfinite(points_tensor).all()):
        raise ValueError("points must contain finite values")
    if not isinstance(max_order, (int, np.integer)) or not 2 <= int(max_order) <= 6:
        raise ValueError("max_order must be an integer from 2 to 6")

    n_points = len(points_tensor)
    if weights is None:
        weight_tensor = torch.full(
            (n_points,),
            1.0 / n_points,
            dtype=result_dtype,
            device=result_device,
        )
    else:
        weight_tensor = torch.as_tensor(weights, dtype=result_dtype, device=result_device)
        if weight_tensor.shape != (n_points,):
            raise ValueError("weights must have shape (n_points,)")
        if not bool(torch.isfinite(weight_tensor).all()):
            raise ValueError("weights must contain finite values")
        if bool(torch.any(weight_tensor < 0.0)):
            raise ValueError("weights must be non-negative")
        if float(torch.sum(weight_tensor).detach().cpu()) <= 0.0:
            raise ValueError("weights must have a positive sum")
        weight_tensor = weight_tensor / torch.sum(weight_tensor)

    centroid = weight_tensor @ points_tensor
    centered = points_tensor - centroid
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
    moments: dict[int, torch.Tensor] = {}
    for order in range(1, int(max_order) + 1):
        output_labels = labels[:order]
        equation = "p," + ",".join(f"p{label}" for label in output_labels) + "->" + output_labels
        moments[order] = torch.einsum(equation, weight_tensor, *([centered] * order))

    return {"centroid": centroid, "covariance": moments[2], "moments": moments}
