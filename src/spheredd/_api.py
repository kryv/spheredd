"""Public backend-dispatching API."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from ._numpy_backend import evaluate_moments_numpy, spherical_design_numpy

Backend = Literal["auto", "numpy", "torch"]


def _looks_like_torch(value: Any) -> bool:
    module = getattr(value.__class__, "__module__", "")
    return module == "torch" or module.startswith("torch.")


@lru_cache(maxsize=1)
def _load_torch_backend() -> Any:
    try:
        from . import _torch_backend
    except (ImportError, OSError) as exc:
        raise ImportError(
            "PyTorch backend requested, but torch could not be imported. "
            "Install a compatible PyTorch build or use backend='numpy'."
        ) from exc
    return _torch_backend


def torch_available() -> bool:
    """Return ``True`` when the optional PyTorch backend can be imported."""
    try:
        _load_torch_backend()
    except ImportError:
        return False
    return True


def _resolve_backend(value: Any, backend: Backend | None) -> Literal["numpy", "torch"]:
    selected = "auto" if backend is None else backend
    if selected not in ("auto", "numpy", "torch"):
        raise ValueError("backend must be 'auto', 'numpy', or 'torch'")
    if selected == "auto":
        return "torch" if _looks_like_torch(value) else "numpy"
    return selected


def spherical_design(
    covariance: Any,
    n_points: int,
    mean: Any | None = None,
    radius: float = 1.0,
    design_order: int = 4,
    rotation_candidates: int = 32,
    seed: int | None = None,
    tol: float = 1e-12,
    *,
    backend: Backend | None = "auto",
    dtype: Any | None = None,
    device: Any | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Generate an equal-weight spherical design shaped by a covariance matrix.

    Let ``covariance`` be a positive semidefinite matrix with numerical rank
    ``r``. The function selects a spherical design ``u_i`` on the unit sphere
    in the active ``r``-dimensional eigenspace and maps it to

    ``x_i = mean + radius * B @ u_i``,

    where ``B @ B.T`` is the rank-``r`` covariance reconstructed from the
    eigenvalues retained by the numerical-rank test. Because a unit-sphere
    design has second moment ``I / r``, the weighted covariance of the returned
    points is

    ``(radius**2 / r) * retained_covariance``.

    Set ``radius=sqrt(r)`` when the returned weighted covariance should equal
    this retained covariance. When no positive eigenvalues are discarded by
    the tolerance, it is the input covariance. Alternatively, pass the returned
    ``radius`` and ``rank`` to :func:`evaluate_moments` to recover moments in
    the canonical covariance normalization.

    ``n_points`` is a lower bound rather than an exact requested size. To
    preserve the design conditions, the function uses an integer number of
    complete base-design blocks. Additional blocks are independently rotated,
    and the best of ``rotation_candidates`` random rotations is chosen to
    spread each new block away from the points already selected.

    Parameters
    ----------
    covariance : array-like, shape (dimension, dimension)
        Symmetric positive semidefinite matrix defining the active subspace and
        ellipsoidal scaling. Its numerical rank, not its matrix dimension, is
        used to select the base design.
    n_points : int
        Minimum number of points to return. The actual number is the smallest
        available multiple of the selected base-design size that is greater
        than or equal to this value. For a zero-rank covariance, exactly
        ``n_points`` copies of ``mean`` are returned.
    mean : array-like, shape (dimension,), optional
        Center of the generated point set. The default is the zero vector.
    radius : float, default=1.0
        Positive scale applied to the unit-sphere directions before the
        covariance transformation. The resulting covariance is proportional
        to ``radius**2`` as described above.
    design_order : {2, 4, 6}, default=4
        Requested spherical-design strength. Some bundled constructions have a
        higher actual strength; that value is reported as
        ``info["base_design_strength"]``. Order-6 designs are currently
        available for numerical ranks 1 through 6.
    rotation_candidates : int, default=32
        Number of random orthogonal rotations considered for every additional
        base-design block. Larger values can improve geometric separation but
        increase generation cost. This argument has no effect when only one
        block is needed.
    seed : int, optional
        Seed for the random rotations of additional blocks. The first base
        block is deterministic. NumPy and PyTorch use different random-number
        generators, so identical seeds do not imply identical point sets
        across backends.
    tol : float, default=1e-12
        Requested absolute tolerance for symmetry, positive-semidefinite, and
        numerical-rank checks. Each backend applies a lower bound derived from
        the selected floating-point dtype. An eigenvalue is active when it is
        greater than the effective tolerance times the covariance scale.
    backend : {"auto", "numpy", "torch"}, optional
        Computational backend. With ``"auto"``, a PyTorch tensor selects the
        PyTorch backend and any other input selects NumPy. PyTorch is imported
        lazily and is not required for NumPy-only use.
    dtype : numpy.dtype or torch.dtype, optional
        Floating-point dtype of the outputs and internal linear algebra. When
        omitted, a floating input dtype is preserved where possible; otherwise
        the backend default is used. Half-precision inputs are promoted to at
        least 32-bit floating point.
    device : torch.device or str, optional
        Output device for the PyTorch backend. By default, a tensor input keeps
        its device and non-tensor input is placed on CPU. Supplying ``device``
        with the NumPy backend raises ``ValueError``.

    Returns
    -------
    points : numpy.ndarray or torch.Tensor, shape (actual_n_points, dimension)
        Covariance-shaped design points centered at ``mean``.
    weights : numpy.ndarray or torch.Tensor, shape (actual_n_points,)
        Equal non-negative weights summing to one.
    info : dict
        Metadata describing the selected construction. Important entries are:

        ``backend``
            ``"numpy"`` or ``"torch"``.
        ``rank``
            Numerical rank of ``covariance``.
        ``design_order`` and ``base_design_strength``
            Requested order and actual strength of the base construction.
        ``base_size``, ``copies``, and ``actual_n_points``
            Base cardinality, number of complete blocks, and returned size.
        ``base_construction`` and ``base_cardinality_status``
            Description of the construction and the status of its known
            cardinality or minimality.
        ``base_design_exact`` and ``base_validation_max_error``
            Whether the coordinates are analytic/algebraic rather than an
            embedded numerical solution, and the validation residual of the
            bundled base coordinates.
        ``base_references``
            Literature references associated with the base construction.
        ``directions``
            Unit-sphere design directions in the active rank-dimensional
            subspace, before covariance shaping.

    Raises
    ------
    ValueError
        If an argument has an invalid shape or value, if ``covariance`` is not
        symmetric positive semidefinite within tolerance, or if the requested
        order is unavailable for the detected rank.
    TypeError
        If ``dtype`` is not a supported real floating-point dtype.
    ImportError
        If the PyTorch backend is requested but PyTorch cannot be imported.

    Notes
    -----
    A union of complete equal-weight designs of the same strength remains a
    design of that strength, which is why block rounding preserves the moment
    conditions. Rank-5 and rank-6 order-6 base designs are bundled numerical
    solutions; ``float64`` is recommended when high-order residual accuracy is
    important.

    For the PyTorch backend, the covariance eigendecomposition participates in
    autograd. Gradients can become ill-conditioned at repeated eigenvalues or
    when the numerical rank changes.

    Examples
    --------
    Generate a rank-4 design whose weighted covariance equals the identity:

    >>> import numpy as np
    >>> from spheredd import spherical_design
    >>> rank = 4
    >>> points, weights, info = spherical_design(
    ...     np.eye(rank),
    ...     n_points=1,
    ...     design_order=4,
    ...     radius=np.sqrt(rank),
    ... )
    >>> points.shape
    (20, 4)
    >>> np.allclose(weights.sum(), 1.0)
    True
    >>> np.allclose((points * weights[:, None]).sum(axis=0), 0.0, atol=1e-12)
    True
    >>> info["rank"], info["base_size"]
    (4, 20)
    """
    selected = _resolve_backend(covariance, backend)
    if selected == "numpy":
        if device is not None:
            raise ValueError("device is only valid for the PyTorch backend")
        return spherical_design_numpy(
            covariance,
            n_points,
            mean=mean,
            radius=radius,
            design_order=design_order,
            rotation_candidates=rotation_candidates,
            seed=seed,
            tol=tol,
            dtype=dtype,
        )

    torch_backend = _load_torch_backend()
    return torch_backend.spherical_design_torch(
        covariance,
        n_points,
        mean=mean,
        radius=radius,
        design_order=design_order,
        rotation_candidates=rotation_candidates,
        seed=seed,
        tol=tol,
        dtype=dtype,
        device=device,
    )


def spherical_2_design(covariance: Any, n_points: int, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Convenience wrapper for ``design_order=2``."""
    return spherical_design(covariance, n_points, design_order=2, **kwargs)


def spherical_4_design(covariance: Any, n_points: int, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Convenience wrapper for ``design_order=4``."""
    return spherical_design(covariance, n_points, design_order=4, **kwargs)


def spherical_6_design(covariance: Any, n_points: int, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Convenience wrapper for ``design_order=6``."""
    return spherical_design(covariance, n_points, design_order=6, **kwargs)


def evaluate_moments(
    points: Any,
    weights: Any | None = None,
    *,
    max_order: int = 4,
    radius: float | None = 1.0,
    source_rank: int | None = None,
    backend: Backend | None = "auto",
    dtype: Any | None = None,
    device: Any | None = None,
) -> dict[str, Any]:
    """Evaluate weighted central-moment tensors through a requested order.

    The function first normalizes ``weights`` to sum to one, computes the
    weighted centroid, and centers all points around that centroid. It then
    returns central moments of every order from 1 through ``max_order``. The
    order-``k`` tensor has shape ``(dimension,) * k`` and entries

    ``M[i1, ..., ik] = sum_n w[n] * y[n, i1] * ... * y[n, ik]``,

    where ``y`` denotes the centered coordinates after the optional canonical
    rescaling described below.

    By default, ``radius`` is not ``None``. In that mode, ``source_rank`` is
    required and centered coordinates are multiplied by
    ``sqrt(source_rank) / radius`` before the moments are evaluated. This
    reverses the radial normalization used by :func:`spherical_design`: when
    ``points`` were produced from a covariance matrix, the returned second
    moment is the retained numerical-rank approximation of that matrix, up to
    numerical error. Set ``radius=None`` to obtain moments directly in the
    physical coordinate units of ``points``.

    Parameters
    ----------
    points : array-like, shape (n_points, dimension)
        Non-empty collection of sample points. A PyTorch tensor selects the
        PyTorch backend when ``backend="auto"``.
    weights : array-like, shape (n_points,), optional
        Non-negative sample weights. They need not already sum to one because
        the function normalizes them internally. Equal weights are used when
        this argument is omitted.
    max_order : int, default=4
        Highest central-moment order to compute. Supported values are 2 through
        6. The returned ``moments`` mapping also includes all lower orders,
        including the first central moment.
    radius : float or None, default=1.0
        Radius used when the points were generated. When not ``None``, centered
        coordinates are multiplied by ``sqrt(source_rank) / radius``. Pass
        ``None`` to disable this scaling and evaluate raw central moments.
    source_rank : int, optional
        Rank of the source spherical design. Required when ``radius`` is not
        ``None``. For output from :func:`spherical_design`, use
        ``info["rank"]``.
    backend : {"auto", "numpy", "torch"}, optional
        Computational backend. With ``"auto"``, a PyTorch tensor selects the
        PyTorch backend and any other input selects NumPy. PyTorch is imported
        lazily.
    dtype : numpy.dtype or torch.dtype, optional
        Floating-point dtype used for the calculation and returned tensors.
        When omitted, a floating input dtype is preserved where possible.
        Half-precision inputs are promoted to at least 32-bit floating point.
    device : torch.device or str, optional
        Device used by the PyTorch backend. By default, tensor inputs keep their
        current device and non-tensor inputs use CPU. This argument is invalid
        for the NumPy backend.

    Returns
    -------
    result : dict
        Dictionary containing:

        ``centroid``
            Weighted centroid in the original, unscaled coordinate system,
            with shape ``(dimension,)``.
        ``covariance``
            Alias for ``moments[2]``. Despite the key name, this is the
            population-style weighted second central moment; no Bessel
            correction is applied.
        ``moments``
            Mapping from each integer order ``1`` through ``max_order`` to its
            full central-moment tensor. If canonical scaling is enabled, these
            tensors are expressed in the rescaled coordinates.

    Raises
    ------
    ValueError
        If ``points`` or ``weights`` have invalid shapes or non-finite values,
        if weights are negative or sum to zero, if ``max_order`` is outside
        2 through 6, if canonical scaling parameters are invalid, or if
        ``device`` is supplied to the NumPy backend.
    TypeError
        If ``dtype`` is not a supported real floating-point dtype.
    ImportError
        If the PyTorch backend is requested but PyTorch cannot be imported.

    Notes
    -----
    The first central moment is zero in exact arithmetic and is returned mainly
    as a diagnostic. The storage and arithmetic cost of the order-``k`` tensor
    grows as ``dimension**k``; sixth-order moments can therefore become large
    even for moderate dimensions.

    PyTorch results remain connected to autograd when tensor inputs require
    gradients. Weight normalization is part of the differentiable calculation.

    Examples
    --------
    Recover the covariance used to generate a spherical design:

    >>> import numpy as np
    >>> from spheredd import evaluate_moments, spherical_design
    >>> covariance = np.diag([1.0, 2.0, 3.0, 4.0])
    >>> points, weights, info = spherical_design(
    ...     covariance, n_points=1, design_order=4, radius=1.0
    ... )
    >>> result = evaluate_moments(
    ...     points,
    ...     weights,
    ...     max_order=4,
    ...     radius=info["radius"],
    ...     source_rank=info["rank"],
    ... )
    >>> np.allclose(result["covariance"], covariance, atol=1e-12)
    True
    >>> result["moments"][4].shape
    (4, 4, 4, 4)

    Evaluate unscaled moments in the coordinates supplied by the caller:

    >>> raw = evaluate_moments(points, weights, max_order=2, radius=None)
    >>> np.allclose(raw["covariance"], covariance / 4.0, atol=1e-12)
    True
    """
    selected = _resolve_backend(points, backend)
    if selected == "numpy":
        if device is not None:
            raise ValueError("device is only valid for the PyTorch backend")
        return evaluate_moments_numpy(
            points,
            weights,
            max_order=max_order,
            radius=radius,
            source_rank=source_rank,
            dtype=dtype,
        )

    torch_backend = _load_torch_backend()
    return torch_backend.evaluate_moments_torch(
        points,
        weights,
        max_order=max_order,
        radius=radius,
        source_rank=source_rank,
        dtype=dtype,
        device=device,
    )


def evaluate_moments2(points: Any, weights: Any | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate weighted central moments through order 2."""
    return evaluate_moments(points, weights, max_order=2, **kwargs)


def evaluate_moments4(points: Any, weights: Any | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate weighted central moments through order 4."""
    return evaluate_moments(points, weights, max_order=4, **kwargs)


def evaluate_moments6(points: Any, weights: Any | None = None, **kwargs: Any) -> dict[str, Any]:
    """Evaluate weighted central moments through order 6."""
    return evaluate_moments(points, weights, max_order=6, **kwargs)
