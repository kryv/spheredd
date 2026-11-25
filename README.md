# spheredd

`spheredd` is a Python package for mapping equal-weight spherical 2-, 4-, and 6-designs on `S^{r-1} \subset R^r` into the active subspace of a positive semidefinite covariance matrix.

- A shared API for NumPy and PyTorch
- PyTorch is optional; all NumPy functionality remains available when `torch` is not installed or cannot be imported
- The base design is selected from the **numerical rank** determined from the covariance eigenvalues, rather than from the matrix dimension
- When the requested number of points exceeds the base size, complete randomly rotated copies of the design are added
- Numerical rank-5 and rank-6 designs are validated in `float64` against their moment conditions when first loaded

## Installation

NumPy only:

```bash
python -m pip install .
```

Install PyTorch as an optional dependency:

```bash
python -m pip install ".[torch]"
```

When a specific PyTorch build is required, such as a CUDA-enabled build, install PyTorch first using the official PyTorch instructions and then run `python -m pip install .`.

## Minimum Base Particle Counts Used by the Package

The following table gives the smallest single-block particle count selected by this package when `n_points=1`. Here, rank means the numerical rank of the input covariance matrix.

| Design order | Rank 2 | Rank 3 | Rank 4 | Rank 5 | Rank 6 |
|---:|---:|---:|---:|---:|---:|
| 2-design | 3 [\*1] | 4 [\*1] | 5 [\*1] | 6 [\*1] | 7 [\*1] |
| 4-design | 5 [\*1] | 12 [\*2] | 20 [\*3] | 29† [\*4] | 27 [\*4] |
| 6-design | 7 [\*1] | 24‡ [\*2] | 42 [\*3] | 87† [\*5] | 146† [\*5] |

† These are double-precision, equal-weight numerical solutions. The 29-point rank-5 4-design is a candidate minimum, but this package does not treat its minimality as proven. Womersley's published data likewise do not claim that the 87-point and 146-point 6-designs are globally minimal.

‡ The 24-point rank-3 construction is McLaren's improved snub cube and is actually a 7-design.

The table reports the smallest base size implemented by this package. Some entries have unproven or conjectural mathematical minimality; the corresponding status is also returned in `info["base_cardinality_status"]`.

\*1 P. Delsarte, J.-M. Goethals, and J. J. Seidel, "Spherical codes and designs," *Geometriae Dedicata* 6 (1977), 363-388. https://doi.org/10.1007/BF03187604

\*2 R. H. Hardin and N. J. A. Sloane, "McLaren's Improved Snub Cube and Other New Spherical Designs in Three Dimensions," *Discrete & Computational Geometry* 15 (1996), 429-441. https://doi.org/10.1007/BF02711518 (arXiv: https://arxiv.org/abs/math/0207211)

\*3 P. Cara, R. H. Hardin, and N. J. A. Sloane, "Spherical Designs in Four Dimensions," *Proceedings of the 2003 IEEE Information Theory Workshop*, 253-257. https://doi.org/10.1109/ITW.2003.1216742

\*4 R. H. Hardin and N. J. A. Sloane, "New spherical 4-designs," *Discrete Mathematics* 106/107 (1992), 255-264. https://doi.org/10.1016/0012-365X(92)90552-Q

\*5 R. S. Womersley, "Efficient Spherical Designs with Good Geometric Properties," in *Contemporary Computational Mathematics* (2018), 1243-1285. https://doi.org/10.1007/978-3-319-72456-0_57

## NumPy

```python
import numpy as np
from spheredd import spherical_design, evaluate_moments6

covariance = np.eye(5)

points, weights, info = spherical_design(
    covariance,
    n_points=1,
    design_order=6,
)

print(points.shape)                 # (87, 5)
print(info["rank"])                 # 5
print(info["base_size"])            # 87
print(info["backend"])              # "numpy"
print(info["base_design_exact"])    # False

moments = evaluate_moments6(
    points,
    weights,
    radius=info["radius"],
    source_rank=info["rank"],
)
print(moments["covariance"])
```

## PyTorch

Passing a PyTorch tensor automatically selects the PyTorch backend. The output dtype and device follow the input tensor.

```python
import torch
from spheredd import spherical_design, evaluate_moments6

covariance = torch.eye(6, dtype=torch.float64)

points, weights, info = spherical_design(
    covariance,
    n_points=1,
    design_order=6,
)

print(points.shape)       # torch.Size([146, 6])
print(points.dtype)       # torch.float64
print(info["backend"])    # "torch"

moments = evaluate_moments6(
    points,
    weights,
    radius=info["radius"],
    source_rank=info["rank"],
)
```

The backend can also be selected explicitly:

```python
points, weights, info = spherical_design(
    covariance,
    1,
    design_order=4,
    backend="torch",
    device="cpu",
    dtype=torch.float64,
)
```

Check whether PyTorch is available with:

```python
import spheredd
print(spheredd.torch_available())
```

`spheredd` does not import PyTorch at package import time. When only NumPy inputs are used, the package and its NumPy backend continue to work even if PyTorch is not installed or cannot be imported.

## Point-Count Rounding

`n_points` is the requested **minimum** number of points. To preserve the design conditions, the package adds complete base-design blocks.

```python
# The base size of the rank-5 6-design is 87.
points, weights, info = spherical_design(
    np.eye(5),
    n_points=88,
    design_order=6,
    seed=123,
)

print(info["actual_n_points"])  # 174
print(info["copies"])           # 2
```

Every block after the first receives a random orthogonal rotation. Among several candidates, the package selects the block with the largest minimum distance from the existing points. A union of complete, equal-weight designs of the same strength remains a design of that strength.

## `radius` and Covariance

A rank-`r` design on the unit sphere has second moment `I / r`. Therefore, the weighted covariance of the returned point set is

```text
(radius**2 / r) * covariance
```

The `evaluate_moments*` functions rescale by `sqrt(r) / radius` and return canonical moments corresponding to the input covariance.

To make the weighted covariance of the returned points equal to the input `covariance`, use `radius=np.sqrt(r)`:

```python
rank = 4
points, weights, info = spherical_design(
    np.eye(rank),
    1,
    design_order=4,
    radius=np.sqrt(rank),
)
```

## API

```python
spherical_design(
    covariance,
    n_points,
    mean=None,
    radius=1.0,
    design_order=4,
    rotation_candidates=32,
    seed=None,
    tol=1e-12,
    *,
    backend="auto",
    dtype=None,
    device=None,
)
```

Convenience wrappers:

```python
spherical_2_design(...)
spherical_4_design(...)
spherical_6_design(...)
```

Moment helpers:

```python
evaluate_moments(..., max_order=2..6)
evaluate_moments2(...)
evaluate_moments4(...)
evaluate_moments6(...)
```

Main `info` fields:

| Key | Description |
|---|---|
| `backend` | `"numpy"` or `"torch"` |
| `rank` | Numerical rank of the covariance matrix |
| `design_order` | Requested design order |
| `base_design_strength` | Actual strength of the selected base construction |
| `base_size` | Number of particles in the base design |
| `copies` | Number of complete design blocks used |
| `actual_n_points` | Number of particles actually returned |
| `base_design_exact` | Whether the base is an analytic or algebraic construction rather than an embedded numerical solution |
| `base_cardinality_status` | Status of the construction's cardinality and known minimality |
| `base_validation_max_error` | Maximum residual from the `float64` base-design validation |
| `directions` | Unit-sphere directions in the active rank subspace |

## Numerical Considerations

- `float64` is recommended for numerical rank-5 and rank-6 designs. `float32` is supported, but its higher-order moment residuals are larger.
- The PyTorch backend uses `torch.linalg.eigh`. Gradients with respect to eigenvectors can become unstable when covariance eigenvalues are repeated or when the numerical rank changes during optimization.
- `tol` is used for both the positive-semidefinite check and numerical-rank detection. For `float32`, the package internally applies a lower bound based on machine precision.

## Development

```bash
python -m pip install -e ".[test]"
pytest
python -m build
```
