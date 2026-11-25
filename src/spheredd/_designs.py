"""Base equal-weight spherical-design constructions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from itertools import permutations, product

import numpy as np

from ._validation import validate_spherical_design

VALID_DESIGN_ORDERS = (2, 4, 6)


@dataclass(frozen=True)
class BaseDesign:
    """An immutable base design and its metadata."""

    points: np.ndarray
    requested_order: int
    strength: int
    construction: str
    exact: bool
    cardinality_status: str
    validation_max_error: float
    references: tuple[str, ...]

    @property
    def rank(self) -> int:
        return int(self.points.shape[1])

    @property
    def size(self) -> int:
        return int(self.points.shape[0])


def _readonly(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    points.setflags(write=False)
    return points


def _regular_polygon(sides: int) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(sides, dtype=np.float64) / sides
    return np.column_stack((np.cos(angles), np.sin(angles)))


def _regular_simplex(rank: int) -> np.ndarray:
    """Return the rank + 1 unit vertices of a regular simplex in R^rank."""
    if not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError("rank must be a positive integer")

    rank = int(rank)
    helmert = np.zeros((rank + 1, rank), dtype=np.float64)
    for column in range(rank):
        denominator = np.sqrt((column + 1) * (column + 2))
        helmert[: column + 1, column] = 1.0 / denominator
        helmert[column + 1, column] = -(column + 1) / denominator

    return np.sqrt((rank + 1) / rank) * helmert


def _polygon_design_from_planes(planes: np.ndarray, sides: int) -> np.ndarray:
    """Place an equally phased regular polygon in each orthonormal plane."""
    planes = np.asarray(planes, dtype=np.float64)
    if planes.ndim != 3 or planes.shape[1] != 2:
        raise ValueError("planes must have shape (n_planes, 2, rank)")
    if not isinstance(sides, (int, np.integer)) or sides < 2:
        raise ValueError("sides must be an integer greater than or equal to 2")

    n_planes = planes.shape[0]
    plane_grams = np.einsum("mai,mbi->mab", planes, planes, optimize=True)
    if not np.allclose(
        plane_grams,
        np.broadcast_to(np.eye(2), (n_planes, 2, 2)),
        rtol=0.0,
        atol=2e-14,
    ):
        raise RuntimeError("plane generators must be orthonormal")

    angles = 2.0 * np.pi * np.arange(int(sides)) / int(sides)
    points = (
        np.cos(angles)[None, :, None] * planes[:, 0, None, :]
        + np.sin(angles)[None, :, None] * planes[:, 1, None, :]
    ).reshape(n_planes * int(sides), planes.shape[2])
    return points


def _rank4_order4() -> np.ndarray:
    s = 1.0 / np.sqrt(3.0)
    planes = np.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, s, s, s]],
            [[0.0, 1.0, 0.0, 0.0], [s, 0.0, s, -s]],
            [[0.0, 0.0, 1.0, 0.0], [s, -s, 0.0, s]],
            [[0.0, 0.0, 0.0, 1.0], [s, s, -s, 0.0]],
        ],
        dtype=np.float64,
    )
    return _polygon_design_from_planes(planes, sides=5)


def _rank6_order4() -> np.ndarray:
    points = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])]
    large = np.sqrt(3.0) / 2.0
    for axis in range(5):
        for sign in (-1.0, 1.0):
            point = np.zeros(6)
            point[axis] = sign * large
            point[5] = -0.5
            points.append(point)

    small = np.sqrt(3.0) / 4.0
    for signs in product((-1.0, 1.0), repeat=5):
        if np.prod(signs) > 0.0:
            point = np.empty(6)
            point[:5] = small * np.asarray(signs)
            point[5] = 0.25
            points.append(point)

    return np.asarray(points, dtype=np.float64)


def _rank4_order6() -> np.ndarray:
    s = 1.0 / np.sqrt(2.0)
    h = 0.5
    planes = np.array(
        [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            [[s, 0.0, -h, h], [0.0, s, -h, -h]],
            [[-h, h, s, 0.0], [-h, -h, 0.0, s]],
            [[-s, 0.0, -h, -h], [0.0, -s, h, -h]],
            [[-h, -h, -s, 0.0], [h, -h, 0.0, -s]],
        ],
        dtype=np.float64,
    )
    return _polygon_design_from_planes(planes, sides=7)


def _permutation_parity(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return inversions % 2


def _rank3_order6() -> np.ndarray:
    """Return McLaren's 24-point improved snub-cube spherical 7-design."""
    # A^2, B^2, C^2 are the three roots of
    # 105 y^3 - 105 y^2 + 21 y - 1 = 0.
    squared_roots = np.roots([105.0, -105.0, 21.0, -1.0])
    if np.max(np.abs(squared_roots.imag)) > 1e-13:
        raise RuntimeError("failed to construct the improved snub cube")
    abc = np.sqrt(np.sort(squared_roots.real)[::-1])

    points: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        parity = _permutation_parity(permutation)
        permuted = abc[list(permutation)]
        for signs in product((-1.0, 1.0), repeat=3):
            negative_parity = sum(sign < 0.0 for sign in signs) % 2
            if negative_parity == parity:
                points.append(permuted * np.asarray(signs))

    result = np.asarray(points, dtype=np.float64)
    if result.shape != (24, 3):
        raise RuntimeError("internal error while constructing improved snub cube")
    return result


@lru_cache(maxsize=1)
def _numerical_designs() -> dict[str, np.ndarray]:
    resource = resources.files("spheredd.data").joinpath("numerical_designs.npz")
    with resource.open("rb") as stream, np.load(stream) as archive:
        return {name: np.asarray(archive[name], dtype=np.float64) for name in archive.files}


def _general_order4(rank: int) -> np.ndarray:
    """Explicit order-4 fallback; not intended to minimize cardinality."""
    signs = np.asarray(list(product((-1.0, 1.0), repeat=rank)), dtype=np.float64)
    q = np.sqrt(2.0 / (rank + 2.0))
    large = np.sqrt((1.0 + (rank - 1) * q) / rank)
    small = np.sqrt((1.0 - q) / rank)
    blocks = []
    for axis in range(rank):
        block = small * signs.copy()
        block[:, axis] = large * signs[:, axis]
        blocks.append(block)
    return np.vstack(blocks)


def _icosahedron() -> np.ndarray:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    points = []
    for sign_1 in (-1.0, 1.0):
        for sign_2 in (-phi, phi):
            points.extend(
                [
                    (0.0, sign_1, sign_2),
                    (sign_1, sign_2, 0.0),
                    (sign_2, 0.0, sign_1),
                ]
            )
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    return points / np.sqrt(1.0 + phi**2)


def _construction(rank: int, order: int) -> tuple[np.ndarray, int, str, bool, str, tuple[str, ...], float]:
    if order == 2:
        return (
            _regular_simplex(rank),
            2,
            "regular simplex",
            True,
            "proven minimum",
            ("Delsarte-Goethals-Seidel (1977)",),
            5e-13,
        )

    if order == 4:
        if rank == 1:
            return (
                np.array([[-1.0], [1.0]]),
                4,
                "two-point design on S^0",
                True,
                "proven minimum",
                ("Delsarte-Goethals-Seidel (1977)",),
                5e-13,
            )
        if rank == 2:
            return (
                _regular_polygon(5),
                4,
                "regular pentagon",
                True,
                "proven minimum",
                ("Delsarte-Goethals-Seidel (1977)",),
                5e-13,
            )
        if rank == 3:
            return (
                _icosahedron(),
                5,
                "regular icosahedron",
                True,
                "smallest construction used by this package",
                ("Hardin-Sloane (1996)",),
                5e-13,
            )
        if rank == 4:
            return (
                _rank4_order4(),
                4,
                "four regular pentagons in four planes",
                True,
                "smallest published construction used by this package",
                ("Cara-Hardin-Sloane (2003)",),
                5e-13,
            )
        if rank == 5:
            return (
                _numerical_designs()["rank5_order4"],
                4,
                "29-point numerical equal-weight spherical 4-design",
                False,
                "best-known numerical; conjectured minimum",
                ("Hardin-Sloane (1992)",),
                5e-13,
            )
        if rank == 6:
            return (
                _rank6_order4(),
                4,
                "27-point Schlaefli tight spherical 4-design",
                True,
                "proven minimum",
                ("Hardin-Sloane (1992)", "Delsarte-Goethals-Seidel (1977)"),
                5e-13,
            )
        return (
            _general_order4(rank),
            4,
            "general explicit spherical 4-design",
            True,
            "explicit fallback; not cardinality-minimal",
            ("Hardin-Sloane (1992)",),
            5e-13,
        )

    if order == 6:
        if rank == 1:
            return (
                np.array([[-1.0], [1.0]]),
                6,
                "two-point design on S^0",
                True,
                "proven minimum",
                ("Delsarte-Goethals-Seidel (1977)",),
                5e-13,
            )
        if rank == 2:
            return (
                _regular_polygon(7),
                6,
                "regular heptagon",
                True,
                "proven minimum",
                ("Delsarte-Goethals-Seidel (1977)",),
                5e-13,
            )
        if rank == 3:
            return (
                _rank3_order6(),
                7,
                "24-point McLaren improved snub cube",
                True,
                "smallest published construction used by this package",
                ("Hardin-Sloane (1996)",),
                5e-13,
            )
        if rank == 4:
            return (
                _rank4_order6(),
                6,
                "six regular heptagons in six planes",
                True,
                "smallest published construction used by this package",
                ("Cara-Hardin-Sloane (2003)",),
                5e-13,
            )
        if rank == 5:
            return (
                _numerical_designs()["rank5_order6"],
                6,
                "87-point numerical equal-weight spherical 6-design",
                False,
                "practical numerical construction; minimum unknown",
                ("Womersley (2018)",),
                5e-13,
            )
        if rank == 6:
            return (
                _numerical_designs()["rank6_order6"],
                6,
                "146-point numerical equal-weight spherical 6-design",
                False,
                "practical numerical construction; minimum unknown",
                ("Womersley (2018)",),
                5e-13,
            )
        raise ValueError(
            "design_order=6 is currently supported only when covariance has "
            "numerical rank from 1 through 6"
        )

    raise ValueError("design_order must be 2, 4, or 6")


@lru_cache(maxsize=None)
def get_base_design(rank: int, design_order: int) -> BaseDesign:
    """Return a validated base design for the requested rank and order."""
    if not isinstance(rank, (int, np.integer)) or int(rank) < 1:
        raise ValueError("rank must be a positive integer")
    if not isinstance(design_order, (int, np.integer)) or int(design_order) not in VALID_DESIGN_ORDERS:
        raise ValueError("design_order must be 2, 4, or 6")

    rank = int(rank)
    design_order = int(design_order)
    points, strength, name, exact, status, refs, atol = _construction(rank, design_order)
    points = _readonly(points)

    if points.ndim != 2 or points.shape[1] != rank:
        raise RuntimeError(f"internal design rank mismatch for rank={rank}, order={design_order}")

    error = validate_spherical_design(
        points,
        max_order=strength,
        name=f"rank-{rank} {name}",
        atol=atol,
    )

    return BaseDesign(
        points=points,
        requested_order=design_order,
        strength=strength,
        construction=name,
        exact=exact,
        cardinality_status=status,
        validation_max_error=error,
        references=refs,
    )
