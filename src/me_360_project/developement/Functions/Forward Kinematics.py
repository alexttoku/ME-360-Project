"""Forward kinematics for the planar Stewart platform.

For known strut lengths ``p1``, ``p2``, and ``p3``, Sauer's derivation
reduces the three unknown pose variables ``(x, y, theta)`` to one scalar
equation in ``theta``.  This module evaluates that equation, locates its
roots, and recovers ``x`` and ``y`` for every valid root.

The ``geometry`` argument is expected to be a ``PlatformGeometry`` object
from ``Geometry & Inverse Kinematics.py``.  The module intentionally uses
the object's public attributes instead of importing that file directly,
because its filename contains spaces and an ampersand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ForwardTerms:
    """Intermediate quantities used by the scalar forward equation."""

    N1: float
    N2: float
    D: float


@dataclass(frozen=True)
class ForwardPose:
    """One platform pose recovered from a set of strut lengths."""

    x: float
    y: float
    theta: float
    residual: float

    @property
    def as_tuple(self) -> tuple[float, float, float]:
        """Return the pose as ``(x, y, theta)``."""
        return self.x, self.y, self.theta


def _validate_strut_lengths(strut_lengths: np.ndarray | tuple[float, ...]) -> np.ndarray:
    lengths = np.asarray(strut_lengths, dtype=float)
    if lengths.shape != (3,):
        raise ValueError("strut_lengths must contain exactly p1, p2, and p3")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("strut lengths must be finite and positive")
    return lengths


def forward_terms(geometry, theta: float, strut_lengths) -> ForwardTerms:
    """Evaluate Sauer's ``N1``, ``N2``, and ``D`` terms at ``theta``."""
    theta = float(theta)
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    p1, p2, p3 = _validate_strut_lengths(strut_lengths)

    A2 = geometry.L3 * np.cos(theta) - geometry.x1
    B2 = geometry.L3 * np.sin(theta)
    A3 = geometry.L2 * np.cos(theta + geometry.gamma) - geometry.x2
    B3 = geometry.L2 * np.sin(theta + geometry.gamma) - geometry.y2

    q2 = p2**2 - p1**2 - A2**2 - B2**2
    q3 = p3**2 - p1**2 - A3**2 - B3**2

    N1 = B3 * q2 - B2 * q3
    N2 = -A3 * q2 + A2 * q3
    D = 2.0 * (A2 * B3 - B2 * A3)
    return ForwardTerms(float(N1), float(N2), float(D))


def forward_equation(geometry, theta: float, strut_lengths) -> float:
    """Evaluate ``f(theta) = N1**2 + N2**2 - p1**2*D**2``."""
    terms = forward_terms(geometry, theta, strut_lengths)
    p1 = _validate_strut_lengths(strut_lengths)[0]
    return terms.N1**2 + terms.N2**2 - p1**2 * terms.D**2


def recover_xy(
    geometry,
    theta: float,
    strut_lengths,
    denominator_tolerance: float = 1e-12,
) -> tuple[float, float]:
    """Recover ``x`` and ``y`` from a valid orientation root.

    A near-zero ``D`` indicates a singular elimination geometry.  In that
    case the formulas for ``x`` and ``y`` are numerically unreliable, so a
    ``ZeroDivisionError`` is raised rather than returning a misleading pose.
    """
    if denominator_tolerance <= 0.0:
        raise ValueError("denominator_tolerance must be positive")
    terms = forward_terms(geometry, theta, strut_lengths)
    if abs(terms.D) <= denominator_tolerance:
        raise ZeroDivisionError("D is too close to zero to recover x and y")
    return terms.N1 / terms.D, terms.N2 / terms.D


def bisect_root(
    function: Callable[[float], float],
    left: float,
    right: float,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> tuple[float, int, int]:
    """Find a bracketed root with bisection.

    Returns ``(root, iterations, function_evaluations)``.  The endpoints
    must have opposite signs, unless one endpoint is already a root.
    """
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance must be positive and max_iterations must be > 0")
    a, b = float(left), float(right)
    fa, fb = float(function(a)), float(function(b))
    evaluations = 2
    if fa == 0.0:
        return a, 0, evaluations
    if fb == 0.0:
        return b, 0, evaluations
    if np.signbit(fa) == np.signbit(fb):
        raise ValueError("bisection endpoints must bracket a sign-changing root")

    for iteration in range(1, max_iterations + 1):
        midpoint = 0.5 * (a + b)
        fm = float(function(midpoint))
        evaluations += 1
        if abs(fm) <= tolerance or 0.5 * abs(b - a) <= tolerance:
            return midpoint, iteration, evaluations
        if np.signbit(fa) != np.signbit(fm):
            b, fb = midpoint, fm
        else:
            a, fa = midpoint, fm
    raise RuntimeError("bisection did not converge within max_iterations")


def root_brackets(
    geometry,
    strut_lengths,
    grid_size: int = 10001,
    theta_min: float = -np.pi,
    theta_max: float = np.pi,
    zero_tolerance: float = 1e-12,
) -> tuple[list[tuple[float, float]], list[float]]:
    """Locate sign-change brackets and grid points close to roots.

    The second returned list contains exact/near-zero grid points.  Returning
    them separately prevents an exact grid root from being lost when it has
    no sign change.  A scan alone can still miss a double root between grid
    points; callers should refine the grid or use an extremum-based check for
    high-confidence root counts.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")
    if zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be positive")

    theta_grid = np.linspace(theta_min, theta_max, grid_size)
    values = np.array(
        [forward_equation(geometry, theta, strut_lengths) for theta in theta_grid]
    )
    brackets: list[tuple[float, float]] = []
    grid_roots: list[float] = []
    scale = max(1.0, float(np.max(np.abs(values))))
    for index, value in enumerate(values):
        if abs(value) <= zero_tolerance * scale:
            grid_roots.append(float(theta_grid[index]))
        if index and np.isfinite(values[index - 1]) and np.isfinite(value):
            if np.signbit(values[index - 1]) != np.signbit(value):
                brackets.append((float(theta_grid[index - 1]), float(theta_grid[index])))
    return brackets, grid_roots


def _unique_angles(angles: list[float], tolerance: float) -> list[float]:
    """Sort and deduplicate angles, treating the interval endpoints alike."""
    normalized = [((angle + np.pi) % (2.0 * np.pi)) - np.pi for angle in angles]
    normalized.sort()
    unique: list[float] = []
    for angle in normalized:
        if not unique or abs(angle - unique[-1]) > tolerance:
            unique.append(angle)
    if len(unique) > 1 and abs((unique[0] + 2.0 * np.pi) - unique[-1]) <= tolerance:
        unique.pop()
    return unique


def solve_forward_kinematics(
    geometry,
    strut_lengths,
    grid_size: int = 10001,
    tolerance: float = 1e-12,
    method: str = "brent",
    denominator_tolerance: float = 1e-12,
) -> list[ForwardPose]:
    """Find all sign-change-based forward-kinematics poses.

    ``method`` may be ``"brent"`` or ``"bisection"``.  Results are sorted
    by orientation.  Double roots that do not change sign can be missed by
    the grid scan and should be investigated separately near pose-count
    transitions.
    """
    if method not in {"brent", "bisection"}:
        raise ValueError("method must be 'brent' or 'bisection'")
    _validate_strut_lengths(strut_lengths)
    brackets, grid_roots = root_brackets(geometry, strut_lengths, grid_size)
    roots = list(grid_roots)
    function = lambda theta: forward_equation(geometry, theta, strut_lengths)

    for left, right in brackets:
        if method == "brent":
            roots.append(float(brentq(function, left, right, xtol=tolerance)))
        else:
            roots.append(bisect_root(function, left, right, tolerance=tolerance)[0])

    poses: list[ForwardPose] = []
    for theta in _unique_angles(roots, max(tolerance, 1e-10)):
        try:
            x, y = recover_xy(
                geometry, theta, strut_lengths, denominator_tolerance
            )
        except ZeroDivisionError:
            continue
        poses.append(
            ForwardPose(
                x=float(x),
                y=float(y),
                theta=float(theta),
                residual=abs(function(theta)),
            )
        )
    return poses


__all__ = [
    "ForwardPose",
    "ForwardTerms",
    "bisect_root",
    "forward_equation",
    "forward_terms",
    "recover_xy",
    "root_brackets",
    "solve_forward_kinematics",
]
