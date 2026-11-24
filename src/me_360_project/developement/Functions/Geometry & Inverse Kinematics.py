"""Geometry and inverse kinematics for a planar Stewart platform.

The platform convention follows Sauer's planar Stewart-platform example:

* the first moving-platform vertex is ``(x, y)``;
* the second vertex is ``(x, y) + L3*[cos(theta), sin(theta)]``;
* the third vertex is ``(x, y) + L2*[cos(theta + gamma),
  sin(theta + gamma)]``.

The fixed strut anchors are, by convention, ``(0, 0)``, ``(x1, 0)``, and
``(x2, y2)``.  The three returned strut lengths are therefore the lengths
from anchor 1 to platform vertex 1, anchor 2 to vertex 2, and anchor 3 to
vertex 3, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ArrayLike = Iterable[float] | np.ndarray


def _point(point: ArrayLike, name: str) -> np.ndarray:
    """Convert a point to a finite two-dimensional float array."""
    value = np.asarray(tuple(point), dtype=float)
    if value.shape != (2,):
        raise ValueError(f"{name} must contain exactly two coordinates")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite coordinates")
    return value


def _positive_length(value: float, name: str) -> float:
    """Validate and return a strictly positive length."""
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


@dataclass(frozen=True)
class PlatformGeometry:
    """Fixed geometry of a planar Stewart platform.

    Parameters
    ----------
    x1, x2, y2:
        Coordinates of the second and third fixed anchors.  The first anchor
        is always the origin.
    L1, L2, L3:
        Side lengths of the moving triangular platform.  ``L1`` is the side
        between the second and third platform vertices; ``L2`` and ``L3``
        are the sides connected to the first vertex.
    gamma:
        Included angle between the platform sides of lengths ``L2`` and
        ``L3``, in radians.
    """

    x1: float
    x2: float
    y2: float
    L1: float
    L2: float
    L3: float
    gamma: float

    def __post_init__(self) -> None:
        for name in ("x1", "x2", "y2"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

        for name in ("L1", "L2", "L3"):
            object.__setattr__(self, name, _positive_length(getattr(self, name), name))

        gamma = float(self.gamma)
        if not np.isfinite(gamma):
            raise ValueError("gamma must be finite")
        object.__setattr__(self, "gamma", gamma)

        # The side lengths must describe a real triangle.
        if (
            self.L1 + self.L2 <= self.L3
            or self.L1 + self.L3 <= self.L2
            or self.L2 + self.L3 <= self.L1
        ):
            raise ValueError("L1, L2, and L3 must satisfy the triangle inequality")

    @property
    def anchors(self) -> np.ndarray:
        """Return fixed anchors as an array with shape ``(3, 2)``."""
        return np.array(((0.0, 0.0), (self.x1, 0.0), (self.x2, self.y2)))

    def vertices(self, x: float, y: float, theta: float) -> np.ndarray:
        """Return the three moving-platform vertices for a given pose.

        The returned array has shape ``(3, 2)`` and is ordered to correspond
        to the three anchors returned by :attr:`anchors`.
        """
        x, y, theta = float(x), float(y), float(theta)
        if not np.all(np.isfinite((x, y, theta))):
            raise ValueError("x, y, and theta must be finite")

        vertex_1 = np.array((x, y))
        vertex_2 = vertex_1 + self.L3 * np.array((np.cos(theta), np.sin(theta)))
        vertex_3 = vertex_1 + self.L2 * np.array(
            (np.cos(theta + self.gamma), np.sin(theta + self.gamma))
        )

        # This check catches inconsistent L1/gamma/L2/L3 input while allowing
        # a small amount of floating-point roundoff.
        side_1 = np.linalg.norm(vertex_3 - vertex_2)
        if not np.isclose(side_1, self.L1, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "L1 is inconsistent with L2, L3, and gamma; "
                "the moving-platform triangle cannot be constructed"
            )
        return np.vstack((vertex_1, vertex_2, vertex_3))


def platform_vertices(
    geometry: PlatformGeometry, x: float, y: float, theta: float
) -> np.ndarray:
    """Functional wrapper returning the platform vertices for a pose."""
    return geometry.vertices(x, y, theta)


def inverse_kinematics(
    geometry: PlatformGeometry, x: float, y: float, theta: float
) -> np.ndarray:
    """Compute the three strut lengths for a specified platform pose.

    Parameters are the platform geometry and pose ``(x, y, theta)``.  The
    result is ``[p1, p2, p3]``.  This is the inverse-kinematics operation:
    pose in, actuator lengths out.
    """
    vertices = geometry.vertices(x, y, theta)
    return np.linalg.norm(vertices - geometry.anchors, axis=1)


def inverse_kinematics_from_pose(
    geometry: PlatformGeometry, pose: ArrayLike
) -> np.ndarray:
    """Compute strut lengths from an iterable ``(x, y, theta)`` pose."""
    pose_array = np.asarray(tuple(pose), dtype=float)
    if pose_array.shape != (3,):
        raise ValueError("pose must contain exactly x, y, and theta")
    return inverse_kinematics(geometry, *pose_array)


def platform_side_lengths(
    geometry: PlatformGeometry, x: float, y: float, theta: float
) -> np.ndarray:
    """Return the realized moving-triangle side lengths as a consistency check."""
    vertices = geometry.vertices(x, y, theta)
    return np.array(
        (
            np.linalg.norm(vertices[1] - vertices[2]),
            np.linalg.norm(vertices[0] - vertices[2]),
            np.linalg.norm(vertices[0] - vertices[1]),
        )
    )


def verify_strut_lengths(
    geometry: PlatformGeometry,
    pose: ArrayLike,
    expected_lengths: ArrayLike,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> bool:
    """Return whether a pose reproduces the supplied strut lengths."""
    actual = inverse_kinematics_from_pose(geometry, pose)
    expected = np.asarray(tuple(expected_lengths), dtype=float)
    if expected.shape != (3,):
        raise ValueError("expected_lengths must contain exactly three lengths")
    return bool(np.allclose(actual, expected, atol=atol, rtol=rtol))


__all__ = [
    "PlatformGeometry",
    "inverse_kinematics",
    "inverse_kinematics_from_pose",
    "platform_side_lengths",
    "platform_vertices",
    "verify_strut_lengths",
]
