"""Verification utilities and plots for the planar Stewart platform.

This module is deliberately independent of the files with spaces in their
names.  It accepts the public interface of ``PlatformGeometry`` and the
``ForwardPose`` objects returned by the forward-kinematics solver.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _pose_array(pose) -> np.ndarray:
    """Convert a pose object or ``(x, y, theta)`` iterable to an array."""
    if hasattr(pose, "x") and hasattr(pose, "y") and hasattr(pose, "theta"):
        values = (pose.x, pose.y, pose.theta)
    else:
        values = pose
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("pose must contain finite x, y, and theta values")
    return array


def _length_array(lengths) -> np.ndarray:
    array = np.asarray(tuple(lengths), dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("strut lengths must contain three finite values")
    if np.any(array < 0.0):
        raise ValueError("strut lengths cannot be negative")
    return array


def computed_strut_lengths(geometry, pose) -> np.ndarray:
    """Compute strut lengths directly from a pose.

    The result is ordered as ``(p1, p2, p3)`` and corresponds to the three
    fixed anchors and the three matching moving-platform vertices.
    """
    x, y, theta = _pose_array(pose)
    vertices = geometry.vertices(x, y, theta)
    return np.linalg.norm(vertices - geometry.anchors, axis=1)


def strut_length_errors(geometry, pose, expected_lengths) -> np.ndarray:
    """Return computed minus expected strut lengths for one pose."""
    return computed_strut_lengths(geometry, pose) - _length_array(expected_lengths)


def verify_pose(
    geometry,
    pose,
    expected_lengths,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> dict[str, object]:
    """Verify one pose and return an analysis-friendly result dictionary.

    The dictionary contains the computed lengths, signed errors, maximum
    absolute error, and a boolean ``passed`` value.
    """
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("atol and rtol must be nonnegative")
    expected = _length_array(expected_lengths)
    computed = computed_strut_lengths(geometry, pose)
    errors = computed - expected
    return {
        "pose": _pose_array(pose),
        "expected_lengths": expected,
        "computed_lengths": computed,
        "errors": errors,
        "max_absolute_error": float(np.max(np.abs(errors))),
        "passed": bool(np.allclose(computed, expected, atol=atol, rtol=rtol)),
    }


def verify_poses(
    geometry,
    poses: Iterable,
    expected_lengths,
    atol: float = 1e-10,
    rtol: float = 1e-10,
) -> list[dict[str, object]]:
    """Verify every pose in an iterable against the same strut lengths."""
    return [
        verify_pose(geometry, pose, expected_lengths, atol=atol, rtol=rtol)
        for pose in poses
    ]


def plot_platform_pose(
    geometry,
    pose,
    ax=None,
    expected_lengths=None,
    title: str | None = None,
    show_labels: bool = True,
):
    """Plot anchors, struts, and one platform pose.

    Returns ``(figure, axes)`` so callers can add annotations or save the
    result with ``figure.savefig(...)``.  If ``expected_lengths`` is supplied,
    the plot title also includes the maximum inverse-kinematics error.
    """
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure

    pose_values = _pose_array(pose)
    vertices = geometry.vertices(*pose_values)
    anchors = np.asarray(geometry.anchors, dtype=float)
    if anchors.shape != (3, 2):
        raise ValueError("geometry.anchors must have shape (3, 2)")

    ax.plot(
        np.r_[vertices[:, 0], vertices[0, 0]],
        np.r_[vertices[:, 1], vertices[0, 1]],
        "o-",
        color="tab:blue",
        label="platform",
    )
    ax.scatter(anchors[:, 0], anchors[:, 1], color="tab:red", marker="s", label="anchors")
    for index, (anchor, vertex) in enumerate(zip(anchors, vertices), start=1):
        ax.plot(
            (anchor[0], vertex[0]),
            (anchor[1], vertex[1]),
            color="tab:gray",
            linestyle="--",
            label="struts" if index == 1 else None,
        )
        if show_labels:
            ax.annotate(f"p{index}", (vertex + anchor) / 2.0)

    if show_labels:
        for index, point in enumerate(anchors, start=1):
            ax.annotate(f"A{index}", point)
        for index, point in enumerate(vertices, start=1):
            ax.annotate(f"V{index}", point)

    if title is None:
        title = f"Planar Stewart platform (theta = {pose_values[2]:.5g} rad)"
    if expected_lengths is not None:
        result = verify_pose(geometry, pose, expected_lengths)
        title += f"; max strut error = {result['max_absolute_error']:.2e}"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return figure, ax


def plot_platform_poses(
    geometry,
    poses: Iterable,
    expected_lengths=None,
    figsize: tuple[float, float] | None = None,
):
    """Plot multiple poses in side-by-side subplots.

    Returns ``(figure, axes)``.  The axes value is always a one-dimensional
    NumPy array, including when there is only one pose.
    """
    pose_list = list(poses)
    if not pose_list:
        raise ValueError("poses must contain at least one pose")
    if figsize is None:
        figsize = (5.0 * len(pose_list), 4.5)
    figure, axes = plt.subplots(1, len(pose_list), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax, pose in zip(axes, pose_list):
        plot_platform_pose(
            geometry,
            pose,
            ax=ax,
            expected_lengths=expected_lengths,
        )
    figure.tight_layout()
    return figure, axes


def plot_forward_equation(
    equation,
    roots: Iterable[float] = (),
    theta_min: float = -np.pi,
    theta_max: float = np.pi,
    samples: int = 5000,
    ax=None,
):
    """Plot ``f(theta)`` and mark known roots.

    ``equation`` should be a callable accepting one scalar angle.  Returns
    ``(figure, axes)`` and does not call ``plt.show()``.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")
    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    theta = np.linspace(theta_min, theta_max, samples)
    values = np.asarray([equation(value) for value in theta], dtype=float)
    ax.plot(theta, values, label="f(theta)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    root_values = np.asarray(tuple(roots), dtype=float)
    if root_values.size:
        root_y = np.asarray([equation(value) for value in root_values])
        ax.scatter(root_values, root_y, color="tab:red", zorder=3, label="roots")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("f(theta)")
    ax.set_title("Forward-kinematics equation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return figure, ax


__all__ = [
    "computed_strut_lengths",
    "plot_forward_equation",
    "plot_platform_pose",
    "plot_platform_poses",
    "strut_length_errors",
    "verify_pose",
    "verify_poses",
]
