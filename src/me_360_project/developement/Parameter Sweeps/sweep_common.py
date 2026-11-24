"""Shared helpers for the four parameter sweeps.

Every sweep script imports this file.  It loads Alex's solver modules from
``../Functions`` (their filenames contain spaces, so a normal ``import`` does
not work), defines the baseline geometry from Sauer's four-pose example, and
adds a few tools the sweeps need:

* ``count_poses``     - number of poses for one geometry + strut lengths,
                        including a check for double roots that the
                        sign-change scan in the solver cannot see.
* ``refine_transition`` - bisection on a sweep parameter to pin down exactly
                        where the pose count changes.
* ``save_pose_grid``  - one figure with every pose drawn side by side.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

SWEEP_DIR = Path(__file__).resolve().parent
FUNCTION_DIR = SWEEP_DIR.parent / "Functions"
OUTPUT_ROOT = SWEEP_DIR / "output"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FUNCTION_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry_module = _load("stewart_geometry", "Geometry & Inverse Kinematics.py")
forward_module = _load("stewart_forward", "Forward Kinematics.py")
plotting_module = _load("stewart_verification", "Verification & Plotting.py")

PlatformGeometry = geometry_module.PlatformGeometry
inverse_kinematics = geometry_module.inverse_kinematics
forward_equation = forward_module.forward_equation
solve_forward_kinematics = forward_module.solve_forward_kinematics
plot_platform_pose = plotting_module.plot_platform_pose
verify_pose = plotting_module.verify_pose

# ----------------------------------------------------------------------
# Baseline: Sauer's four-pose example (Reality Check 1, step 4).
# Each sweep changes one thing away from this baseline.
# ----------------------------------------------------------------------
BASE = dict(x1=5.0, x2=0.0, y2=6.0, L1=3.0, L2=3.0 * np.sqrt(2.0), L3=3.0,
            gamma=np.pi / 4)
BASE_STRUTS = np.array([5.0, 5.0, 3.0])

GRID_SIZE = 2001  # theta samples on [-pi, pi]; checked against 10001 below


def make_geometry(**changes) -> PlatformGeometry:
    """Baseline geometry with some parameters replaced."""
    params = dict(BASE)
    params.update(changes)
    return PlatformGeometry(**params)


def output_dir(name: str) -> Path:
    folder = OUTPUT_ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def f_values(geometry, struts, thetas: np.ndarray) -> np.ndarray:
    return np.array([forward_equation(geometry, t, struts) for t in thetas])


def double_roots(geometry, struts, grid_size: int = GRID_SIZE,
                 rel_tol: float = 1e-9) -> list[float]:
    """Find roots where f(theta) touches zero without changing sign.

    The solver only finds roots by sign changes.  At a double root (where two
    assembly modes merge) f just touches zero, so we look at every local
    minimum of |f| on the grid, refine it with a bounded 1-D minimizer, and
    accept it if |f| is essentially zero there.
    """
    thetas = np.linspace(-np.pi, np.pi, grid_size)
    values = f_values(geometry, struts, thetas)
    scale = max(1.0, float(np.max(np.abs(values))))
    magnitude = np.abs(values)
    found = []
    for i in range(1, grid_size - 1):
        if magnitude[i] <= magnitude[i - 1] and magnitude[i] <= magnitude[i + 1]:
            # Skip ordinary sign-changing roots (the solver already has them).
            if np.signbit(values[i - 1]) != np.signbit(values[i + 1]):
                continue
            result = minimize_scalar(
                lambda t: abs(forward_equation(geometry, t, struts)),
                bounds=(thetas[i - 1], thetas[i + 1]), method="bounded",
                options={"xatol": 1e-13},
            )
            if result.fun <= rel_tol * scale:
                found.append(float(result.x))
    return found


def tangent_minimum(geometry, struts, grid_size: int = 20001) -> float:
    """Smallest |f| at a local minimum where f does NOT cross zero.

    Returned relative to max|f| on [-pi, pi].  At a pose-count transition
    two roots merge into a double root, so just past the transition this
    number is tiny; away from transitions it is clearly positive.  The
    theta grid is treated as periodic so minima at +/- pi are not missed.
    """
    thetas = np.linspace(-np.pi, np.pi, grid_size)[:-1]  # drop duplicate +pi
    step = thetas[1] - thetas[0]
    values = f_values(geometry, struts, thetas)
    scale = max(1.0, float(np.max(np.abs(values))))
    magnitude = np.abs(values)
    prev_v, next_v = np.roll(values, 1), np.roll(values, -1)
    prev_m, next_m = np.abs(prev_v), np.abs(next_v)
    best = np.inf
    for i in np.where((magnitude <= prev_m) & (magnitude <= next_m))[0]:
        if np.signbit(prev_v[i]) != np.signbit(next_v[i]):
            continue  # ordinary crossing root, already found by the solver
        result = minimize_scalar(
            lambda t: abs(forward_equation(geometry, t, struts)),
            bounds=(thetas[i] - step, thetas[i] + step), method="bounded",
            options={"xatol": 1e-13},
        )
        best = min(best, result.fun / scale)
    return float(best)


def count_poses(geometry, struts, grid_size: int = GRID_SIZE,
                check_double_roots: bool = False):
    """Return ``(count, poses, n_double)`` for one configuration.

    ``count`` is the number of distinct real poses from the solver.
    ``n_double`` is how many extra tangent (double) roots were found; these
    only show up exactly at a transition and are reported separately.
    """
    poses = solve_forward_kinematics(geometry, struts, grid_size=grid_size)
    n_double = len(double_roots(geometry, struts, grid_size)) if check_double_roots else 0
    return len(poses), poses, n_double


def refine_transition(count_at, low: float, high: float,
                      tol: float = 1e-6, max_iter: int = 60) -> float:
    """Bisection on a *parameter* to find where the pose count changes.

    ``count_at(value)`` returns the pose count.  ``low`` and ``high`` must give
    different counts.  This is the same bisection idea as in Sauer ch. 1, but
    applied to the sweep parameter instead of to theta.
    """
    c_low = count_at(low)
    for _ in range(max_iter):
        if high - low <= tol:
            break
        mid = 0.5 * (low + high)
        if count_at(mid) == c_low:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_pose_grid(geometry, struts, poses, path: Path, title: str) -> None:
    """Draw every pose of one configuration in a single figure."""
    n = max(1, len(poses))
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (k, pose) in zip(axes.flat, enumerate(poses, start=1)):
        ax.set_visible(True)
        plot_platform_pose(geometry, pose, ax=ax, expected_lengths=struts,
                           title=f"Pose {k}: θ = {pose.theta:.3f} rad")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_f(ax, geometry, struts, label: str, samples: int = 1500) -> None:
    thetas = np.linspace(-np.pi, np.pi, samples)
    ax.plot(thetas, f_values(geometry, struts, thetas), label=label)
