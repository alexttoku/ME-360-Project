"""Reusable textbook-validation exports.

The validation driver keeps only case inputs and the order in which cases are
run.  The file contains the reusable work: loading the solver modules,
running cases, writing CSV files, and exporting figures.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FUNCTION_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    """Load a module whose filename contains spaces or punctuation."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry_module = load_module(
    "stewart_geometry", FUNCTION_DIR / "Geometry & Inverse Kinematics.py"
)
forward_module = load_module(
    "stewart_forward", FUNCTION_DIR / "Forward Kinematics.py"
)
verification_module = load_module(
    "stewart_verification", FUNCTION_DIR / "Verification & Plotting.py"
)
PlatformGeometry = geometry_module.PlatformGeometry


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a list of dictionaries as a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_equation_plot(geometry, strut_lengths, poses, folder: Path) -> None:
    """Save the scalar forward equation with recovered roots marked."""
    folder.mkdir(parents=True, exist_ok=True)
    equation = lambda theta: forward_module.forward_equation(
        geometry, theta, strut_lengths
    )
    figure, _ = verification_module.plot_forward_equation(
        equation, [pose.theta for pose in poses]
    )
    figure.savefig(folder / "forward_equation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_pose_plots(geometry, strut_lengths, poses, folder: Path) -> None:
    """Save every recovered pose as an individual image."""
    folder.mkdir(parents=True, exist_ok=True)
    for index, pose in enumerate(poses, start=1):
        figure, _ = verification_module.plot_platform_pose(
            geometry,
            pose,
            expected_lengths=strut_lengths,
            title=f"Pose {index} (theta = {pose.theta:.8f} rad)",
        )
        figure.savefig(folder / f"pose_{index}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def pose_rows(geometry, strut_lengths, poses) -> list[dict[str, object]]:
    """Create a CSV-ready table of poses and inverse-kinematics errors."""
    rows = []
    for index, pose in enumerate(poses, start=1):
        result = verification_module.verify_pose(geometry, pose, strut_lengths)
        rows.append(
            {
                "pose": index,
                "theta_rad": f"{pose.theta:.15g}",
                "x": f"{pose.x:.15g}",
                "y": f"{pose.y:.15g}",
                "p1_error": f"{result['errors'][0]:.6e}",
                "p2_error": f"{result['errors'][1]:.6e}",
                "p3_error": f"{result['errors'][2]:.6e}",
                "max_absolute_error": f"{result['max_absolute_error']:.6e}",
                "passed": result["passed"],
            }
        )
    return rows


def run_figure_115_test(geometry, strut_lengths, output_dir: Path) -> dict[str, object]:
    """Run Sauer's Figure 1.15 case and check its two known roots."""
    test_folder = output_dir / "figure_1_15"
    test_folder.mkdir(parents=True, exist_ok=True)
    poses = forward_module.solve_forward_kinematics(geometry, strut_lengths)
    rows = pose_rows(geometry, strut_lengths, poses)
    write_csv(test_folder / "poses.csv", rows)
    save_equation_plot(geometry, strut_lengths, poses, test_folder)
    save_pose_plots(geometry, strut_lengths, poses, test_folder)

    known_angles = np.array([-np.pi / 4, np.pi / 4])
    recovered_angles = np.array([pose.theta for pose in poses])
    angle_matches = [
        bool(np.any(np.isclose(recovered_angles, angle, atol=1e-8)))
        for angle in known_angles
    ]
    return {
        "name": "Figure 1.15",
        "expected_pose_count": 2,
        "actual_pose_count": len(poses),
        "known_angle_matches": angle_matches,
        "max_absolute_error": max(
            (float(row["max_absolute_error"]) for row in rows), default=0.0
        ),
        "passed": (
            len(poses) == 2
            and all(angle_matches)
            and all(row["passed"] for row in rows)
        ),
    }


def run_textbook_pose_test(
    name: str,
    geometry,
    strut_lengths,
    expected_count: int,
    output_dir: Path,
) -> tuple[dict[str, object], list]:
    """Run and export one four-pose or six-pose acceptance case."""
    folder_name = "four_pose" if expected_count == 4 else "six_pose"
    test_folder = output_dir / folder_name
    test_folder.mkdir(parents=True, exist_ok=True)
    poses = forward_module.solve_forward_kinematics(geometry, strut_lengths)
    rows = pose_rows(geometry, strut_lengths, poses)
    write_csv(test_folder / "poses.csv", rows)
    save_equation_plot(geometry, strut_lengths, poses, test_folder)
    save_pose_plots(geometry, strut_lengths, poses, test_folder)
    result = {
        "name": name,
        "expected_pose_count": expected_count,
        "actual_pose_count": len(poses),
        "max_absolute_error": max(
            (float(row["max_absolute_error"]) for row in rows), default=0.0
        ),
        "passed": len(poses) == expected_count and all(row["passed"] for row in rows),
    }
    return result, poses


def run_convergence_test(geometry, strut_lengths, poses, output_dir: Path) -> dict[str, object]:
    """Compare bisection and Brent using the same root brackets."""
    test_folder = output_dir / "convergence_comparison"
    test_folder.mkdir(parents=True, exist_ok=True)
    brackets, _ = forward_module.root_brackets(geometry, strut_lengths)
    function = lambda theta: forward_module.forward_equation(
        geometry, theta, strut_lengths
    )
    rows = []
    for index, (left, right) in enumerate(brackets, start=1):
        bisection_root, iterations, evaluations = forward_module.bisect_root(
            function, left, right, tolerance=1e-12
        )
        brent_evaluations = [0]

        def counted_function(theta):
            brent_evaluations[0] += 1
            return function(theta)

        brent_root = forward_module.brentq(
            counted_function, left, right, xtol=1e-12
        )
        rows.append(
            {
                "root": index,
                "left": f"{left:.15g}",
                "right": f"{right:.15g}",
                "bisection_root": f"{bisection_root:.15g}",
                "brent_root": f"{brent_root:.15g}",
                "bisection_iterations": iterations,
                "bisection_function_evaluations": evaluations,
                "brent_function_evaluations": brent_evaluations[0],
                "root_difference": f"{abs(bisection_root - brent_root):.6e}",
                "bisection_residual": f"{abs(function(bisection_root)):.6e}",
                "brent_residual": f"{abs(function(brent_root)):.6e}",
            }
        )
    write_csv(test_folder / "bisection_vs_brent.csv", rows)
    return {
        "name": "Bisection versus Brent",
        "bracket_count": len(brackets),
        "root_count_match": len(brackets) == len(poses),
        "max_root_difference": max(
            (float(row["root_difference"]) for row in rows), default=0.0
        ),
        "passed": len(brackets) == len(poses),
    }


def write_validation_summary(results: list[dict[str, object]], output_dir: Path) -> dict[str, object]:
    """Write JSON and text summaries and return the summary dictionary."""
    summary = {
        "tests": results,
        "all_passed": all(result["passed"] for result in results),
        "output_directory": str(output_dir),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "validation_summary.txt").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(f"{result['name']}: {'PASS' if result['passed'] else 'FAIL'}\n")
            file.write(f"  {json.dumps(result)}\n")
        file.write(f"Overall: {'PASS' if summary['all_passed'] else 'FAIL'}\n")
    return summary


__all__ = [
    "PlatformGeometry",
    "run_convergence_test",
    "run_figure_115_test",
    "run_textbook_pose_test",
    "write_validation_summary",
]
