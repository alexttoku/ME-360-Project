"""Run the planar Stewart-platform textbook acceptance tests.

The reusable implementation is in ``Functions/Exports.py``.  This file is
kept as a case-input driver: inputs are declared in spaced sections and then
passed to exported functions below each case comment.
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FUNCTION_DIR = SCRIPT_DIR.parent / "Functions"
OUTPUT_DIR = SCRIPT_DIR / "output"


def load_exports():
    """Load Exports.py by path so editors can resolve the file reliably."""
    exports_path = FUNCTION_DIR / "Exports.py"
    spec = importlib.util.spec_from_file_location("stewart_exports", exports_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validation exports from {exports_path}")
    exports = importlib.util.module_from_spec(spec)
    sys.modules["stewart_exports"] = exports
    spec.loader.exec_module(exports)
    return exports


exports = load_exports()
PlatformGeometry = exports.PlatformGeometry
run_convergence_test = exports.run_convergence_test
run_figure_115_test = exports.run_figure_115_test
run_textbook_pose_test = exports.run_textbook_pose_test
write_validation_summary = exports.write_validation_summary


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sauer Figure 1.15: two known poses at theta = +/- pi/4.
    # ------------------------------------------------------------------
    figure_115_geometry = PlatformGeometry(
        x1=4.0,
        x2=0.0,
        y2=4.0,
        L1=2.0,
        L2=np.sqrt(2),
        L3=np.sqrt(2),
        gamma=np.pi / 2,
    )

    figure_115_strut_lengths = np.array(
        [np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0)]
    )

    figure_115_result = run_figure_115_test(
        geometry=figure_115_geometry,
        strut_lengths=figure_115_strut_lengths,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Sauer four-pose example.
    # ------------------------------------------------------------------
    four_pose_geometry = PlatformGeometry(
        x1=5.0,
        x2=0.0,
        y2=6.0,
        L1=3.0,
        L2=3.0 * np.sqrt(2),
        L3=3.0,
        gamma=np.pi / 4,
    )

    four_pose_strut_lengths = np.array([5.0, 5.0, 3.0])

    four_pose_result, four_pose_solutions = run_textbook_pose_test(
        name="Four-pose example",
        geometry=four_pose_geometry,
        strut_lengths=four_pose_strut_lengths,
        expected_count=4,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Sauer six-pose example: change only p2 from 5 to 7.
    # ------------------------------------------------------------------
    six_pose_strut_lengths = np.array([5.0, 7.0, 3.0])

    six_pose_result, _ = run_textbook_pose_test(
        name="Six-pose example",
        geometry=four_pose_geometry,
        strut_lengths=six_pose_strut_lengths,
        expected_count=6,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # Root-method comparison using the four-pose case and same brackets.
    # ------------------------------------------------------------------
    convergence_result = run_convergence_test(
        geometry=four_pose_geometry,
        strut_lengths=four_pose_strut_lengths,
        poses=four_pose_solutions,
        output_dir=OUTPUT_DIR,
    )

    results = [
        figure_115_result,
        four_pose_result,
        six_pose_result,
        convergence_result,
    ]
    summary = write_validation_summary(results, OUTPUT_DIR)
    print(summary)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
