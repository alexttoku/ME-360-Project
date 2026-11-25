"""Convergence history of bisection vs. Brent's method (report Fig. 4).

Uses the same solver functions, brackets and tolerance (1e-12) as the
"Bisection versus Brent" test in Textbook Validation.py, on Sauer's
four-pose case.  Every point at which f(theta) is evaluated is recorded,
so the plot shows |theta_k - theta*| against the number of function
evaluations for each root.

Run from this folder:
    uv run python "Convergence Plot.py"
Writes output/convergence_comparison/convergence_history.png and .csv
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FUNCTION_DIR = SCRIPT_DIR.parent / "Functions"
OUT = SCRIPT_DIR / "output" / "convergence_comparison"
TOL = 1e-12


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, FUNCTION_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geometry_module = _load("stewart_geometry", "Geometry & Inverse Kinematics.py")
forward = _load("stewart_forward", "Forward Kinematics.py")


class Recorder:
    """Wrap f(theta) and remember every theta it is evaluated at."""

    def __init__(self, func):
        self.func = func
        self.points: list[float] = []

    def __call__(self, theta):
        self.points.append(float(theta))
        return self.func(theta)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geometry = geometry_module.PlatformGeometry(
        x1=5.0, x2=0.0, y2=6.0, L1=3.0, L2=3.0 * np.sqrt(2), L3=3.0, gamma=np.pi / 4)
    struts = np.array([5.0, 5.0, 3.0])
    f = lambda t: forward.forward_equation(geometry, t, struts)

    brackets, _ = forward.root_brackets(geometry, struts)
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10.colors
    for k, (a, b) in enumerate(brackets, start=1):
        # Reference root: Brent to machine precision
        reference = forward.brentq(f, a, b, xtol=1e-15, rtol=4 * np.finfo(float).eps)

        bis = Recorder(f)
        forward.bisect_root(bis, a, b, tolerance=TOL)
        bre = Recorder(f)
        forward.brentq(bre, a, b, xtol=TOL)

        for method, rec in (("bisection", bis), ("brent", bre)):
            for n, theta in enumerate(rec.points, start=1):
                rows.append({"root": k, "method": method, "evaluation": n,
                             "theta": f"{theta:.16f}",
                             "abs_error": f"{abs(theta - reference):.3e}"})

        c = colors[(k - 1) % 10]
        # First two bisection evaluations are the bracket ends; plot from the 3rd
        e_bis = np.maximum(np.abs(np.array(bis.points[2:]) - reference), 1e-17)
        e_bre = np.maximum(np.abs(np.array(bre.points) - reference), 1e-17)
        ax.semilogy(np.arange(3, 3 + len(e_bis)), e_bis, "-", color=c, lw=1,
                    label="Bisection" if k == 1 else None)
        ax.semilogy(np.arange(1, 1 + len(e_bre)), e_bre, "--o", color=c, ms=3, lw=1,
                    label="Brent" if k == 1 else None)
        print(f"root {k}: theta* = {reference:.10f}, bisection {len(bis.points)} f evals, "
              f"Brent {len(bre.points)} f evals")

    ax.axhline(TOL, color="gray", ls=":", lw=1, label="tolerance $10^{-12}$")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(r"$|\theta_k - \theta^*|$ (rad)")
    ax.set_title("Bisection vs. Brent, four-pose case (one color per root)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "convergence_history.png", dpi=200)
    plt.close(fig)

    with (OUT / "convergence_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUT / 'convergence_history.png'}")


if __name__ == "__main__":
    main()
