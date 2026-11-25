"""Sweep 2: anchor positions.

Keep the moving platform (L1, L2, L3, gamma) and the strut lengths
(p1, p2, p3) = (5, 5, 3) from Sauer's four-pose example fixed, and move the
fixed anchors instead:

  Part A: slide anchor 2 along the x-axis, x1 from 0.25 to 10.
  Part B: move anchor 3 over a 2-D grid (once with struts (5, 5, 3) and
          once with Sauer's six-pose struts (5, 7, 3)) of (x2, y2) and colour each point
          by the number of poses (the "pose-count map").
  Part C: compare f(theta) and the platform drawings for a few anchor layouts.

Outputs (in output/sweep2_anchors/):
    pose_count_vs_x1.png/.csv, x1_transitions.csv,
    pose_count_map_x2_y2.png/.csv, f_theta_anchor_examples.png,
    poses_<layout>.png
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

import sweep_common as sc

OUT = sc.output_dir("sweep2_anchors")
STRUTS = sc.BASE_STRUTS


def count(**anchor_changes):
    return sc.count_poses(sc.make_geometry(**anchor_changes), STRUTS)[0]


def count_fine(**anchor_changes):
    return sc.count_poses(sc.make_geometry(**anchor_changes), STRUTS, grid_size=20001)[0]


def part_a():
    # Step 0.01: a 0.05 step jumps over the narrow 6-pose window near x1 = 3.
    x1_values = np.round(np.arange(0.25, 10.0001, 0.01), 4)
    counts = np.array([count(x1=x1) for x1 in x1_values])
    sc.write_csv(OUT / "pose_count_vs_x1.csv",
                 [{"x1": x, "pose_count": int(c)} for x, c in zip(x1_values, counts)])

    transitions = []
    for i in range(1, len(x1_values)):
        if counts[i] != counts[i - 1]:
            x1_star = sc.refine_transition(lambda v: count_fine(x1=v),
                                           x1_values[i - 1], x1_values[i])
            transitions.append({"x1_transition": f"{x1_star:.6f}",
                                "count_below": int(counts[i - 1]),
                                "count_above": int(counts[i])})
    sc.write_csv(OUT / "x1_transitions.csv", transitions)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.step(x1_values, counts, where="mid")
    ax.axvline(sc.BASE["x1"], color="tab:red", ls=":", label="Sauer baseline $x_1=5$")
    for t in transitions:
        ax.axvline(float(t["x1_transition"]), color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Anchor 2 position $x_1$")
    ax.set_ylabel("Number of poses")
    ax.set_yticks([0, 2, 4, 6])
    ax.set_title("Sweep 2A: pose count vs. $x_1$ (anchor 3 at (0, 6))")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pose_count_vs_x1.png", dpi=150)
    plt.close(fig)
    return transitions


def part_b(struts, tag):
    x2_values = np.round(np.arange(-4.0, 6.0001, 0.2), 4)
    y2_values = np.round(np.arange(1.0, 10.0001, 0.2), 4)
    grid = np.zeros((len(y2_values), len(x2_values)), dtype=int)
    for j, y2 in enumerate(y2_values):
        for i, x2 in enumerate(x2_values):
            grid[j, i] = sc.count_poses(sc.make_geometry(x2=x2, y2=y2), struts)[0]

    # Cells next to a region boundary are where two roots can be closer
    # together than the default theta grid spacing and get missed.  Recount
    # those cells with a 10x finer theta grid.
    boundary = np.zeros_like(grid, dtype=bool)
    boundary[:, 1:] |= grid[:, 1:] != grid[:, :-1]
    boundary[:, :-1] |= grid[:, 1:] != grid[:, :-1]
    boundary[1:, :] |= grid[1:, :] != grid[:-1, :]
    boundary[:-1, :] |= grid[1:, :] != grid[:-1, :]
    changed = 0
    for j, i in zip(*np.where(boundary)):
        fine = sc.count_poses(sc.make_geometry(x2=x2_values[i], y2=y2_values[j]),
                              struts, grid_size=20001)[0]
        changed += int(fine != grid[j, i])
        grid[j, i] = fine
    print(f"  [{tag}] rechecked {int(boundary.sum())} boundary cells with a fine grid; "
          f"{changed} changed")

    rows = [{"x2": x2, "y2": y2, "pose_count": int(grid[j, i])}
            for j, y2 in enumerate(y2_values) for i, x2 in enumerate(x2_values)]
    sc.write_csv(OUT / f"pose_count_map_x2_y2_{tag}.csv", rows)

    cmap = ListedColormap(["#eeeeee", "#9ecae1", "#3182bd", "#08519c"])
    norm = BoundaryNorm([-1, 1, 3, 5, 7], cmap.N)  # odd counts (rare) share a colour with count-1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    step = 0.1
    mesh = ax.pcolormesh(x2_values, y2_values, grid, cmap=cmap, norm=norm,
                         shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 2, 4, 6])
    cbar.set_label("Number of poses")
    # Odd counts mean a double root (two poses merged into one): the anchor
    # sits exactly on a transition.  For struts (5, 7, 3) many of these are the
    # flat singular pose at theta = 0, where strut 1, strut 2 and the platform
    # side L3 all lie on the x-axis (vertex 1 at (-p1, 0)).  That pose exists
    # whenever x1 + p1 - L3 = p2, for any anchor 3 at distance p3 from
    # vertex 3, i.e. on a circle.  We draw that circle when it applies.
    odd = [(r["x2"], r["y2"]) for r in rows if r["pose_count"] % 2]
    if odd:
        ax.plot(*zip(*odd), "rx", ms=7, mew=2, label="odd count = double root")
    g0 = sc.make_geometry()
    p1, p2, p3 = struts
    if np.isclose(g0.x1 + p1 - g0.L3, p2):
        v3 = g0.vertices(-p1, 0.0, 0.0)[2]
        a = np.linspace(0, 2 * np.pi, 300)
        ax.plot(v3[0] + p3 * np.cos(a), v3[1] + p3 * np.sin(a), "r--", lw=1,
                label="flat singular pose exists (θ = 0)")
        ax.set_xlim(x2_values[0] - 0.1, x2_values[-1] + 0.1)
        ax.set_ylim(y2_values[0] - 0.1, y2_values[-1] + 0.1)
    ax.plot(sc.BASE["x2"], sc.BASE["y2"], "r*", ms=12, label="Sauer baseline (0, 6)")
    ax.set_xlabel("Anchor 3 $x_2$")
    ax.set_ylabel("Anchor 3 $y_2$")
    ax.set_title("Sweep 2B: pose count vs. anchor 3 position\n"
                 f"(anchors 1, 2 fixed at (0, 0), (5, 0); struts ({struts[0]:g}, {struts[1]:g}, {struts[2]:g}))")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"pose_count_map_x2_y2_{tag}.png", dpi=150)
    plt.close(fig)

    summary = {c: int(np.sum(grid == c)) for c in sorted(set(grid.flat))}
    six = [(r["x2"], r["y2"]) for r in rows if r["pose_count"] == 6]
    return summary, six, rows


def part_c(six_pose_points, six_struts):
    layouts = {"baseline_x1_5_anchor3_0_6": dict()}
    layouts["wide_base_x1_8"] = dict(x1=8.0)
    layouts["narrow_base_x1_2"] = dict(x1=2.0)
    if six_pose_points:
        # pick the six-pose point closest to the baseline anchor
        x2, y2 = min(six_pose_points, key=lambda p: (p[0] - 0) ** 2 + (p[1] - 6) ** 2)
        layouts[f"six_pose_anchor3_{x2:g}_{y2:g}"] = dict(x2=x2, y2=y2)
    # (the six-pose layouts use struts (5, 7, 3); the rest use (5, 5, 3))

    fig, ax = plt.subplots(figsize=(7, 4))
    results = []
    for name, changes in layouts.items():
        struts = six_struts if name.startswith("six_pose") else STRUTS
        g = sc.make_geometry(**changes)
        n, poses, _ = sc.count_poses(g, struts)
        sc.plot_f(ax, g, struts, f"{name} ({n} poses)")
        sc.save_pose_grid(g, struts, poses, OUT / f"poses_{name}.png",
                          f"{name}: {n} poses, struts {tuple(float(v) for v in struts)}")
        worst = max((sc.verify_pose(g, p, struts)["max_absolute_error"] for p in poses),
                    default=0.0)
        results.append({"layout": name, **{k: v for k, v in changes.items()},
                        "struts": "-".join(f"{v:g}" for v in struts),
                        "pose_count": n, "max_strut_error": f"{worst:.2e}"})
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$f(\theta)$")
    ax.set_title(r"$f(\theta)$ for different anchor layouts")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "f_theta_anchor_examples.png", dpi=150)
    plt.close(fig)
    sc.write_csv(OUT / "layout_examples.csv",
                 [{k: r.get(k, "") for k in ("layout", "x1", "x2", "y2", "struts", "pose_count",
                                             "max_strut_error")} for r in results])
    return results


def main():
    transitions = part_a()
    print("Part A transitions in x1:", transitions)
    summary, _, _ = part_b(STRUTS, "struts_5_5_3")
    print("Part B, struts (5,5,3), grid points by pose count:", summary)
    six_struts = np.array([5.0, 7.0, 3.0])
    summary6, six, _ = part_b(six_struts, "struts_5_7_3")
    print("Part B, struts (5,7,3), grid points by pose count:", summary6)
    if six:
        xs, ys = zip(*six)
        print(f"  six-pose region: x2 in [{min(xs)}, {max(xs)}], y2 in [{min(ys)}, {max(ys)}]")
    for row in part_c(six, six_struts):
        print("Part C:", row)


if __name__ == "__main__":
    main()
