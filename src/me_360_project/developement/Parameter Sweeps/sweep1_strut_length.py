"""Sweep 1: strut length p2.

Hold Sauer's four-pose geometry and p1 = 5, p3 = 3 fixed, and vary p2.
For each p2 we count the poses, then refine every place where the count
changes and check that a double root sits exactly at each transition.

Outputs (in output/sweep1_p2/):
    pose_count_vs_p2.png, pose_count_vs_p2.csv, transitions.csv,
    f_theta_examples.png, poses_p2_<value>.png (one per pose-count regime)
"""

import matplotlib.pyplot as plt
import numpy as np

import sweep_common as sc

OUT = sc.output_dir("sweep1_p2")
geometry = sc.make_geometry()
P1, P3 = 5.0, 3.0


def struts(p2):
    return np.array([P1, p2, P3])


def count_at(p2):
    return sc.count_poses(geometry, struts(p2))[0]


def count_fine(p2):
    # 10x finer theta grid, used when pinning down transitions, because two
    # roots that are about to merge can sit closer together than the default
    # grid spacing (seen near p2 = 7.85).
    return sc.count_poses(geometry, struts(p2), grid_size=20001)[0]


def main():
    # 1) Coarse sweep -------------------------------------------------------
    p2_values = np.round(np.arange(0.5, 12.0001, 0.05), 4)
    counts = np.array([count_at(p2) for p2 in p2_values])
    sc.write_csv(OUT / "pose_count_vs_p2.csv",
                 [{"p2": p, "pose_count": int(c)} for p, c in zip(p2_values, counts)])

    # 2) Refine each change in pose count -----------------------------------
    transitions = []
    for i in range(1, len(p2_values)):
        if counts[i] != counts[i - 1]:
            p2_star = sc.refine_transition(count_fine, p2_values[i - 1], p2_values[i])
            coarse_star = sc.refine_transition(count_at, p2_values[i - 1], p2_values[i])
            # Double-root check: step to the side with FEWER poses.  There the
            # two merging roots have just vanished, so f has a local minimum
            # that barely misses zero.  Its relative size should shrink toward
            # 0 as we approach the transition (1e-5 away vs 0.05 away).
            side = -1.0 if counts[i - 1] < counts[i] else 1.0
            tangent_near = sc.tangent_minimum(geometry, struts(p2_star + side * 1e-5))
            tangent_far = sc.tangent_minimum(geometry, struts(p2_star + side * 0.05))
            transitions.append({
                "p2_transition": f"{p2_star:.6f}",
                "p2_transition_default_grid": f"{coarse_star:.6f}",
                "count_below": int(counts[i - 1]),
                "count_above": int(counts[i]),
                "near_zero_min_rel_f_1e-5_away": f"{tangent_near:.2e}",
                "near_zero_min_rel_f_0.05_away": f"{tangent_far:.2e}",
            })
    sc.write_csv(OUT / "transitions.csv", transitions)

    # Interval table (0, 2, 4, 6 poses)
    edges = [p2_values[0]] + [float(t["p2_transition"]) for t in transitions] + [p2_values[-1]]
    regime_counts = [int(counts[0])] + [t["count_above"] for t in transitions]
    intervals = [{"p2_from": f"{a:.4f}", "p2_to": f"{b:.4f}", "pose_count": c}
                 for a, b, c in zip(edges[:-1], edges[1:], regime_counts)]
    sc.write_csv(OUT / "intervals.csv", intervals)

    # 3) Grid-resolution check: same counts with a 5x finer theta grid ------
    check = p2_values[::10]
    fine = [sc.count_poses(geometry, struts(p), grid_size=10001)[0] for p in check]
    coarse = [count_at(p) for p in check]
    grid_ok = fine == coarse

    # 4) Plots -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.step(p2_values, counts, where="mid", color="tab:blue")
    for t in transitions:
        ax.axvline(float(t["p2_transition"]), color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Strut length $p_2$")
    ax.set_ylabel("Number of poses")
    ax.set_yticks([0, 2, 4, 6])
    ax.set_title("Sweep 1: pose count vs. $p_2$ ($p_1=5$, $p_3=3$)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pose_count_vs_p2.png", dpi=150)
    plt.close(fig)

    # One representative p2 from the middle of each interval
    reps = []
    for row in intervals:
        mid = 0.5 * (float(row["p2_from"]) + float(row["p2_to"]))
        if row["pose_count"] not in [r[1] for r in reps] or row["pose_count"] == 6:
            reps.append((round(mid, 3), row["pose_count"]))
    reps = [r for r in reps if r[1] > 0][:4]

    fig, ax = plt.subplots(figsize=(7, 4))
    for p2, c in reps:
        sc.plot_f(ax, geometry, struts(p2), f"$p_2$ = {p2} ({c} poses)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$f(\theta)$")
    ax.set_title(r"$f(\theta)$ for representative $p_2$ values")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "f_theta_examples.png", dpi=150)
    plt.close(fig)

    for p2, c in reps:
        _, poses, _ = sc.count_poses(geometry, struts(p2))
        sc.save_pose_grid(geometry, struts(p2), poses,
                          OUT / f"poses_p2_{p2:.3f}.png",
                          f"$p_2$ = {p2}: {c} poses")

    print("Sweep 1 intervals:")
    for row in intervals:
        print(f"  {row['p2_from']:>8} < p2 < {row['p2_to']:<8}  ->  {row['pose_count']} poses")
    print("Transitions:", transitions)
    print("Grid check (2001 vs 10001 theta points) agrees:", grid_ok)


if __name__ == "__main__":
    main()
