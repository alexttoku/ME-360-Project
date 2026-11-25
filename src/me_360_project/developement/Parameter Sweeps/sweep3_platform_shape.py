"""Sweep 3: platform size and shape, plus the reachable workspace.

Anchors stay at Sauer's (0,0), (5,0), (0,6) and struts at (5, 5, 3).

  Part A: scale the whole triangle, L_i = s * L_i(baseline), gamma fixed.
  Part B: change gamma with L2 = 3*sqrt(2), L3 = 3 fixed; L1 follows from the
          law of cosines so the triangle stays valid.
  Part C: workspace.  Real struts have travel limits, here
          P_MIN <= p_i <= P_MAX for every strut.  A pose (x, y, theta) is
          reachable if inverse kinematics gives three lengths inside the
          limits.  We map which (x, y) are reachable at fixed theta, and for
          how many orientations theta each (x, y) is reachable.  Everything
          outside the coloured region can never be reached by moving the
          struts alone.

Outputs (in output/sweep3_platform/):
    pose_count_vs_scale.png/.csv, pose_count_vs_gamma.png/.csv,
    transitions_scale.csv, transitions_gamma.csv,
    workspace_fixed_theta.png, workspace_orientation_count.png,
    workspace_area_vs_scale.csv
"""

import matplotlib.pyplot as plt
import numpy as np

import sweep_common as sc

OUT = sc.output_dir("sweep3_platform")
STRUTS = sc.BASE_STRUTS
L1_0, L2_0, L3_0, GAMMA_0 = sc.BASE["L1"], sc.BASE["L2"], sc.BASE["L3"], sc.BASE["gamma"]

# Strut travel limits used for the workspace (assumed actuator stroke).
P_MIN, P_MAX = 2.0, 8.0
X_RANGE, Y_RANGE = (-9.0, 14.0), (-9.0, 15.0)  # wide enough: p1 <= 8 circle fits
WORKSPACE_STEP = 0.1
N_THETA = 72  # orientations checked per (x, y): every 5 degrees


def scaled(s):
    return sc.make_geometry(L1=s * L1_0, L2=s * L2_0, L3=s * L3_0)


def with_gamma(g):
    L1 = np.sqrt(L2_0**2 + L3_0**2 - 2 * L2_0 * L3_0 * np.cos(g))
    return sc.make_geometry(L1=L1, gamma=g)


def sweep(values, make, fine_make=None):
    counts = np.array([sc.count_poses(make(v), STRUTS)[0] for v in values])
    transitions = []
    for i in range(1, len(values)):
        if counts[i] != counts[i - 1]:
            star = sc.refine_transition(
                lambda v: sc.count_poses(make(v), STRUTS, grid_size=20001)[0],
                values[i - 1], values[i])
            transitions.append({"transition": f"{star:.6f}",
                                "count_below": int(counts[i - 1]),
                                "count_above": int(counts[i])})
    return counts, transitions


def step_plot(values, counts, transitions, xlabel, title, path, baseline):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.step(values, counts, where="mid")
    ax.axvline(baseline, color="tab:red", ls=":", label="Sauer baseline")
    for t in transitions:
        ax.axvline(float(t["transition"]), color="gray", lw=0.7, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of poses")
    ax.set_yticks([0, 2, 4, 6])
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strut_lengths_grid(geometry, X, Y, theta):
    """Vectorised inverse kinematics (same formulas as Geometry & IK.py)."""
    v2x = X + geometry.L3 * np.cos(theta)
    v2y = Y + geometry.L3 * np.sin(theta)
    v3x = X + geometry.L2 * np.cos(theta + geometry.gamma)
    v3y = Y + geometry.L2 * np.sin(theta + geometry.gamma)
    p1 = np.hypot(X, Y)
    p2 = np.hypot(v2x - geometry.x1, v2y)
    p3 = np.hypot(v3x - geometry.x2, v3y - geometry.y2)
    return p1, p2, p3


def reachable(geometry, X, Y, theta):
    p = strut_lengths_grid(geometry, X, Y, theta)
    return np.all([(P_MIN <= pi) & (pi <= P_MAX) for pi in p], axis=0)


def part_c():
    xs = np.arange(X_RANGE[0], X_RANGE[1] + 1e-9, WORKSPACE_STEP)
    ys = np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, WORKSPACE_STEP)
    X, Y = np.meshgrid(xs, ys)
    cell = WORKSPACE_STEP**2

    # Check the vectorised IK against the shared solver code at a few points.
    g0 = scaled(1.0)
    for x, y, t in [(1.0, 2.0, 0.3), (-2.0, 4.0, -1.0), (3.0, 3.0, 2.5)]:
        assert np.allclose(strut_lengths_grid(g0, x, y, t),
                           sc.inverse_kinematics(g0, x, y, t))

    # (1) fixed-orientation workspaces for three platform scales
    thetas = [0.0, np.pi / 4, np.pi / 2, -np.pi / 2]
    scales = [0.5, 1.0, 1.5]
    fig, axes = plt.subplots(len(scales), len(thetas), figsize=(13, 10),
                             sharex=True, sharey=True)
    for r, s in enumerate(scales):
        g = scaled(s)
        for c, t in enumerate(thetas):
            ax = axes[r, c]
            ax.contourf(X, Y, reachable(g, X, Y, t).astype(float),
                        levels=[0.5, 1.5], colors=["#3182bd"])
            ax.plot(*g.anchors.T, "ks", ms=4)
            ax.set_aspect("equal")
            ax.set_title(f"scale {s}, θ = {t:.2f}", fontsize=9)
            if r == len(scales) - 1:
                ax.set_xlabel("x")
            if c == 0:
                ax.set_ylabel("y")
    fig.suptitle(f"Reachable positions of vertex 1 at fixed θ "
                 f"(strut limits {P_MIN} ≤ p ≤ {P_MAX}); black squares = anchors")
    fig.tight_layout()
    fig.savefig(OUT / "workspace_fixed_theta.png", dpi=150)
    plt.close(fig)

    # (2) how many orientations are reachable at each (x, y)
    theta_samples = np.linspace(-np.pi, np.pi, N_THETA, endpoint=False)
    rows = []
    fig, axes = plt.subplots(1, len(scales), figsize=(14, 4.8))
    for ax, s in zip(axes, scales):
        g = scaled(s)
        n_reach = sum(reachable(g, X, Y, t).astype(int) for t in theta_samples)
        mesh = ax.pcolormesh(X, Y, np.where(n_reach > 0, n_reach / N_THETA * 360, np.nan),
                             shading="auto", cmap="viridis", vmin=0, vmax=360)
        ax.plot(*g.anchors.T, "rs", ms=4)
        ax.set_aspect("equal")
        ax.set_title(f"scale {s}")
        ax.set_xlabel("x")
        rows.append({
            "scale": s,
            "area_reachable_any_theta": f"{np.sum(n_reach > 0) * cell:.2f}",
            "area_reachable_all_theta": f"{np.sum(n_reach == N_THETA) * cell:.2f}",
            "area_theta_0": f"{np.sum(reachable(g, X, Y, 0.0)) * cell:.2f}",
        })
    axes[0].set_ylabel("y")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.85)
    cbar.set_label("Reachable range of θ at this (x, y) [degrees]")
    fig.suptitle("Orientation workspace: white = vertex 1 can never be placed here")
    fig.savefig(OUT / "workspace_orientation_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (3) workspace area vs scale
    for s in [0.25, 0.75, 1.25, 1.75, 2.0]:
        g = scaled(s)
        n_reach = sum(reachable(g, X, Y, t).astype(int) for t in theta_samples)
        rows.append({
            "scale": s,
            "area_reachable_any_theta": f"{np.sum(n_reach > 0) * cell:.2f}",
            "area_reachable_all_theta": f"{np.sum(n_reach == N_THETA) * cell:.2f}",
            "area_theta_0": f"{np.sum(reachable(g, X, Y, 0.0)) * cell:.2f}",
        })
    rows.sort(key=lambda r: r["scale"])
    sc.write_csv(OUT / "workspace_area_vs_scale.csv", rows)
    return rows


def main():
    # Part A: uniform scale
    # Step 0.005: a 0.02 step jumps over the narrow 4-pose window near s = 0.3.
    scales = np.round(np.arange(0.2, 2.0001, 0.005), 4)
    counts, trans = sweep(scales, scaled)
    sc.write_csv(OUT / "pose_count_vs_scale.csv",
                 [{"scale": s, "pose_count": int(c)} for s, c in zip(scales, counts)])
    sc.write_csv(OUT / "transitions_scale.csv", trans)
    step_plot(scales, counts, trans, "Platform scale factor s",
              "Sweep 3A: pose count vs. platform size (γ = π/4)",
              OUT / "pose_count_vs_scale.png", 1.0)
    print("Scale transitions:", trans)

    # Part B: gamma
    gammas = np.round(np.linspace(0.05, np.pi - 0.05, 150), 6)
    counts_g, trans_g = sweep(gammas, with_gamma)
    sc.write_csv(OUT / "pose_count_vs_gamma.csv",
                 [{"gamma_rad": g, "gamma_deg": f"{np.degrees(g):.2f}",
                   "L1": f"{with_gamma(g).L1:.4f}", "pose_count": int(c)}
                  for g, c in zip(gammas, counts_g)])
    sc.write_csv(OUT / "transitions_gamma.csv",
                 [{**t, "transition_deg": f"{np.degrees(float(t['transition'])):.3f}"}
                  for t in trans_g])
    step_plot(gammas, counts_g, trans_g, r"Platform angle $\gamma$ (rad)",
              r"Sweep 3B: pose count vs. $\gamma$ ($L_2=3\sqrt{2}$, $L_3=3$)",
              OUT / "pose_count_vs_gamma.png", GAMMA_0)
    print("Gamma transitions:", [(t["transition"], t["count_below"], t["count_above"])
                                  for t in trans_g])

    # Representative pose drawings
    for s in [0.6, 1.0, 1.5]:
        g = scaled(s)
        n, poses, _ = sc.count_poses(g, STRUTS)
        sc.save_pose_grid(g, STRUTS, poses, OUT / f"poses_scale_{s}.png",
                          f"scale {s}: {n} poses")

    # Part C: workspace
    for row in part_c():
        print("Workspace:", row)


if __name__ == "__main__":
    main()
