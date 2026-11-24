"""Sweep 4: simultaneous strut actuation, mode tracking, and animation.

Part A (prescribed motion):
    Choose a smooth platform motion x(t), y(t), theta(t) starting from one of
    Sauer's four poses.  Inverse kinematics gives the strut-length history
    p_i(t).  We then forget the motion, feed only p_i(t) to the forward
    solver, and at each time step keep the pose closest to the previous one
    (position + wrapped angle).  The difference from the prescribed motion
    measures how well the solver + tracking recover the motion.

Part B (assembly-mode switching):
    Start in each of the four poses at (p1, p2, p3) = (5, 5, 3) and slowly
    lengthen strut 2.  When the tracked pose meets another pose (a double
    root / singular configuration) it disappears.  At that point we reverse
    strut 2 and continue on the *partner* branch back to p2 = 5.  The struts
    return to exactly (5, 5, 3) but the platform ends up in a different
    assembly mode: the answer to "can strut motion take the platform from one
    solution to another?" is yes, but only by passing through a singularity.

Outputs (in output/sweep4_actuation/):
    A_strut_lengths_vs_time.png, A_corner_trajectories.png, A_pose_error.png, A_theta_tracking.png,
    A_motion.gif, A_history.csv,
    B_branch_diagram.png, B_mode_switch.gif, B_mode_switch_path.csv,
    B_mode_switch_summary.csv, B_mode_switch_start_end.png
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import sweep_common as sc

OUT = sc.output_dir("sweep4_actuation")
geometry = sc.make_geometry()
GRID = 4001  # theta grid for each forward solve


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose_distance(a, b):
    """Distance between poses: position plus wrapped orientation."""
    return np.hypot(a[0] - b[0], a[1] - b[1]) + abs(wrap(a[2] - b[2]))


def nearest(poses, reference):
    if not poses:
        return None
    return min(poses, key=lambda p: pose_distance(p.as_tuple, reference))


# ----------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------
def part_a():
    start_poses = sc.solve_forward_kinematics(geometry, sc.BASE_STRUTS)
    start = start_poses[1]  # second of Sauer's four poses
    T, N = 10.0, 201
    t = np.linspace(0.0, T, N)
    w = 2 * np.pi / T
    x = start.x + 0.8 * np.sin(w * t)
    y = start.y + 0.5 * (1 - np.cos(w * t))
    theta = start.theta + 0.35 * np.sin(w * t)  # rotation, angular speed 0.35*w

    struts = np.array([sc.inverse_kinematics(geometry, *q) for q in zip(x, y, theta)])

    def run_tracking(use_predictor):
        """Feed only the strut lengths to the solver and follow one pose.

        Naive: keep the pose nearest the previous pose.
        Predictor: keep the pose nearest a linear extrapolation
        2*q[k-1] - q[k-2], i.e. assume the platform keeps its current velocity.
        This matters where two assembly modes pass very close to each other.
        """
        out, n_poses, err, s_err = [], [], [], []
        for k in range(N):
            poses = sc.solve_forward_kinematics(geometry, struts[k], grid_size=GRID)
            if k == 0:
                reference = start.as_tuple
            elif k == 1 or not use_predictor:
                reference = out[-1]
            else:
                q1, q0 = np.array(out[-1]), np.array(out[-2])
                reference = tuple(2 * q1 - q0)
            best = nearest(poses, reference)
            out.append(best.as_tuple)
            n_poses.append(len(poses))
            err.append(pose_distance(best.as_tuple, (x[k], y[k], theta[k])))
            s_err.append(sc.verify_pose(geometry, best, struts[k])["max_absolute_error"])
        return np.array(out), n_poses, err, s_err

    naive, _, naive_err, _ = run_tracking(False)
    tracked, counts, errors, ik_err = run_tracking(True)
    switch_k = next((k for k, e in enumerate(naive_err) if e > 1e-6), None)

    sc.write_csv(OUT / "A_history.csv", [
        {"t": f"{t[k]:.3f}", "p1": f"{struts[k,0]:.6f}", "p2": f"{struts[k,1]:.6f}",
         "p3": f"{struts[k,2]:.6f}", "x_prescribed": f"{x[k]:.6f}",
         "y_prescribed": f"{y[k]:.6f}", "theta_prescribed": f"{theta[k]:.6f}",
         "x_solved": f"{tracked[k,0]:.6f}", "y_solved": f"{tracked[k,1]:.6f}",
         "theta_solved": f"{tracked[k,2]:.6f}", "pose_count": counts[k],
         "pose_error": f"{errors[k]:.3e}", "strut_error": f"{ik_err[k]:.3e}",
         "theta_naive_tracking": f"{naive[k,2]:.6f}",
         "pose_error_naive_tracking": f"{naive_err[k]:.3e}"}
        for k in range(N)])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i in range(3):
        ax.plot(t, struts[:, i], label=f"$p_{i+1}$")
    ax.set_xlabel("time t")
    ax.set_ylabel("strut length")
    ax.set_title("Sweep 4A: strut lengths needed for the prescribed motion")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "A_strut_lengths_vs_time.png", dpi=150)
    plt.close(fig)

    corners = np.array([geometry.vertices(*q) for q in tracked])  # (N, 3, 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, colour in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(corners[:, i, 0], corners[:, i, 1], color=colour, label=f"corner {i+1}")
    sc.plot_platform_pose(geometry, tracked[0], ax=ax, show_labels=False,
                          title="Sweep 4A: corner trajectories (platform at t = 0)")
    ax.plot(*geometry.anchors.T, "ks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "A_corner_trajectories.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.semilogy(t, np.maximum(naive_err, 1e-17), color="tab:red", alpha=0.7,
                label="pose error, naive nearest-pose tracking")
    ax.semilogy(t, np.maximum(errors, 1e-17), label="pose error, predictor tracking")
    ax.semilogy(t, np.maximum(ik_err, 1e-17), label="max strut-length error")
    ax2 = ax.twinx()
    ax2.step(t, counts, where="mid", color="gray", alpha=0.5)
    ax2.set_ylabel("poses available", color="gray")
    ax2.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("time t")
    ax.set_ylabel("error")
    ax.set_title("Sweep 4A: tracking error")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "A_pose_error.png", dpi=150)
    plt.close(fig)

    animate(tracked, [f"t = {tk:.1f}" for tk in t], OUT / "A_motion.gif",
            "Sweep 4A: platform following prescribed motion", trail=True)
    # theta histories: where the naive tracker jumps branch
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(t, theta, "k--", lw=2, label="prescribed θ(t)")
    ax.plot(t, naive[:, 2], color="tab:red", label="naive tracking")
    ax.plot(t, tracked[:, 2], color="tab:blue", lw=1, label="predictor tracking")
    if switch_k is not None:
        ax.axvline(t[switch_k], color="gray", ls=":", label="naive tracker switches mode")
    ax.set_xlabel("time t")
    ax.set_ylabel("θ (rad)")
    ax.set_title("Sweep 4A: orientation recovered from strut lengths only")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "A_theta_tracking.png", dpi=150)
    plt.close(fig)

    return {"max_pose_error_predictor": max(errors),
            "max_pose_error_naive": max(naive_err),
            "naive_switch_time": None if switch_k is None else float(t[switch_k]),
            "max_strut_error": max(ik_err),
            "min_pose_count": min(counts), "max_pose_count": max(counts)}


# ----------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------
def local_root(struts, theta_guess, half_width=0.03, samples=61):
    """Find the root of f(theta) closest to theta_guess, searching nearby only.

    Much faster than a full [-pi, pi] scan, which is what makes tracking
    thousands of small strut-length steps practical.  Returns None if f has
    no sign change near the guess (the tracked pose has vanished).
    """
    thetas = np.linspace(theta_guess - half_width, theta_guess + half_width, samples)
    vals = sc.f_values(geometry, struts, thetas)
    roots = []
    for i in range(samples - 1):
        if np.signbit(vals[i]) != np.signbit(vals[i + 1]):
            roots.append(sc.forward_module.brentq(
                lambda t: sc.forward_equation(geometry, t, struts),
                thetas[i], thetas[i + 1], xtol=1e-13))
    if not roots:
        return None
    return min(roots, key=lambda r: abs(r - theta_guess))


def partner_root(struts, theta_own, half_width=0.05):
    """Root of f closest to theta_own that is NOT theta_own itself.

    Just before two poses merge, their thetas are very close, so we search a
    small window with a very fine grid and exclude our own root.
    """
    thetas = np.linspace(theta_own - half_width, theta_own + half_width, 40001)
    vals = sc.f_values(geometry, struts, thetas)
    roots = []
    for i in range(len(thetas) - 1):
        if np.signbit(vals[i]) != np.signbit(vals[i + 1]):
            roots.append(sc.forward_module.brentq(
                lambda t: sc.forward_equation(geometry, t, struts),
                thetas[i], thetas[i + 1], xtol=1e-14))
    others = [r for r in roots if abs(r - theta_own) > 1e-10]
    return min(others, key=lambda r: abs(r - theta_own)) if others else None


def track_branch(theta_start, p2_values):
    """Follow one pose as p2 changes.  Stops when the pose disappears.

    Uses a linear predictor in theta, like Part A.
    """
    path = []
    for p2 in p2_values:
        struts = np.array([5.0, p2, 3.0])
        if len(path) >= 2:
            guess = 2 * path[-1][3] - path[-2][3]
        elif path:
            guess = path[-1][3]
        else:
            guess = theta_start
        theta = local_root(struts, guess)
        if theta is None:
            return path
        x, y = sc.forward_module.recover_xy(geometry, theta, struts)
        path.append((p2, x, y, theta))
    return path


def part_b():
    # branch diagram: theta of every pose vs p2
    p2_grid = np.arange(3.0, 9.8, 0.01)
    pts = []
    for p2 in p2_grid:
        for p in sc.solve_forward_kinematics(geometry, [5.0, p2, 3.0], grid_size=GRID):
            pts.append((p2, p.theta))
    pts = np.array(pts)

    starts = sc.solve_forward_kinematics(geometry, sc.BASE_STRUTS)
    step, p2_top = 0.0005, 9.8
    summary, switch_paths = [], []
    for k, s in enumerate(starts, start=1):
        # Continuation loop: move p2 in one direction; whenever the tracked
        # pose vanishes, hop onto the partner root it merged with and reverse.
        # Stop once we are back at p2 = 5.
        full, theta0, p2_now, direction, singular = [], s.theta, 5.0, +1, []
        for _ in range(8):
            if direction > 0:
                values = np.arange(p2_now, p2_top, step)
            else:
                values = np.append(np.arange(p2_now, 5.0, -step), 5.0)
            leg = track_branch(theta0, values)
            full += leg
            p2_now, *last = leg[-1]
            if direction < 0 and np.isclose(p2_now, 5.0):
                break  # back home
            singular.append(p2_now)
            partner = partner_root(np.array([5.0, p2_now, 3.0]), last[2])
            if partner is None:
                break
            theta0, direction = partner, -direction
        end = full[-1]
        end_mode = 1 + int(np.argmin([pose_distance(q.as_tuple, end[1:]) for q in starts]))
        end_err = sc.verify_pose(geometry, end[1:], sc.BASE_STRUTS)["max_absolute_error"]
        jumps = np.max([pose_distance(a[1:], b[1:]) for a, b in zip(full[:-1], full[1:])])
        summary.append({"start_mode": k, "start_theta": f"{s.theta:.4f}",
                        "singular_p2_values": " ".join(f"{v:.4f}" for v in singular),
                        "largest_step_in_pose": f"{jumps:.2e}",
                        "returned_to_p2_5": bool(np.isclose(end[0], 5.0)),
                        "end_mode": end_mode, "end_theta": f"{end[3]:.4f}",
                        "end_strut_error": f"{end_err:.2e}"})
        switch_paths.append(np.array(full))

    sc.write_csv(OUT / "B_mode_switch_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=1, color="lightgray", label="all poses")
    for k, path in enumerate(switch_paths, start=1):
        ax.plot(path[:, 0], path[:, 3], lw=1.5,
                label=f"start mode {k} → ends in mode {summary[k-1]['end_mode']}")
    ax.axvline(5.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("$p_2$")
    ax.set_ylabel(r"$\theta$ (rad)")
    ax.set_title("Sweep 4B: solution branches and mode-switching paths\n"
                 "(lengthen $p_2$ to a singularity, then shorten back to 5)")
    ax.legend(fontsize=7, markerscale=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "B_branch_diagram.png", dpi=150)
    plt.close(fig)

    # Animate the first path that ends in a different mode
    chosen = next((i for i, r in enumerate(summary) if r["end_mode"] != r["start_mode"]), 0)
    path = switch_paths[chosen]
    sc.write_csv(OUT / "B_mode_switch_path.csv",
                 [{"p2": f"{r[0]:.4f}", "x": f"{r[1]:.6f}", "y": f"{r[2]:.6f}",
                   "theta": f"{r[3]:.6f}"} for r in path])
    idx = np.linspace(0, len(path) - 1, 160).astype(int)
    animate(path[idx, 1:], [f"p2 = {path[i, 0]:.3f}" for i in idx],
            OUT / "B_mode_switch.gif",
            f"Sweep 4B: mode {summary[chosen]['start_mode']} → "
            f"mode {summary[chosen]['end_mode']} with struts back at (5, 5, 3)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    sc.plot_platform_pose(geometry, path[0, 1:], ax=axes[0], expected_lengths=sc.BASE_STRUTS,
                          title=f"start: mode {summary[chosen]['start_mode']}")
    sc.plot_platform_pose(geometry, path[-1, 1:], ax=axes[1], expected_lengths=sc.BASE_STRUTS,
                          title=f"end: mode {summary[chosen]['end_mode']}")
    fig.suptitle("Same strut lengths (5, 5, 3), different assembly mode")
    fig.tight_layout()
    fig.savefig(OUT / "B_mode_switch_start_end.png", dpi=150)
    plt.close(fig)
    return summary


def animate(poses, labels, path, title, trail=False):
    corners = np.array([geometry.vertices(*q) for q in poses])
    anchors = geometry.anchors
    allpts = np.vstack([corners.reshape(-1, 2), anchors])
    lo, hi = allpts.min(axis=0) - 0.7, allpts.max(axis=0) + 0.7

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)
    ax.plot(*anchors.T, "ks", ms=7)
    tri, = ax.plot([], [], "-", color="tab:red", lw=2)
    struts = [ax.plot([], [], "-", color="tab:blue", lw=1.5)[0] for _ in range(3)]
    trails = [ax.plot([], [], "-", lw=0.8, alpha=0.6)[0] for _ in range(3)] if trail else []
    text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

    def update(k):
        c = corners[k]
        tri.set_data(*np.vstack([c, c[:1]]).T)
        for i in range(3):
            struts[i].set_data([anchors[i, 0], c[i, 0]], [anchors[i, 1], c[i, 1]])
        for i, line in enumerate(trails):
            line.set_data(corners[:k + 1, i, 0], corners[:k + 1, i, 1])
        text.set_text(labels[k])
        return [tri, *struts, *trails, text]

    anim = FuncAnimation(fig, update, frames=len(corners), blit=True)
    anim.save(path, writer=PillowWriter(fps=20))
    plt.close(fig)


def main():
    import sys
    if "--skip-a" not in sys.argv:
        print("Part A:", part_a())
    for row in part_b():
        print("Part B:", row)


if __name__ == "__main__":
    main()
