# Sweep 4: move all three struts at the same time
#   Part A: pick a smooth platform motion, get the strut lengths from inverse
#           kinematics, then recover the motion from ONLY the strut lengths using
#           the forward solver. At each time step there are several poses, so we
#           have to pick the right one (track it).
#   Part B: can strut motion move the platform from one pose to a different pose?
#           Lengthen p2 until our pose merges with another one, then shorten p2
#           and follow the other branch back to p2 = 5.
#   Animations are saved as GIFs.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from stewart import base_geometry, find_poses, strut_lengths, corners, plot_platform

os.makedirs("output/sweep4", exist_ok=True)
geo = base_geometry()


def angle_diff(a, b):
    # difference between two angles, wrapped to [-pi, pi]
    return (a - b + np.pi) % (2*np.pi) - np.pi


def distance(pose_a, pose_b):
    # how different two poses are: position difference + angle difference
    return np.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1]) + abs(angle_diff(pose_a[2], pose_b[2]))


def closest(poses, guess):
    best = poses[0]
    for q in poses:
        if distance(q, guess) < distance(best, guess):
            best = q
    return best


def animate(poses, labels, filename, title):
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.gca()

    def draw(k):
        ax.clear()
        plot_platform(poses[k][0], poses[k][1], poses[k][2], geo, ax, title)
        ax.set_xlim(-6, 8)
        ax.set_ylim(-1, 10)
        ax.text(-5.5, 9.2, labels[k])

    anim = FuncAnimation(fig, draw, frames=len(poses))
    anim.save(filename, writer=PillowWriter(fps=20))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Part A: prescribed motion
# ---------------------------------------------------------------------------
start = find_poses([5, 5, 3], geo)[1]          # second of Sauer's four poses
t = np.linspace(0, 10, 201)
w = 2*np.pi/10
x_true = start[0] + 0.8*np.sin(w*t)
y_true = start[1] + 0.5*(1 - np.cos(w*t))
theta_true = start[2] + 0.35*np.sin(w*t)

# strut lengths needed for this motion (inverse kinematics)
p_hist = np.array([strut_lengths(x_true[k], y_true[k], theta_true[k], geo) for k in range(len(t))])

# Method 1 (naive): pick the pose closest to the previous pose
# Method 2 (predictor): guess the next pose by assuming the platform keeps
#                       moving the same way: guess = 2*(last pose) - (pose before)
naive = [start]
predictor = [start]
for k in range(1, len(t)):
    poses = find_poses(p_hist[k], geo)
    naive.append(closest(poses, naive[-1]))
    if k == 1:
        guess = predictor[-1]
    else:
        guess = 2*np.array(predictor[-1]) - np.array(predictor[-2])
    predictor.append(closest(poses, guess))
naive = np.array(naive)
predictor = np.array(predictor)

true = np.column_stack([x_true, y_true, theta_true])
err_naive = np.array([distance(naive[k], true[k]) for k in range(len(t))])
err_pred = np.array([distance(predictor[k], true[k]) for k in range(len(t))])
print(f"Part A: largest error, naive tracking = {err_naive.max():.2e}, "
      f"predictor tracking = {err_pred.max():.2e}")
if err_naive.max() > 1e-6:
    print(f"  naive tracking jumps to the wrong pose at t = {t[np.argmax(err_naive > 1e-6)]:.2f}")

fig = plt.figure(figsize=(7, 4))
for i in range(3):
    plt.plot(t, p_hist[:, i], label=f"p{i+1}")
plt.xlabel("time")
plt.ylabel("strut length")
plt.title("Sweep 4A: strut lengths for the prescribed motion")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep4/A_strut_lengths.png", dpi=150)
plt.close(fig)

fig = plt.figure(figsize=(7, 4))
plt.plot(t, theta_true, 'k--', linewidth=2, label="prescribed theta")
plt.plot(t, naive[:, 2], 'r', label="naive tracking")
plt.plot(t, predictor[:, 2], 'b', linewidth=1, label="predictor tracking")
plt.xlabel("time")
plt.ylabel("theta (rad)")
plt.title("Sweep 4A: angle recovered from strut lengths only")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep4/A_theta_tracking.png", dpi=150)
plt.close(fig)

fig = plt.figure(figsize=(6, 6))
u = np.array([corners(q[0], q[1], q[2], geo)[0] for q in predictor])
v = np.array([corners(q[0], q[1], q[2], geo)[1] for q in predictor])
for i in range(3):
    plt.plot(u[:, i], v[:, i], label=f"corner {i+1}")
plot_platform(predictor[0][0], predictor[0][1], predictor[0][2], geo,
              title="Sweep 4A: paths of the 3 corners (platform at t = 0)")
plt.legend()
plt.tight_layout()
plt.savefig("output/sweep4/A_corner_paths.png", dpi=150)
plt.close(fig)

animate(predictor, [f"t = {tk:.1f}" for tk in t], "output/sweep4/A_motion.gif",
        "Sweep 4A: prescribed motion")

# ---------------------------------------------------------------------------
# Part B: switching between poses
# ---------------------------------------------------------------------------
step = 0.002
starts = find_poses([5, 5, 3], geo)
paths = []
for m in range(len(starts)):
    pose = starts[m]
    p2 = 5.0
    direction = +1                      # +1 = lengthen p2, -1 = shorten p2
    path = [[p2] + list(pose)]
    singular_points = []
    while True:
        new_p2 = p2 + direction*step
        if direction == -1 and new_p2 <= 5.0:
            new_p2 = 5.0
        poses = find_poses([5, new_p2, 3], geo)
        if len(poses) > 0:
            best = closest(poses, pose)
        if len(poses) == 0 or abs(angle_diff(best[2], pose[2])) > 0.2:
            # Our pose vanished: it merged with another pose (a singular point).
            # Switch to the closest OTHER pose at the last p2, and turn around.
            old = find_poses([5, p2, 3], geo)
            others = [q for q in old if distance(q, pose) > 1e-9]
            pose = closest(others, pose)
            singular_points.append(round(p2, 3))
            direction = -direction
            continue
        pose, p2 = best, new_p2
        path.append([p2] + list(pose))
        if direction == -1 and p2 == 5.0:
            break
    end_mode = 1 + int(np.argmin([distance(q, pose) for q in starts]))
    print(f"Part B: start in pose {m+1}, pass singular points at p2 = {singular_points}, "
          f"end in pose {end_mode} with struts (5, 5, 3)")
    paths.append(np.array(path))

# all poses for a range of p2 (gray), with the switching paths on top
fig = plt.figure(figsize=(7.5, 4.5))
for p2 in np.arange(3.0, 9.8, 0.01):
    for q in find_poses([5, p2, 3], geo):
        plt.plot(p2, q[2], '.', color='lightgray', markersize=2)
for m in range(len(paths)):
    # poses 1 and 2 travel the same loop (in opposite directions), as do 3 and 4,
    # so their lines lie on top of each other
    plt.plot(paths[m][:, 0], paths[m][:, 3], linewidth=1.5,
             label=f"start in pose {m+1} (loop shared with pose {[2, 1, 4, 3][m]})")
plt.axvline(5, color='k', linestyle=':')
plt.xlabel("p2")
plt.ylabel("theta (rad)")
plt.title("Sweep 4B: every pose vs p2 (gray) and the switching paths")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep4/B_branches.png", dpi=150)
plt.close(fig)

# animation of the first switch (pose 1 -> other pose)
path = paths[0]
frames = np.linspace(0, len(path) - 1, 160).astype(int)
animate([path[k, 1:] for k in frames], [f"p2 = {path[k, 0]:.3f}" for k in frames],
        "output/sweep4/B_mode_switch.gif", "Sweep 4B: same struts (5, 5, 3), different pose")
