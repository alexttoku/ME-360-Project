# Sweep 2: move the fixed anchors
# Keep the platform triangle and the strut lengths the same, and move the anchors.
#   Part A: slide anchor 2 along the x-axis (change x1)
#   Part B: move anchor 3 around a grid of (x2, y2) and make a map of the pose count
#   Part C: compare f(theta) and the poses for a few anchor layouts

import os
import numpy as np
import matplotlib.pyplot as plt
from stewart import base_geometry, f, find_poses, plot_all_poses

os.makedirs("output/sweep2", exist_ok=True)
p = [5, 5, 3]


def count(p, **changes):
    geo = base_geometry()
    geo.update(changes)          # e.g. count(p, x1=7) changes only x1
    return len(find_poses(p, geo))


# ---------------------------------------------------------------------------
# Part A: x1 from 0.25 to 10
# ---------------------------------------------------------------------------
x1_values = np.arange(0.25, 10.0001, 0.01)
counts = np.array([count(p, x1=x1) for x1 in x1_values])
np.savetxt("output/sweep2/pose_count_vs_x1.csv", np.column_stack([x1_values, counts]),
           delimiter=",", header="x1,pose_count", comments="", fmt=["%.2f", "%d"])

print("Part A transitions in x1:")
for i in range(1, len(x1_values)):
    if counts[i] != counts[i-1]:
        a, b = x1_values[i-1], x1_values[i]
        while b - a > 1e-6:                      # bisection on x1
            c = (a + b)/2
            if count(p, x1=c) == counts[i-1]:
                a = c
            else:
                b = c
        print(f"  x1 = {(a+b)/2:.6f}: {counts[i-1]} -> {counts[i]} poses")

fig = plt.figure(figsize=(7, 4))
plt.plot(x1_values, counts, linewidth=2)
plt.axvline(5, color='r', linestyle=':', label="Sauer's x1 = 5")
plt.xlabel("anchor 2 position x1")
plt.ylabel("number of poses")
plt.yticks([0, 2, 4, 6])
plt.title("Sweep 2A: number of poses vs x1  (struts 5, 5, 3)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep2/pose_count_vs_x1.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Part B: map of pose count vs anchor 3 position (x2, y2)
# Done twice: Sauer's 4-pose struts (5, 5, 3) and 6-pose struts (5, 7, 3)
# ---------------------------------------------------------------------------
x2_values = np.arange(-4, 6.0001, 0.2)
y2_values = np.arange(1, 10.0001, 0.2)

for struts in ([5, 5, 3], [5, 7, 3]):
    name = f"{struts[0]}_{struts[1]}_{struts[2]}"
    grid = np.zeros((len(y2_values), len(x2_values)))
    for j in range(len(y2_values)):
        for i in range(len(x2_values)):
            grid[j, i] = count(struts, x2=x2_values[i], y2=y2_values[j])
    np.savetxt(f"output/sweep2/pose_count_map_{name}.csv", grid, delimiter=",", fmt="%d")

    values, number = np.unique(grid, return_counts=True)
    print(f"Part B, struts {struts}: grid points with each pose count:",
          dict(zip(values.astype(int).tolist(), number.tolist())))

    fig = plt.figure(figsize=(6.5, 5.5))
    plt.pcolormesh(x2_values, y2_values, grid, cmap='Blues', vmin=-1, vmax=6, shading='nearest')
    plt.colorbar(label="number of poses", ticks=[0, 2, 4, 6])
    plt.plot(0, 6, 'r*', markersize=14, label="Sauer's anchor 3 at (0, 6)")

    # Odd counts (3 or 5) are double roots: two poses have merged into one.
    # For struts (5, 7, 3) these happen when the platform lies flat on the x-axis:
    # corner 1 at (-5, 0), corner 2 at (-2, 0) (5 + 5 - 3 = 7). That flat pose is
    # possible whenever anchor 3 is 3 units from corner 3 = (-2, 3): a circle.
    odd = np.argwhere(grid % 2 == 1)
    if len(odd) > 0:
        plt.plot(x2_values[odd[:, 1]], y2_values[odd[:, 0]], 'rx', markersize=8,
                 markeredgewidth=2, label="odd count = double root")
    if struts == [5, 7, 3]:
        a = np.linspace(0, 2*np.pi, 300)
        plt.plot(-2 + 3*np.cos(a), 3 + 3*np.sin(a), 'r--', linewidth=1,
                 label="flat singular pose possible")
        plt.xlim(x2_values[0] - 0.1, x2_values[-1] + 0.1)
        plt.ylim(y2_values[0] - 0.1, y2_values[-1] + 0.1)
    plt.xlabel("anchor 3: x2")
    plt.ylabel("anchor 3: y2")
    plt.title(f"Sweep 2B: number of poses vs anchor 3 position\n"
              f"(anchors 1, 2 at (0, 0), (5, 0); struts {struts})")
    plt.gca().set_aspect('equal')
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(f"output/sweep2/pose_count_map_{name}.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Part C: f(theta) and poses for a few anchor layouts
# ---------------------------------------------------------------------------
layouts = [("baseline", {}, [5, 5, 3]),
           ("narrow base x1 = 2", {'x1': 2}, [5, 5, 3]),
           ("wide base x1 = 8", {'x1': 8}, [5, 5, 3]),
           ("baseline, struts 5 7 3", {}, [5, 7, 3])]

theta = np.linspace(-np.pi, np.pi, 2000)
fig = plt.figure(figsize=(7, 4))
for k, (label, changes, struts) in enumerate(layouts):
    geo = base_geometry()
    geo.update(changes)
    poses = find_poses(struts, geo)
    plt.plot(theta, f(theta, struts, geo), label=f"{label} ({len(poses)} poses)")
    plot_all_poses(poses, geo, struts, f"output/sweep2/poses_layout_{k+1}.png",
                   f"{label}: {len(poses)} poses")
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel("theta (rad)")
plt.ylabel("f(theta)")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep2/f_theta_layouts.png", dpi=150)
plt.close(fig)
