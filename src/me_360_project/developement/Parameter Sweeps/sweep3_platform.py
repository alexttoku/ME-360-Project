# Sweep 3: platform size and shape, and the reachable workspace
# Anchors and struts stay at Sauer's values: (0,0), (5,0), (0,6) and p = (5, 5, 3).
#   Part A: make the whole triangle bigger or smaller (multiply L1, L2, L3 by s)
#   Part B: change the angle gamma (keep L2, L3; L1 from the law of cosines)
#   Part C: workspace. Real struts can only extend between P_MIN and P_MAX.
#           A pose is reachable if all three strut lengths are inside those limits.

import os
import numpy as np
import matplotlib.pyplot as plt
from stewart import base_geometry, find_poses, strut_lengths, plot_all_poses

os.makedirs("output/sweep3", exist_ok=True)
p = [5, 5, 3]
base = base_geometry()


def scaled_geometry(s):
    geo = base_geometry()
    geo['L1'], geo['L2'], geo['L3'] = s*base['L1'], s*base['L2'], s*base['L3']
    return geo


def gamma_geometry(gamma):
    geo = base_geometry()
    geo['gamma'] = gamma
    geo['L1'] = np.sqrt(geo['L2']**2 + geo['L3']**2 - 2*geo['L2']*geo['L3']*np.cos(gamma))
    return geo


def sweep(values, make_geo, name, xlabel, baseline):
    counts = np.array([len(find_poses(p, make_geo(v))) for v in values])
    np.savetxt(f"output/sweep3/pose_count_vs_{name}.csv", np.column_stack([values, counts]),
               delimiter=",", header=f"{name},pose_count", comments="", fmt=["%.4f", "%d"])
    print(f"Transitions in {name}:")
    for i in range(1, len(values)):
        if counts[i] != counts[i-1]:
            a, b = values[i-1], values[i]
            while b - a > 1e-6:                 # bisection on the parameter
                c = (a + b)/2
                if len(find_poses(p, make_geo(c))) == counts[i-1]:
                    a = c
                else:
                    b = c
            print(f"  {name} = {(a+b)/2:.6f}: {counts[i-1]} -> {counts[i]} poses")

    fig = plt.figure(figsize=(7, 4))
    plt.plot(values, counts, linewidth=2)
    plt.axvline(baseline, color='r', linestyle=':', label="Sauer's value")
    plt.xlabel(xlabel)
    plt.ylabel("number of poses")
    plt.yticks([0, 2, 4, 6])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"output/sweep3/pose_count_vs_{name}.png", dpi=150)
    plt.close(fig)


# Part A and Part B
sweep(np.arange(0.2, 2.0001, 0.005), scaled_geometry, "scale", "platform scale factor s", 1.0)
sweep(np.linspace(0.05, np.pi - 0.05, 400), gamma_geometry, "gamma", "gamma (rad)", np.pi/4)

for s in [0.6, 1.0, 1.6]:
    geo = scaled_geometry(s)
    poses = find_poses(p, geo)
    plot_all_poses(poses, geo, p, f"output/sweep3/poses_scale_{s}.png",
                   f"scale {s}: {len(poses)} poses")

# ---------------------------------------------------------------------------
# Part C: workspace with strut limits
# ---------------------------------------------------------------------------
P_MIN, P_MAX = 2.0, 8.0      # assumed strut stroke
x = np.arange(-9, 14.0001, 0.1)
y = np.arange(-9, 15.0001, 0.1)
X, Y = np.meshgrid(x, y)
cell_area = 0.1*0.1


def reachable(geo, theta):
    lengths = strut_lengths(X, Y, theta, geo)    # works on the whole grid at once
    return np.all((lengths >= P_MIN) & (lengths <= P_MAX), axis=0)


# (1) reachable positions of corner 1 at a few fixed angles
thetas = [0, np.pi/4, np.pi/2, -np.pi/2]
scales = [0.5, 1.0, 1.5]
fig, axes = plt.subplots(3, 4, figsize=(13, 10), sharex=True, sharey=True)
for r in range(3):
    geo = scaled_geometry(scales[r])
    for c in range(4):
        ax = axes[r, c]
        ax.contourf(X, Y, reachable(geo, thetas[c]).astype(float), levels=[0.5, 1.5], colors=['C0'])
        ax.plot([0, geo['x1'], geo['x2']], [0, 0, geo['y2']], 'ks', markersize=4)
        ax.set_aspect('equal')
        ax.set_title(f"scale {scales[r]}, theta = {thetas[c]:.2f}", fontsize=9)
fig.suptitle(f"Where corner 1 can be at a fixed angle (strut limits {P_MIN} to {P_MAX})")
fig.tight_layout()
fig.savefig("output/sweep3/workspace_fixed_theta.png", dpi=150)
plt.close(fig)

# (2) for each (x, y): how many angles (out of 72, every 5 degrees) are reachable
theta_list = np.linspace(-np.pi, np.pi, 72, endpoint=False)
print("Workspace area (scale, any angle, every angle, theta = 0):")
rows = []
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for s in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
    geo = scaled_geometry(s)
    n_reach = np.zeros(X.shape)
    for th in theta_list:
        n_reach += reachable(geo, th)
    any_angle = np.sum(n_reach > 0)*cell_area
    every_angle = np.sum(n_reach == 72)*cell_area
    at_zero = np.sum(reachable(geo, 0.0))*cell_area
    rows.append([s, any_angle, every_angle, at_zero])
    print(f"  {s:4.2f}  {any_angle:7.2f}  {every_angle:6.2f}  {at_zero:6.2f}")
    if s in scales:
        ax = axes[scales.index(s)]
        im = ax.pcolormesh(X, Y, np.where(n_reach > 0, n_reach*5, np.nan),
                           cmap='viridis', vmin=0, vmax=360, shading='auto')
        ax.plot([0, geo['x1'], geo['x2']], [0, 0, geo['y2']], 'rs', markersize=4)
        ax.set_aspect('equal')
        ax.set_title(f"scale {s}")
fig.colorbar(im, ax=axes, label="reachable range of theta (degrees)")
fig.suptitle("White = corner 1 can never be placed there")
fig.savefig("output/sweep3/workspace_orientation_range.png", dpi=150, bbox_inches='tight')
plt.close(fig)
np.savetxt("output/sweep3/workspace_area_vs_scale.csv", np.array(rows), delimiter=",",
           header="scale,area_any_theta,area_every_theta,area_theta_0", comments="", fmt="%.2f")
