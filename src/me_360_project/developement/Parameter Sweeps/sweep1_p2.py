# Sweep 1: change strut length p2
# Keep Sauer's geometry and p1 = 5, p3 = 3. Count the poses for each p2,
# then use bisection on p2 to find exactly where the number of poses changes.

import os
import numpy as np
import matplotlib.pyplot as plt
from stewart import base_geometry, f, find_poses, plot_all_poses

os.makedirs("output/sweep1", exist_ok=True)
geo = base_geometry()
p1, p3 = 5, 3


def count(p2):
    return len(find_poses([p1, p2, p3], geo))


# 1) count poses for p2 from 0.5 to 12
p2_values = np.arange(0.5, 12.0001, 0.01)
counts = np.array([count(p2) for p2 in p2_values])
np.savetxt("output/sweep1/pose_count_vs_p2.csv", np.column_stack([p2_values, counts]),
           delimiter=",", header="p2,pose_count", comments="", fmt=["%.2f", "%d"])

# 2) find each transition with bisection on p2
#    (same idea as the bisection method, but the "sign change" is a change in pose count)
transitions = []
for i in range(1, len(p2_values)):
    if counts[i] != counts[i-1]:
        a, b = p2_values[i-1], p2_values[i]
        while b - a > 1e-6:
            c = (a + b)/2
            if count(c) == counts[i-1]:
                a = c
            else:
                b = c
        transitions.append([(a + b)/2, counts[i-1], counts[i]])

print("Transitions (p2, poses before, poses after):")
for t in transitions:
    print(f"  p2 = {t[0]:.6f}: {t[1]} -> {t[2]} poses")
np.savetxt("output/sweep1/transitions.csv", np.array(transitions), delimiter=",",
           header="p2,poses_before,poses_after", comments="", fmt=["%.6f", "%d", "%d"])

# 3) plots
fig = plt.figure(figsize=(7, 4))
plt.plot(p2_values, counts, linewidth=2)
for t in transitions:
    plt.axvline(t[0], color='gray', linestyle='--', linewidth=0.7)
plt.xlabel("strut length p2")
plt.ylabel("number of poses")
plt.yticks([0, 2, 4, 6])
plt.title("Sweep 1: number of poses vs p2  (p1 = 5, p3 = 3)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep1/pose_count_vs_p2.png", dpi=150)
plt.close(fig)

# f(theta) and the poses for one p2 value with 2, 4 and 6 poses
theta = np.linspace(-np.pi, np.pi, 2000)
fig = plt.figure(figsize=(7, 4))
for p2 in [4.3, 6.0, 7.0]:
    plt.plot(theta, f(theta, [p1, p2, p3], geo), label=f"p2 = {p2} ({count(p2)} poses)")
    plot_all_poses(find_poses([p1, p2, p3], geo), geo, [p1, p2, p3],
                   f"output/sweep1/poses_p2_{p2}.png", f"p2 = {p2}")
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel("theta (rad)")
plt.ylabel("f(theta)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/sweep1/f_theta_examples.png", dpi=150)
plt.close(fig)
