# Check stewart.py against Sauer's examples before running the sweeps.

import numpy as np
from stewart import base_geometry, find_poses, strut_lengths

# Figure 1.15: roots should be at theta = -pi/4 and +pi/4
geo = {'x1': 4, 'x2': 0, 'y2': 4, 'L1': 2, 'L2': np.sqrt(2), 'L3': np.sqrt(2), 'gamma': np.pi/2}
p = [np.sqrt(5), np.sqrt(5), np.sqrt(5)]
poses = find_poses(p, geo)
print("Figure 1.15 thetas:", [round(float(q[2]), 6) for q in poses], " expected +/-", round(np.pi/4, 6))

# Step 4: four poses.  Step 5: p2 = 7 gives six poses.
geo = base_geometry()
for p in ([5, 5, 3], [5, 7, 3]):
    poses = find_poses(p, geo)
    worst = max(np.max(np.abs(strut_lengths(x, y, th, geo) - np.array(p))) for x, y, th in poses)
    print(f"p = {p}: {len(poses)} poses, largest strut-length error = {worst:.1e}")
