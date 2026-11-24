# ME 360 Final Project - Planar Stewart Platform
# Basic functions used by all four parameter sweeps.
#
# Notation follows Sauer, Reality Check 1:
#   anchors at (0, 0), (x1, 0), (x2, y2)
#   platform triangle with sides L1, L2, L3 and angle gamma
#   strut lengths p1, p2, p3
#   platform pose (x, y, theta): (x, y) is the corner attached to strut 1
#
# The geometry is stored in a dictionary called geo, for example
#   geo = {'x1': 5, 'x2': 0, 'y2': 6, 'L1': 3, 'L2': 3*np.sqrt(2), 'L3': 3, 'gamma': np.pi/4}

import numpy as np
import matplotlib.pyplot as plt


# Sauer's four-pose example (Reality Check 1, step 4). Every sweep starts from this.
def base_geometry():
    return {'x1': 5.0, 'x2': 0.0, 'y2': 6.0,
            'L1': 3.0, 'L2': 3*np.sqrt(2), 'L3': 3.0, 'gamma': np.pi/4}


# ---------------------------------------------------------------------------
# Forward kinematics: f(theta) from Sauer eq. (1.39)-(1.40)
# theta can be a single number or a numpy array (so we can plot f easily).
# ---------------------------------------------------------------------------
def f_terms(theta, p, geo):
    p1, p2, p3 = p
    A2 = geo['L3']*np.cos(theta) - geo['x1']
    B2 = geo['L3']*np.sin(theta)
    A3 = geo['L2']*np.cos(theta + geo['gamma']) - geo['x2']
    B3 = geo['L2']*np.sin(theta + geo['gamma']) - geo['y2']

    N1 = B3*(p2**2 - p1**2 - A2**2 - B2**2) - B2*(p3**2 - p1**2 - A3**2 - B3**2)
    N2 = -A3*(p2**2 - p1**2 - A2**2 - B2**2) + A2*(p3**2 - p1**2 - A3**2 - B3**2)
    D = 2*(A2*B3 - B2*A3)
    return N1, N2, D


def f(theta, p, geo):
    N1, N2, D = f_terms(theta, p, geo)
    return N1**2 + N2**2 - p[0]**2 * D**2


def get_xy(theta, p, geo):
    N1, N2, D = f_terms(theta, p, geo)
    return N1/D, N2/D


# ---------------------------------------------------------------------------
# Inverse kinematics: strut lengths for a given pose (Sauer eq. 1.38)
# Also works when x, y are numpy arrays (used for the workspace maps).
# ---------------------------------------------------------------------------
def corners(x, y, theta, geo):
    u = [x, x + geo['L3']*np.cos(theta), x + geo['L2']*np.cos(theta + geo['gamma'])]
    v = [y, y + geo['L3']*np.sin(theta), y + geo['L2']*np.sin(theta + geo['gamma'])]
    return u, v


def strut_lengths(x, y, theta, geo):
    u, v = corners(x, y, theta, geo)
    p1 = np.sqrt(u[0]**2 + v[0]**2)
    p2 = np.sqrt((u[1] - geo['x1'])**2 + v[1]**2)
    p3 = np.sqrt((u[2] - geo['x2'])**2 + (v[2] - geo['y2'])**2)
    return np.array([p1, p2, p3])


# ---------------------------------------------------------------------------
# Bisection method (same as Lecture 10 / HW3)
# ---------------------------------------------------------------------------
def bisection(func, a, b, tol=1e-12):
    if func(a)*func(b) > 0:
        raise ValueError("Invalid interval for bisection method!")
    while (b - a)/2 > tol:
        c = (a + b)/2
        if func(c) == 0:
            return c
        elif func(a)*func(c) < 0:
            b = c
        else:
            a = c
    return (a + b)/2


# ---------------------------------------------------------------------------
# Find ALL poses for given strut lengths p = [p1, p2, p3]
#   1. evaluate f on n points between -pi and pi
#   2. every sign change brackets a root
#   3. refine each root with bisection, then get x and y
# Returns a list of poses [x, y, theta]. Poses where D = 0 are skipped.
#
# Note: a "double root" (f touches zero without crossing) has no sign change,
# so it can be missed. Two roots closer together than the grid spacing can
# also be missed, which is why we use a fine grid (n = 20001).
# ---------------------------------------------------------------------------
def find_poses(p, geo, n=20001):
    theta = np.linspace(-np.pi, np.pi, n)
    fvals = f(theta, p, geo)
    poses = []
    for i in range(n - 1):
        if fvals[i] == 0:
            root = theta[i]
        elif fvals[i]*fvals[i+1] < 0:
            root = bisection(lambda t: f(t, p, geo), theta[i], theta[i+1])
        else:
            continue
        N1, N2, D = f_terms(root, p, geo)
        if abs(D) < 1e-10:
            # D = 0 is a singular configuration: x = N1/D, y = N2/D cannot be
            # computed, so this pose is skipped.
            print(f"  warning: D is almost 0 at theta = {root:.4f}, pose skipped")
            continue
        poses.append([N1/D, N2/D, root])
    return poses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_platform(x, y, theta, geo, ax=None, title=None):
    if ax is None:
        ax = plt.gca()
    u, v = corners(x, y, theta, geo)
    anchors_x = [0, geo['x1'], geo['x2']]
    anchors_y = [0, 0, geo['y2']]
    ax.plot([u[0], u[1], u[2], u[0]], [v[0], v[1], v[2], v[0]], 'r', linewidth=2)  # platform
    for i in range(3):
        ax.plot([anchors_x[i], u[i]], [anchors_y[i], v[i]], 'b', linewidth=1.5)    # struts
    ax.plot(anchors_x, anchors_y, 'bo')                                              # anchors
    ax.plot(u, v, 'ro')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    if title is not None:
        ax.set_title(title, fontsize=10)


def plot_all_poses(poses, geo, p, filename, title):
    n = len(poses)
    cols = min(n, 3)
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols, 4.2*rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k in range(n):
        x, y, th = poses[k]
        error = np.max(np.abs(strut_lengths(x, y, th, geo) - np.array(p)))
        ax = axes.flat[k]
        ax.set_visible(True)
        plot_platform(x, y, th, geo, ax,
                      f"pose {k+1}: theta = {th:.3f}\nstrut error = {error:.1e}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
