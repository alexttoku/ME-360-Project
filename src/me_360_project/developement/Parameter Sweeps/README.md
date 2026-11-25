# Parameter Sweeps

Four parameter sweeps for the planar Stewart platform. All four use the shared solver in `../Functions/` (Alex's code) and do not change it.

> **AI-use note for the report:** These scripts and this README were written with Claude (Anthropic) on 2026-11-23. Declare that in the report the way the syllabus requires. The numbers below are measured results to *use*; the report text itself still has to be written by the group.

## How to run

From the repository root, with the project's Python environment (`uv sync` installs numpy, scipy, matplotlib):

```bash
cd "src/me_360_project/developement/Parameter Sweeps"
uv run python sweep1_strut_length.py      # ~40 s
uv run python sweep2_anchor_positions.py  # ~5 min
uv run python sweep3_platform_shape.py    # ~30 s
uv run python sweep4_actuation.py         # ~3 min
```

(Plain `python` works too if numpy/scipy/matplotlib/pillow are installed.) Every script writes its CSVs and figures into `output/<sweep name>/` and prints a summary.

## Files

| File | What it does |
|---|---|
| `sweep_common.py` | Loads the solver, defines the baseline (Sauer's four-pose example), and has shared tools: pose counting, a double-root check, bisection on a sweep parameter to locate transitions, and plotting helpers. |
| `sweep1_strut_length.py` | Sweep 1: vary strut length p2. |
| `sweep2_anchor_positions.py` | Sweep 2: move the fixed anchors (x1, and anchor 3 at (x2, y2)). |
| `sweep3_platform_shape.py` | Sweep 3: platform size and angle γ, plus the reachable workspace with strut travel limits. |
| `sweep4_actuation.py` | Sweep 4: move all struts together, track the pose, switch assembly modes, and make the animations. |

**Baseline** (Sauer, Reality Check 1, step 4): anchors (0,0), (5,0), (0,6); L1 = 3, L2 = 3√2, L3 = 3, γ = π/4; struts (p1, p2, p3) = (5, 5, 3), which gives 4 poses. "Sauer baseline (0, 6)" in the plots means anchor 3 at its textbook position x2 = 0, y2 = 6.

## Results (measured)

### Sweep 1: strut length p2 (p1 = 5, p3 = 3)

| p2 interval | poses |
|---|---|
| p2 < 3.7105 | 0 |
| 3.7105 – 4.8637 | 2 |
| 4.8637 – 6.9673 | 4 |
| 6.9673 – 7.0223 | 6 |
| 7.0223 – 7.8491 | 4 |
| 7.8491 – 9.2624 | 2 |
| p2 > 9.2624 | 0 |

- Transitions are located by bisection on p2 to 1e-6 (`transitions.csv`). Each one is a double root: just past it, the smallest non-crossing minimum of |f| is ~1e-7 of max|f|, compared with ~1e-3 at a distance of 0.05.
- Pose counts are the same with 2001 and 10001 θ grid points across the sweep.
- **Grid limitation found:** near p2 = 7.849 the two merging roots are only ~0.002 rad apart, less than the default grid spacing (0.0031 rad), so the coarse scan loses them early (transition 7.84905 instead of 7.84909). Transitions are therefore refined with a 20001-point grid.

### Sweep 2: anchor positions (platform and struts fixed)

- **x1 (anchor 2 along the x-axis), struts (5,5,3), step 0.01:** 2 poses for x1 < 1.004, 4 for 1.004 – 2.971, **6 for 2.971 – 3.027** (except 4 for 2.99986 – 3.00014, around the singular point x1 = 3), 4 for 3.027 – 5.186, 2 for 5.186 – 8.692, 0 beyond 8.692. The textbook x1 = 5 sits just 0.19 below a 4→2 transition. The earlier 0.05 step jumped over the 6-pose window entirely.
- **Anchor 3 map, struts (5,5,3):** 0, 2 or 4 poses; never 6 anywhere on the grid (x2 from −4 to 6, y2 from 1 to 10, step 0.2).
- **Anchor 3 map, struts (5,7,3)** (Sauer's six-pose case): a large six-pose region (x2 from −3.8 to 1.2, y2 from 1.0 to 6.0). The textbook anchor (0, 6) is on the **edge** of that region.
- **Odd counts (3 or 5) at 5 grid points** are real double roots, not bugs. They are a flat singular pose at θ = 0: vertex 1 at (−5, 0), with strut 1, strut 2 and side L3 all on the x-axis (5 + 5 − 3 = 7 = p2). That pose exists whenever anchor 3 is 3 units from vertex 3 = (−2, 3), i.e. on the dashed circle in the map, and all 5 odd points lie on it.
- Cells on region boundaries are rechecked with a 20001-point θ grid. 3 of 447 cells changed for (5,7,3), which shows the default grid can miss close root pairs.
- All recovered poses reproduce the strut lengths to < 3e-13.

### Sweep 3: platform size and shape (anchors and struts fixed)

- **Uniform scale s** (γ fixed, step 0.005): 2 poses for s < 0.300, **4 for 0.300 – 0.313** (narrow), 2 for 0.313 – 0.660, 4 for 0.660 – 1.550, **6 for 1.550 – 1.835**, 4 above 1.835 (up to s = 2). The earlier 0.02 step jumped over the 0.300 – 0.313 window.
- **γ** (L2 = 3√2, L3 = 3, L1 from the law of cosines): 2 poses for γ < 34.5°, 4 for 34.5° – 134.5°, 2 above.
- **Workspace** with strut limits 2 ≤ p ≤ 8 (assumed actuator stroke). Area of the positions vertex 1 can reach:

| scale | any θ | every θ | θ = 0 |
|---|---|---|---|
| 0.25 | 69.6 | 22.9 | 50.7 |
| 0.5 | 96.0 | 6.4 | 57.8 |
| 1.0 | 149.1 | 0 | 76.1 |
| 1.5 | 185.1 | 0 | 91.9 |
| 1.75 | 188.2 | 0 | 91.5 |
| 2.0 | 179.7 | 0 | 78.3 |

  The upper bound is the annulus 2 ≤ r ≤ 8 around anchor 1 (area 188.5), so at scale 1.75 nearly every allowed position is reachable at *some* angle. Above a scale of about 0.75, no position can reach every orientation. White areas in `workspace_orientation_count.png` can never be reached by moving the struts alone.

### Sweep 4: simultaneous actuation

- **Part A, prescribed motion** (translation plus ±0.35 rad rotation over 10 s, 201 steps): the strut histories come from inverse kinematics, then the forward solver recovers the motion.
  - *Naive* tracking (pick the pose nearest the previous one) **switches assembly mode at t = 6.95**, where two modes pass within 0.005 rad of each other. Max error 0.87.
  - *Predictor* tracking (extrapolate from the last two poses) follows the motion with max pose error 2.6e-12 and max strut error 2.4e-12.
- **Part B, assembly-mode switching:** start in each textbook pose, lengthen p2 until the pose merges with another (a singularity), then reverse and continue on the partner branch:
  - Mode 1 ↔ mode 2: passes singular points at p2 = 7.022, 6.968, 7.849, then returns to p2 = 5.
  - Mode 3 ↔ mode 4: passes the singular point at p2 = 9.262.
  - In every case the struts return to exactly (5, 5, 3) but the platform is in a **different** pose (strut error < 1e-14). So strut motion can move the platform from one solution to another, but only by passing through a singular configuration.
- Animations: `A_motion.gif`, `B_mode_switch.gif`.

## Known limitations (for the Discussion section)

1. The solver finds roots by sign changes on a θ grid, so a double root (tangent) or two roots closer than the grid spacing can be missed. This was handled with finer grids near transitions and a separate check for minima of |f|, but the maps use a 0.2 step in anchor position, so very thin regions could still be missed. The 1-D sweeps had the same problem with the parameter step: x1 at 0.05 and scale at 0.02 missed narrow windows that show up at 0.01 and 0.005.
2. Poses where D ≈ 0 are skipped by the solver (x, y can't be recovered from N/D).
3. The workspace strut limits [2, 8] are an assumption, not from a real actuator.
4. Nearest-pose tracking can jump between assembly modes near singularities; the predictor fixes it for the tested motion, and a smaller step size would also help.
