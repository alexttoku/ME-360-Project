# Parameter Sweeps

Four parameter sweeps for the planar Stewart platform, written in the same style as the ME 360 lectures and homework: plain functions, NumPy arrays, the Lecture 10 bisection method, and Matplotlib plots.

> **AI-use note for the report:** This code was written with Claude (Anthropic) on 2026-11-23. Declare that in the report the way the syllabus requires. The numbers below are measured results; the report text itself still has to be written by the group.

## How to run

Open a terminal in this folder and run:

```bash
python textbook_check.py      # checks the solver on Sauer's examples (1 s)
python sweep1_p2.py           # ~10 s
python sweep2_anchors.py      # ~30 s
python sweep3_platform.py     # ~10 s
python sweep4_actuation.py    # ~2 min
```

You need numpy and matplotlib (`uv run python ...` also works from the repo). Each script prints its results and saves CSVs and figures into `output/sweep1` to `output/sweep4`.

## Files

| File | What it is |
|---|---|
| `stewart.py` | The basic functions: `f(theta)` and `get_xy` from Sauer eq. 1.39-1.40, `strut_lengths` (inverse kinematics), `bisection` (from Lecture 10), `find_poses` (finds every pose), and plotting. |
| `textbook_check.py` | Reproduces Figure 1.15 (θ = ±π/4), the 4-pose case and the 6-pose case. |
| `sweep1_p2.py` | Sweep 1: change strut length p2. |
| `sweep2_anchors.py` | Sweep 2: move the anchors (x1, and anchor 3 at (x2, y2)). |
| `sweep3_platform.py` | Sweep 3: platform size and angle γ, plus the reachable workspace. |
| `sweep4_actuation.py` | Sweep 4: move all struts together, track the pose, switch between poses, animations. |

**How `find_poses` works:** evaluate f(θ) at 20001 points between −π and π, find every sign change, refine each root with bisection (tolerance 1e-12), then compute x = N1/D and y = N2/D. Poses where D ≈ 0 (a singular configuration) are skipped with a warning.

**To find where the number of poses changes,** each sweep uses bisection on the *parameter* (p2, x1, scale or γ): keep the half of the interval where the pose count changes, until the interval is smaller than 1e-6.

**Baseline** (Sauer, Reality Check 1, step 4): anchors (0,0), (5,0), (0,6); L1 = 3, L2 = 3√2, L3 = 3, γ = π/4; struts (5, 5, 3), which gives 4 poses. "Sauer's anchor 3 at (0, 6)" in the plots means the textbook position of the third anchor.

## Results

### Sweep 1: strut length p2 (p1 = 5, p3 = 3)

| p2 range | poses |
|---|---|
| below 3.7105 | 0 |
| 3.7105 – 4.8637 | 2 |
| 4.8637 – 6.9673 | 4 |
| 6.9673 – 7.0223 | 6 |
| 7.0223 – 7.8491 | 4 |
| 7.8491 – 9.2624 | 2 |
| above 9.2624 | 0 |

At each transition two roots of f(θ) merge into a double root and disappear (or appear).

### Sweep 2: anchor positions (struts (5, 5, 3) unless noted)

- **x1:** 2 poses below x1 = 1.004, 4 poses up to 5.186, 2 poses up to 8.692, then 0. There is also a **narrow 6-pose window for x1 ≈ 2.971 to 3.027**, with a singular point exactly at x1 = 3 (D = 0 at θ = 0). Sauer's x1 = 5 is only 0.19 below the 4 → 2 transition.
- **Anchor 3 map, struts (5, 5, 3):** only 0, 2 or 4 poses (x2 from −4 to 6, y2 from 1 to 10, step 0.2).
- **Anchor 3 map, struts (5, 7, 3):** a large 6-pose region (448 of 2346 grid points). Sauer's (0, 6) is on its edge.
- **Odd counts (3 or 5):** these are double roots, where two poses have merged into one. Most are the flat pose at θ = 0 (corner 1 at (−5, 0), with strut 1, strut 2 and side L3 all on the x-axis, since 5 + 5 − 3 = 7). That pose exists whenever anchor 3 is 3 units from corner 3 = (−2, 3), which is the dashed circle on the map.

### Sweep 3: platform size and shape

- **Scale s** (all sides × s): 2 poses below s = 0.300, 4 poses for 0.300 – 0.313 (narrow window), 2 poses up to 0.660, 4 poses up to 1.550, **6 poses for 1.550 – 1.835**, then 4.
- **γ** (L2 = 3√2, L3 = 3 fixed): 2 poses below 0.602 rad (34.5°), 4 poses up to 2.347 rad (134.5°), then 2.
- **Workspace** (assumed strut limits 2 ≤ p ≤ 8), area where corner 1 can be:

| scale | any angle | every angle | θ = 0 |
|---|---|---|---|
| 0.25 | 69.6 | 22.9 | 50.7 |
| 0.5 | 96.0 | 6.4 | 57.8 |
| 1.0 | 149.1 | 0 | 76.1 |
| 1.5 | 185.1 | 0 | 91.9 |
| 1.75 | 188.2 | 0 | 91.5 |
| 2.0 | 179.7 | 0 | 78.3 |

  The most corner 1 could ever cover is the ring 2 ≤ r ≤ 8 around anchor 1 (area 188.5). From a scale of 0.75 up, no position can be reached at every angle. White areas in the workspace plots can never be reached by moving the struts.

### Sweep 4: moving all struts together

- **Part A:** the platform follows a smooth motion for 10 s (201 steps). From only the strut lengths:
  - *Naive tracking* (pick the pose closest to the last one) **jumps to the wrong pose at t = 6.95**, where two poses pass very close to each other. Largest error 0.87.
  - *Predictor tracking* (guess = 2·last − one before) stays on the right pose, with largest error 7e-12.
- **Part B:** start in each of the 4 poses, lengthen p2 until the pose merges with another one, then turn around:
  - Pose 1 ↔ pose 2 through singular points at p2 = 7.022, 6.968, 7.848.
  - Pose 3 ↔ pose 4 through the singular point at p2 = 9.262.
  - Every time, the struts end at exactly (5, 5, 3) but the platform is in a **different pose**. So moving the struts can take the platform to another solution, but only by going through a singular configuration.
- Animations: `A_motion.gif` and `B_mode_switch.gif`.

## Limitations (for the Discussion section)

1. Roots are found from sign changes on a grid, so a double root (f touches zero without crossing) is not detected, and two roots closer together than the grid spacing (3e-4 rad here) can be missed.
2. The sweeps use a finite step in the parameter, so very narrow windows can be skipped. The first version of this code used larger steps and missed the windows near x1 = 3 and scale = 0.3.
3. Poses with D ≈ 0 are skipped, because x and y can't be computed there.
4. The strut limits [2, 8] are an assumption, not from a real actuator.
5. Picking the closest pose can jump to the wrong pose near singular points. The predictor fixes this for our test motion; smaller time steps would also help.
