# ME 360 Development Plan: Planar Stewart Platform

## Project objective

Develop and validate a Python solver for the forward kinematics of a planar Stewart platform, then use the solver to investigate how strut lengths, anchor geometry, platform shape, and simultaneous strut actuation affect the number and motion of possible platform poses.

The completed project should include:

- A reusable forward-kinematics solver.
- Inverse-kinematics verification for every computed pose.
- Comparisons between bisection and Brent root-finding methods.
- Four parameter sweeps, matching the project proposal.
- Clear plots, platform diagrams, and an animation.
- An IEEE-style final report and reproducible source code.

## 1. Shared solver development

Implement the core functionality under `src/me_360_project/` using NumPy, SciPy, Matplotlib, and pytest.

### Geometry and inverse kinematics

Create functions that:

1. Store the fixed anchor points `(0, 0)`, `(x1, 0)`, and `(x2, y2)`.
2. Compute the three moving-platform vertices from `(x, y, theta)`, `L2`, `L3`, and `gamma`.
3. Calculate the three strut lengths for a given platform pose.
4. Validate triangle dimensions and use radians consistently.

### Forward kinematics

Implement Sauer’s reduction to a scalar equation in `theta`:

```text
f(theta) = N1(theta)^2 + N2(theta)^2 - p1^2 D(theta)^2
x = N1 / D
y = N2 / D
```

The solver must:

- Search `theta` over `[-pi, pi]`.
- Detect sign-changing root brackets.
- Handle roots that lie directly on grid points.
- Avoid counting `-pi` and `pi` as separate equivalent roots.
- Refine roots with custom bisection and SciPy’s `brentq`.
- Recover `(x, y)` from every valid root.
- Detect or flag cases where `D` is too close to zero.
- Return diagnostics such as residuals, iteration counts, and function evaluations.

### Verification and plotting

For every recovered pose, recompute all three strut lengths and compare them with the input lengths. Report the maximum absolute error.

Create a reusable plotting function that draws:

- Fixed anchors.
- Platform vertices and sides.
- Struts.
- Equal-scale x/y axes.
- Optional pose labels and titles.

## 2. Solver acceptance tests

Before running the parameter sweeps, reproduce the examples in `Context/Stewart Platform Problem.pdf`.

### Figure 1.15 test

- Anchors: `(0, 0)`, `(4, 0)`, `(0, 4)`.
- Platform: `L2 = L3 = sqrt(2)`, `gamma = pi/2`.
- Struts: `p1 = p2 = p3 = sqrt(5)`.
- Expected orientations include `theta = -pi/4` and `theta = pi/4`.

Reproduce both platform drawings and verify the strut lengths.

### Four-pose test

- Anchors: `(0, 0)`, `(5, 0)`, `(0, 6)`.
- Platform: `L1 = L3 = 3`, `L2 = 3sqrt(2)`, `gamma = pi/4`.
- Struts: `(p1, p2, p3) = (5, 5, 3)`.

Recover four distinct poses and produce a table of `theta`, `x`, `y`, and maximum strut-length error.

### Six-pose test

Use the same geometry as the four-pose test, but set `p2 = 7`. Recover six distinct poses.

### Root-method comparison

For identical brackets and tolerances, compare bisection and Brent by recording:

- Number of iterations.
- Function evaluations.
- Final angle error or residual.
- Runtime, if practical.

Repeat with a finer scan grid to verify that the root count is stable. Document the possibility that sign-change scanning can miss double roots.

## 3. Parameter sweeps

### Sweep 1: Strut length `p2` — Brytnee

Hold the Sauer four-pose geometry fixed and vary `p2` continuously.

Tasks:

- Count the number of valid poses for each value.
- Identify intervals with 0, 2, 4, and 6 poses.
- Refine every transition between intervals.
- Investigate boundaries for double roots or merging assembly modes.
- Check near-zero extrema of `f(theta)` so roots that do not change sign are not missed.

Deliverables:

- Pose-count-versus-`p2` plot.
- Table of transition values and pose-count intervals.
- Representative platform diagrams for each pose-count regime.

### Sweep 2: Anchor positions — Kate

Vary `x1` and `(x2, y2)` while keeping platform geometry and strut lengths fixed.

Tasks:

- Compare how anchor geometry changes `f(theta)`.
- Count the number of solutions for each configuration.
- Plot representative platform poses.
- Identify configurations that produce more or fewer assembly modes.

Deliverables:

- Anchor-parameter pose-count map.
- Representative `f(theta)` plots.
- Platform diagrams and an explanation of the geometric effects.

### Sweep 3: Platform size and shape — Taylor

Study platform scaling and shape independently.

Tasks:

- Uniformly scale the platform while holding `gamma` fixed.
- Vary `gamma` while holding selected side lengths fixed.
- Optionally vary valid side-length combinations while maintaining triangle inequalities.
- Determine how these changes affect pose count and reachability.
- Clearly define strut travel limits and the sampling range used for workspace plots.

Deliverables:

- Pose count versus scale and `gamma`.
- Reachable/unreachable workspace plots.
- Discussion of design changes versus actual degrees of freedom.

### Sweep 4: Simultaneous strut actuation — Alex

Prescribe a continuous platform motion, such as a translation at fixed orientation or a rotation at a selected angular velocity.

Tasks:

- Generate strut-length histories using inverse kinematics.
- Recover poses using the forward solver at each time step.
- Track the assembly mode continuously using the previous pose as the next-pose reference.
- Track position and wrapped orientation together to avoid accidental mode switching.
- Measure deviation from the prescribed motion.
- Identify behavior near singular or assembly-mode-transition configurations.

Deliverables:

- Strut lengths versus time.
- Trajectories of all three platform corners.
- Pose-error plot.
- Matplotlib animation.
- Discussion of whether the actuation path can switch assembly modes continuously.

## 4. Report development

Write the final report in IEEE format with the following sections:

1. **Introduction and Motivation** — Explain Stewart platforms, forward kinematics, multiple assembly modes, and engineering relevance.
2. **Method** — Present the geometry, scalar root equation, bracketing procedure, bisection, Brent’s method, inverse verification, and numerical tolerances.
3. **Results** — Report solver validation, convergence comparison, and all four parameter sweeps.
4. **Discussion** — Evaluate accuracy, performance, robustness, missed double roots, grid resolution, singular configurations, and possible improvements.
5. **Conclusion** — Summarize actual findings and describe future extensions to a spatial six-degree-of-freedom Stewart platform.
6. **Acknowledgement** — Document each group member’s actual contribution.
7. **References** — Include valid outside sources, including Sauer, Stewart, Merlet, Gosselin and Angeles, and the SciPy documentation where appropriate.

Do not leave planned results in the final report. Replace all placeholders with measured values, figures, and conclusions. Follow the course AI-use and citation requirements.

## 5. Schedule and milestones

- **Initial implementation:** Build the geometry, inverse kinematics, forward solver, bisection method, Brent wrapper, and shared plotting function.
- **Validation milestone:** Reproduce Figure 1.15, the four-pose case, and the six-pose case. Add automated tests and error tables.
- **Draft milestone:** Complete the method section, insert preliminary validation figures, and document the experimental setup for all four sweeps.
- **Experiment milestone:** Run all sweeps, save raw data and parameters, refine uncertain transition values, and generate final plots and animation.
- **Interpretation milestone:** Compare numerical methods and connect the results to design, control, reachability, and singularity implications.
- **Final submission milestone:** Regenerate all figures from documented commands, complete the IEEE report, clean and comment the code, and submit the report PDF, source code, and animation.

## 6. Definition of done

The project is complete when:

- The solver reproduces the textbook examples with small inverse-kinematics errors.
- Bisection and Brent results are quantitatively compared.
- All four proposed parameter sweeps contain reproducible data and labeled figures.
- Root-count transitions and double-root limitations are discussed.
- The actuation experiment includes trajectories and an animation.
- The report contains actual numerical findings rather than placeholders.
- The source code is organized, tested, documented, and runnable from the repository.
- The report and code satisfy the course formatting, citation, and AI-use requirements.
