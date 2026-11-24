[alexttoku/ME-360-Project](https://github.com/alexttoku/ME-360-Project)   
**Forward Kinematics of a Planar Stewart Platform**  
ME 360 Final Project Proposal, Fall 2026  
*Group members: Winry Cigna, Brytnee Farley, Kate Kaneshiro, Taylor Todd, Alex Tokunaga*

**Abstract:**

The forward kinematics of a planar Stewart platform, finding the platform pose from three strut lengths, has no closed-form solution, and one set of lengths can give several poses. We reduce the problem to finding the roots of one nonlinear function of the platform angle. We then implement a Python solver that brackets each root by sign changes and refines it with the bisection and Brent methods. We verify poses with inverse kinematics, compare how fast the two methods converge, and report four parameter sweeps over strut length, anchor positions, platform shape, and strut motion. *\[Add one or two sentences with the main numerical findings once available.\]*

**Keywords:** Stewart platform, parallel manipulator, forward kinematics, root finding, bisection method, Brent's method

**I. Introduction and Motivation**

Stewart introduced a six-degree-of-freedom platform moved by variable-length struts as a flight simulator base \[1\]. Parallel manipulators of this kind are now used in motion simulators, machine tools, telescopes, and precision positioning stages \[2\]. They can be stiff and accurate, but their kinematics are harder than those of serial arms: finding strut lengths for a desired pose (inverse kinematics) is direct, while finding the pose for given strut lengths (forward kinematics) is not \[2\], \[3\].

This difficulty matters in practice. Control and calibration need the actual pose, and the existence of several valid poses (assembly modes) means a controller must know which one the platform is in. Singular configurations, where assembly modes merge, also limit control and accuracy \[4\].

We study the planar Stewart platform of Sauer \[3\]: a triangle with side lengths $L_1$, $L_2$, $L_3$ and angle $\gamma$, held by three struts of lengths $p_1$, $p_2$, $p_3$ attached to fixed anchors at $(0, 0)$, $(x_1, 0)$, and $(x_2, y_2)$. Although simple, it shows the main features of the full problem, including multiple assembly modes. Our goals are to (i) build a reliable solver, (ii) compare the bisection and Brent methods on it, and (iii) explore how strut lengths, base geometry, and platform shape change the set of possible poses.

*\[Insert Figure 1 here: diagram of the planar Stewart platform, adapted from Sauer \[3\], showing anchors, struts p₁ to p₃, triangle sides L₁ to L₃, angle γ, and pose (x, y, θ).\]*

## **II. Method**

### **A. Problem Formulation**

Let the platform pose be $(x, y, \theta)$, with one triangle vertex at $(x, y)$. The other two vertices lie at $(x + L_3 \cos \theta, y + L_3 \sin \theta)$ and $(x + L_2 \cos(\theta + \gamma), y + L_2 \sin(\theta + \gamma))$. Writing the three strut-length constraints and eliminating $x$ and $y$ leaves one equation in $\theta$ \[3\]:

$$f(\theta) = N_1^2 + N_2^2 - p_1^2 D^2 = 0, \qquad x = \frac{N_1}{D}, \quad y = \frac{N_2}{D}$$

The terms are built from the auxiliary quantities

$$A_2 = L_3\cos\theta - x_1, \quad B_2 = L_3\sin\theta, \quad A_3 = L_2\cos(\theta+\gamma) - x_2, \quad B_3 = L_2\sin(\theta+\gamma) - y_2$$

$$N_1 = B_3\,(p_2^2 - p_1^2 - A_2^2 - B_2^2) - B_2\,(p_3^2 - p_1^2 - A_3^2 - B_3^2)$$

$$N_2 = -A_3\,(p_2^2 - p_1^2 - A_2^2 - B_2^2) + A_2\,(p_3^2 - p_1^2 - A_3^2 - B_3^2)$$

$$D = 2\,(A_2 B_3 - B_2 A_3)$$

The function $f(\theta)$ has at most six roots on $[-\pi, \pi]$, so a set of strut lengths gives 0, 2, 4, or 6 poses \[3\]. *\[Check these formulas against the textbook before submitting.\]*

### **B. Root Bracketing**

We evaluate $f$ on a fine grid over $[-\pi, \pi]$ and record every interval where $f$ changes sign. Each such interval brackets at least one root. *\[State the grid size and justify it, e.g., by showing the root count does not change when the grid is refined.\]* A double root, where f touches zero without changing sign, can be missed by this step. We return to this in Section IV.

### **C. Bisection and Brent Refinement**

The bisection method halves a bracketing interval at each step, so the error bound shrinks by a factor of 2 per iteration (linear convergence) \[3\]. Brent's method mixes bisection with secant and inverse quadratic interpolation steps, and falls back to bisection when an interpolation step is unreliable. It keeps the bracketing guarantee but typically converges much faster \[5\]. We write bisection ourselves and use scipy.optimize.brentq for Brent's method. For each refined root $\theta$, we compute $x = N_1/D$ and $y = N_2/D$.

### **D. Verification**

For every pose found, we compute $p_1$, $p_2$, $p_3$ with inverse kinematics and compare them with the input lengths. We also reproduce the examples in Sauer's Reality Check 1 \[3\]: the roots $\theta = \pm\pi/4$, the four-pose case ($x_1 = 5$, $(x_2, y_2) = (0, 6)$, $L_1 = L_3 = 3$, $L_2 = 3\sqrt{2}$, $\gamma = \pi/4$, $p_1 = p_2 = 5$, $p_3 = 3$), and the six-pose case ($p_2 = 7$).

### **E. Implementation**

The solver is written in Python with NumPy, SciPy, and Matplotlib. A shared plotting function draws the triangle, struts, and anchors. *\[Add the repository link, or note that the code is submitted with the report.\]*

## **III. Results**

### **A. Solver Validation**

*\[Report whether the solver reproduces Sauer's examples: the roots at $\theta = \pm\pi/4$, the four-pose case, and the six-pose case. Add a table of $\theta$, $x$, $y$ for each pose and the largest strut-length error from inverse kinematics.\]*

*\[Insert Figure 2 here: plot of f(θ) with roots marked, and the matching platform poses (reproduce Sauer's Figure 1.15).\]*

### **B. Convergence of Bisection and Brent**

*\[Add a table of iterations and function evaluations needed to reach a fixed tolerance (e.g., 10⁻¹⁰) for each root, for both methods. Add a log-scale plot of error versus iteration as Figure 3.\]*

### **C. Sweep 1: Strut Length p₂**

We vary p₂ continuously with the other parameters fixed as in the Sauer example, count the roots at each value, and find the intervals of p₂ that give 0, 2, 4, and 6 poses. *\[Report the interval boundaries and describe what happens at each boundary (double roots).\]*

*\[Insert Figure 4 here: number of poses versus p₂.\]*

### **D. Sweep 2: Anchor Positions**

*\[Describe how varying x₁ and (x₂, y₂) changes the shape of f(θ), the number of poses, and their locations. State whether some base configurations give more poses than others for the same struts and platform.\]*

### **E. Sweep 3: Platform Size and Shape**

*\[Describe the effect of varying (L₁, L₂, L₃) and γ on the number of poses. Show reachable and unreachable poses in workspace plots (Figure 5). Discuss whether changing platform dimensions adds a degree of freedom.\]*

### **F. Sweep 4: Simultaneous Strut Actuation**

*\[Describe the prescribed path in strut space (e.g., pure translation or rotation at a set angular velocity), the root-tracking method, and the corner trajectories (Figure 6). Say whether a path can move the platform between assembly modes and what happens near a singular configuration.\]*

### **G. Engineering Implications**

*\[Summarize what the results mean for design and control, e.g., how to choose strut lengths and base geometry to avoid ambiguous poses or singularities.\]*

## **IV. Discussion**

*\[State how well the methods performed: accuracy against Sauer's examples, speed of bisection versus Brent, and robustness across the sweeps.\]*

The sign-change scan has a known weakness: it can miss a double root, where f touches zero without crossing it, and it can miss two roots that fall inside one grid cell. *\[Say whether this happened in the sweeps, especially near the p₂ values where the pose count changes.\]* Possible improvements include refining the grid near small values of $|f|$, using the minimum of $f$ to flag near-double roots, or writing $f$ as a polynomial in $\tan(\theta/2)$ and solving it directly as a cross-check. *\[Edit to match what the team actually tries.\]*

## **V. Conclusion**

We plan to present a solver for the planar Stewart platform forward kinematics problem, validate it against published examples, compare bisection with Brent's method, and report how strut lengths, base geometry, platform shape, and strut motion affect the possible poses. *\[Rewrite as a summary of the actual findings.\]* Future work could extend the approach to the full six-degree-of-freedom spatial platform or add a method that detects double roots directly. *\[Revise based on results.\]*

## **VI. Acknowledgement**

Winry Cigna wrote the core solver (root bracketing, bisection and Brent refinement, recovery of x and y) and verified it against Sauer's examples. Brytnee Farley carried out Sweep 1 (strut length p₂). Kate Kaneshiro carried out Sweep 2 (anchor positions) and built the platform drawing function. Taylor Todd carried out Sweep 3 (platform size and shape, workspace plots). Alex Tokunaga carried out Sweep 4 (simultaneous strut actuation, trajectory plots, animation). *\[Confirm and edit to match actual contributions.\]*

## **References**

[1] D. Stewart, "A platform with six degrees of freedom," *Proc. Inst. Mech. Eng.*, vol. 180, no. 1, pp. 371–386, 1965.

[2] J.-P. Merlet, *Parallel Robots*, 2nd ed. Dordrecht, Netherlands: Springer, 2006.

[3] T. Sauer, *Numerical Analysis*, 3rd ed. Hoboken, NJ: Pearson, 2018, ch. 1, Reality Check 1.

[4] C. Gosselin and J. Angeles, "Singularity analysis of closed-loop kinematic chains," *IEEE Trans. Robot. Autom.*, vol. 6, no. 3, pp. 281–290, 1990.

[5] SciPy documentation, "scipy.optimize.brentq." [Online]. Available: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html