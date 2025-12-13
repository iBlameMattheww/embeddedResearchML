# Simple Harmonic Oscillator (SHO) Simulator Notes

## System Definition

We simulate a 1D simple harmonic oscillator in **phase space** with state:

\[
(x(t), y(t))
\]

where:
- \(x(t)\) = position
- \(y(t)\) = momentum (or velocity since \(m = 1\))

The governing equations are:

\[
\dot{x} = y
\]
\[
\dot{y} = -x
\]

This corresponds to a Hamiltonian:

\[
H(x,y) = \tfrac{1}{2}(x^2 + y^2)
\]

which is conserved.

---

## Phase Space Interpretation

- The system evolves in **phase space**, not physical space.
- Each point \((x,y)\) represents the full state at one time.
- The trajectory forms a **closed circular orbit**.
- Each orbit corresponds to a constant energy level.

---

## Initial Conditions

For each trajectory:
\[
x_0 \sim \mathcal{U}(-1, 1)
\]
\[
y_0 \sim \mathcal{U}(-1, 1)
\]

Different initial conditions produce circles of different radii.

---

## Time Parameters

- Natural frequency: \(\omega = 1\)
- Period:  
  \[
  T = 2\pi
  \]

Simulation parameters:
- Total time: \(T = 2\pi\)
- Number of steps: 500
- Time step: \(\Delta t = T / (N - 1)\)

---

## Numerical Integration

We use **Velocity Verlet**, a symplectic (leapfrog) integrator:

1. Half-step momentum update:
\[
y_{k+\frac{1}{2}} = y_k - \frac{\Delta t}{2} x_k
\]

2. Full-step position update:
\[
x_{k+1} = x_k + \Delta t \, y_{k+\frac{1}{2}}
\]

3. Half-step momentum update:
\[
y_{k+1} = y_{k+\frac{1}{2}} - \frac{\Delta t}{2} x_{k+1}
\]

Properties:
- Symplectic (preserves phase-space volume)
- Time-reversible
- Bounded energy error
- Suitable for long-term rollouts

---

## Visualization

Phase-space plots use:
```python
plt.plot(x, y)
plt.axis("equal")
