# Simple Harmonic Oscillator (SHO) Simulator

## System Overview

We simulate a 1D simple harmonic oscillator in **phase space** with state:

$$
(x(t), y(t))
$$

where:
- $x(t)$ is the **position**
- $y(t)$ is the **velocity** (or momentum since $m = 1$)

---

## Governing Equations

The continuous-time dynamics are:

$$
\dot{x} = y
$$

$$
\dot{y} = -x
$$

These equations correspond to the Hamiltonian:

$$
H(x, y) = \tfrac{1}{2}(x^2 + y^2)
$$

The Hamiltonian is conserved over time.

---

## Phase Space Interpretation

- The system evolves in **phase space**, not physical space.
- Each point $(x, y)$ represents the full system state at one time.
- Trajectories form **closed circular orbits**.
- Each orbit corresponds to a constant energy level.

---

## Initial Conditions

For each trajectory, initial conditions are sampled as:

$$
x_0 \sim \mathcal{U}(-1, 1)
$$

$$
y_0 \sim \mathcal{U}(-1, 1)
$$

Different initial conditions produce circles of different radii.

---

## Time Parameters

The natural frequency is $\omega = 1$.

The oscillation period is:

$$
T = 2\pi
$$

Simulation parameters:
- Total time: $T = 2\pi$
- Number of steps: $N = 500$
- Time step:

$$
\Delta t = \frac{T}{N - 1}
$$

---

## Numerical Integration: Velocity Verlet

The simple harmonic oscillator satisfies the second-order equation

$$
\ddot{x} = -x
$$

which corresponds to the force

$$
F(x) = -x.
$$

Given the state $(x_k, y_k)$ at time $t_k$, where $y_k = \dot{x}_k$, the Velocity Verlet update is defined by the following steps.

First, the velocity is updated by a half step:

$$
y_{k+\frac{1}{2}} = y_k - \frac{\Delta t}{2} x_k
$$

Next, the position is updated by a full step:

$$
x_{k+1} = x_k + \Delta t \, y_{k+\frac{1}{2}}
$$

Finally, the velocity is updated by another half step:

$$
y_{k+1} = y_{k+\frac{1}{2}} - \frac{\Delta t}{2} x_{k+1}
$$

Velocity Verlet is a symplectic, time-reversible, second-order integrator.  
It preserves phase-space structure and produces bounded energy error over long simulations.


---

## Visualization

Phase-space plots use:

```python
plt.plot(x, y)
plt.axis("equal")

