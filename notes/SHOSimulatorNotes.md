# Simple Harmonic Oscillator (SHO) Simulator

## System Overview

We simulate a 1D simple harmonic oscillator in **phase space** with state:

$$
(q(t), p(t))
$$

where:
- $q(t)$ is the **position**
- $p(t)$ is the **momentum** (for $m = 1$, $p = \dot{q}$)

---

## Governing Equations

The continuous-time dynamics are:

$$
\dot{q} = p
$$

$$
\dot{p} = -q
$$

These equations correspond to the Hamiltonian:

$$
H(q, p) = \tfrac{1}{2}(q^2 + p^2)
$$

The Hamiltonian is conserved over time.

---

## Phase Space Interpretation

- The system evolves in **phase space**, not physical space.
- Each point $(q, p)$ represents the full system state at one time.
- Trajectories form **closed circular orbits**.
- Each orbit corresponds to a constant energy level.

---

## Initial Conditions

For each trajectory, initial conditions are sampled as:

$$
q_0 \sim \mathcal{U}(-1, 1)
$$

$$
p_0 \sim \mathcal{U}(-1, 1)
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
\ddot{q} = -q
$$

which corresponds to the force

$$
F(q) = -q.
$$

Given the state $(q_k, p_k)$ at time $t_k$, where $p_k = \dot{q}_k$, the Velocity Verlet update is defined by the following steps.

First, the momentum is updated by a half step:

$$
p_{k+\frac{1}{2}} = p_k - \frac{\Delta t}{2} q_k
$$

Next, the position is updated by a full step:

$$
q_{k+1} = q_k + \Delta t \, p_{k+\frac{1}{2}}
$$

Finally, the momentum is updated by another half step:

$$
p_{k+1} = p_{k+\frac{1}{2}} - \frac{\Delta t}{2} q_{k+1}
$$

Velocity Verlet is a symplectic, time-reversible, second-order integrator.  
It preserves phase-space structure and produces bounded energy error over long simulations.

---

## Visualization

Phase-space plots use:

```python
plt.plot(q, p)
plt.axis("equal")
```

