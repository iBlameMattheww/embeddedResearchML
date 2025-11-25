# Lorenz System Simulation Data Generation

## Overview

The Lorenz system is a classic example of a chaotic dynamical system. This workflow generates multiple 3D trajectories using random initial conditions and visualizes them for analysis or ML tasks.

---

## How to Generate Lorenz Trajectories

- The pipeline script numerically solves the Lorenz equations for many random starting points using `scipy.integrate.odeint`.
- Each trajectory is a time series of 3D points (`x`, `y`, `z`).
- All trajectories are stacked into a single numpy array and saved for later use.

**Run the pipeline script:**
```sh
python lorenzSystemSimulator/lorenzSimPipeline.py
```
- Output will be saved to:
  ```
  lorenzSystemSimulator/data/lorenz_trajectories500.npy
  ```
  (Change the sample count in the script to adjust filename/size.)

---

## Visualize Trajectories

- The view script loads the saved numpy file and plots the first trajectory in 3D using Plotly.
- The plot is interactive, allowing you to explore the attractor’s structure.

**Run the view script:**
```sh
python lorenzSystemSimulator/viewNP.py
```
- This will plot the first trajectory in 3D using Plotly.

---

## Simulation Details

- **Time Steps:** Each trajectory is computed with 10,000 time steps, providing high temporal resolution.
- **Simulation Duration:** The time array spans 0 to 60 seconds, so each trajectory covers 60 seconds of Lorenz system evolution.
- **Initial Conditions:** Each run starts from a random position in 3D space (uniformly sampled between -20 and 20 for each axis).
- **Parameters:**
  - Sigma (σ): 10
  - Beta (β): 8/3
  - Rho (ρ): 28
- **Numerical Integration:** Uses `scipy.integrate.odeint` to solve the Lorenz differential equations for each initial condition.
- **Output Shape:** The saved numpy array has shape `(samples, 10000, 3)` (samples × time steps × coordinates).

---

## Customization & Analysis

- You can modify the pipeline to change the number of samples, time range, or Lorenz parameters (`sigma`, `beta`, `rho`).
- The generated data can be used for ML experiments, e.g., training models to predict or classify chaotic behavior.
- The output file is large and excluded from git via `.gitignore`.

---

## Requirements

- Python packages: `numpy`, `scipy`, `plotly` (install with `pip install numpy scipy plotly`).

---

For more details, see the source scripts in `lorenzSystemSimulator/`.
