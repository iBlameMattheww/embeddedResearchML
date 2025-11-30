# Phase portraits, time evoultion plots, etc
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("seaborn-v0_8")   # cleaner plots


# -------------------------------------------------------
# 1. Plot loss curves
# -------------------------------------------------------
def plot_losses(train_losses, recon_losses, energy_losses, savePath=None):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction")
    plt.plot(energy_losses, label="Energy Drift")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)

    if savePath:
        plt.savefig(savePath, dpi=200)
        print(f"[✓] Saved loss curve plot → {savePath}")

    plt.show()


# -------------------------------------------------------
# 2. Plot 3D Lorenz trajectory (ground truth + predicted)
# -------------------------------------------------------
def plot_lorenz_plotly(trueTraj, predTraj=None, title="Lorenz Attractor", show=True, savePath=None):
    fig = go.Figure()

    # Ground truth
    fig.add_trace(
        go.Scatter3d(
            x=trueTraj[:, 0],
            y=trueTraj[:, 1],
            z=trueTraj[:, 2],
            mode="lines",
            name="Ground Truth",
            line=dict(
                color=np.linspace(0, 1, len(trueTraj)),
                colorscale="Viridis",
                width=4
            )
        )
    )

    # Prediction
    if predTraj is not None:
        fig.add_trace(
            go.Scatter3d(
                x=predTraj[:, 0],
                y=predTraj[:, 1],
                z=predTraj[:, 2],
                mode="lines",
                name="SymplecticNN Prediction",
                line=dict(
                    color=np.linspace(0, 1, len(predTraj)),
                    colorscale="Portland",
                    width=4
                )
            )
        )

    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Save + Show
    if savePath:
        fig.write_html(savePath)
        print(f"[✓] Saved interactive Plotly plot → {savePath}")

    if show:
        fig.show()

    return fig


# -------------------------------------------------------
# 3. Phase Portraits
# -------------------------------------------------------
def plot_phase_portraits(trueTraj, predTraj=None, savePath=None):
    x, y, z = trueTraj[:,0], trueTraj[:,1], trueTraj[:,2]

    fig, axs = plt.subplots(1,3, figsize=(14,4))

    axs[0].plot(x, y, label="True")
    if predTraj is not None:
        axs[0].plot(predTraj[:,0], predTraj[:,1], label="Predicted")
    axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
    axs[0].set_title("x-y plane")

    axs[1].plot(x, z, label="True")
    if predTraj is not None:
        axs[1].plot(predTraj[:,0], predTraj[:,2])
    axs[1].set_xlabel("x"); axs[1].set_ylabel("z")
    axs[1].set_title("x-z plane")

    axs[2].plot(y, z, label="True")
    if predTraj is not None:
        axs[2].plot(predTraj[:,1], predTraj[:,2])
    axs[2].set_xlabel("y"); axs[2].set_ylabel("z")
    axs[2].set_title("y-z plane")

    for ax in axs:
        ax.grid(True)
        ax.legend()

    if savePath:
        plt.savefig(savePath, dpi=200)
        print(f"[✓] Saved phase portraits → {savePath}")

    plt.show()


# -------------------------------------------------------
# 4. Energy Drift
# -------------------------------------------------------
def plot_energy_drift(energies, savePath=None):
    plt.figure(figsize=(8,5))
    plt.plot(np.abs(energies), label="|H(t) - H(0)|")

    plt.xlabel("Time Step")
    plt.ylabel("Energy Drift")
    plt.title("Energy Drift Over Time")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    if savePath:
        plt.savefig(savePath, dpi=200)
        print(f"[✓] Saved energy drift plot → {savePath}")

    plt.show()
