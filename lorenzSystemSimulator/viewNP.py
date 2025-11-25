import numpy as np
import plotly.graph_objects as go
import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')
data = np.load(os.path.join(data_dir, 'lorenz_trajectories500.npy'))
print(data.shape)        # (samples, timesteps, 3)
print(data[0])           # First trajectory

traj = data[0]           # shape: (timesteps, 3)
fig = go.Figure(data=[go.Scatter3d(
    x=traj[:, 0],
    y=traj[:, 1],
    z=traj[:, 2],
    mode='lines',
    line=dict(color=np.linspace(0, 1, len(traj)), colorscale='Viridis', width=4)
)])
fig.update_layout(
    title='Lorenz Attractor (First Trajectory)',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        bgcolor='black',
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)
fig.show()