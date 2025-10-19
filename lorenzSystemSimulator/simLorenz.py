import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import odeint
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def plot_lorenz_plotly(xSolved, ySolved, zSolved):
    import plotly.io as pio
    pio.renderers.default = 'browser'
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=xSolved,
        y=ySolved,
        z=zSolved,
        mode='lines',
        line=dict(color=np.linspace(0, 1, len(xSolved)), colorscale='Viridis', width=4)
    )])
    fig.update_layout(
        title=f'Lorenz Attractor',
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

def LorenzSystem(state, t, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]




def main():
    sigma = 10
    beta = 8 / 3
    rho = 28
    position0 = [0.0, 1.0, 1.0]
    timePoints = np.linspace(0, 60, 10000)
    states = odeint(LorenzSystem, position0, timePoints, args=(sigma, beta, rho))
    xSolved, ySolved, zSolved = states[:,0], states[:,1], states[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_title('Lorenz Attractor', color='white')
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.tick_params(colors='white')

    
    points = np.array([xSolved, ySolved, zSolved]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(timePoints.min(), timePoints.max())
    lc = Line3DCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(timePoints)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim(xSolved.min(), xSolved.max())
    ax.set_ylim(ySolved.min(), ySolved.max())
    ax.set_zlim(zSolved.min(), zSolved.max())


   
    # Plot with Plotly for GPU-accelerated interactive visualization
    plot_lorenz_plotly(xSolved, ySolved, zSolved)
if __name__ == "__main__":
    main()