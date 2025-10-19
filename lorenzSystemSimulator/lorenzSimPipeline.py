import numpy as np
from scipy.integrate import odeint


def LorenzSystem(state, t, sigma, beta, rho):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def generateLorenzData(samples):
    sigma = 10
    beta = 8 / 3
    rho = 28
    totalData = []
    for i in range(samples):
        position0 = np.random.uniform(-20, 20, 3)
        timePoints = np.linspace(0, 60, 10000)
        states = odeint(LorenzSystem, position0, timePoints, args=(sigma, beta, rho))
        xSolved, ySolved, zSolved = states[:,0], states[:,1], states[:,2]
        data = np.stack([xSolved, ySolved, zSolved], axis=1)
        totalData.append(data)
    totalData = np.array(totalData)
    np.save('C:\\Users\\Matthew\\Downloads\\lorenz_trajectories.npy', totalData)



def main():
    generateLorenzData(samples=10000)

if __name__ == "__main__":
    main()