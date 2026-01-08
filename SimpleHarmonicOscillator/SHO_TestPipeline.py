import os 
import numpy as np
import matplotlib.pyplot as plt

RNG_IID = np.random.default_rng(100)
RNG_OOD = np.random.default_rng(200)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'data')

SHO_TEST_IID_DATA_PATH = os.path.join(DATA_PATH, 'sho_Test_IID_Trajectories.npy')
SHO_TEST_OOD_DATA_PATH = os.path.join(DATA_PATH, 'sho_Test_OOD_Trajectories.npy')

def VelocityVerlet(state, dt):
    yHalf = state[1] - 0.5 * dt * state[0]
    xNew = state[0] + dt * yHalf
    yNew = yHalf - 0.5 * dt * xNew
    return [xNew, yNew]

def GenerateSHO_Test_IID_Data(samples):
    k = 1.0  # spring constant
    m = 1.0  # mass
    T = 2 * np.pi  # total time
    steps = 500
    dt = T / (steps - 1)
    totalData = []
    
    for sample in range(samples):
        q, p = RNG_IID.uniform(-1, 1, 2)
        trajectory = np.zeros((steps, 2))

        for step in range(steps):
            trajectory[step] = [q, p]
            q, p = VelocityVerlet([q, p], dt)
        
        totalData.append(trajectory)
    
    totalData = np.array(totalData)
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory at {DATA_PATH}")
    if not os.path.exists(SHO_TEST_IID_DATA_PATH):
        np.save(SHO_TEST_IID_DATA_PATH, totalData)
        print(f"Test data saved to {SHO_TEST_IID_DATA_PATH}")
    else:
        print(f"Test data file already exists at {SHO_TEST_IID_DATA_PATH}, skipping save.")
        import matplotlib.pyplot as plt
        data = np.load(SHO_TEST_IID_DATA_PATH)
        traj = data[0]
        plt.plot(traj[:,0], traj[:,1])
        plt.axis("equal")
        plt.show()

def GenerateSHO_Test_OOD_Data(samples):
    k = 1.0  # spring constant
    m = 1.0  # mass
    T = 2 * np.pi  # total time
    steps = 500
    dt = T / (steps - 1)
    totalData = []
    
    for sample in range(samples):
        q, p = RNG_OOD.uniform(-1.5, 1.5, 2)
        trajectory = np.zeros((steps, 2))

        for step in range(steps):
            trajectory[step] = [q, p]
            q, p = VelocityVerlet([q, p], dt)
        
        totalData.append(trajectory)
    
    totalData = np.array(totalData)
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created directory at {DATA_PATH}")
    if not os.path.exists(SHO_TEST_OOD_DATA_PATH):
        np.save(SHO_TEST_OOD_DATA_PATH, totalData)
        print(f"Test data saved to {SHO_TEST_OOD_DATA_PATH}")
    else:
        print(f"Test data file already exists at {SHO_TEST_OOD_DATA_PATH}, skipping save.")
        import matplotlib.pyplot as plt
        data = np.load(SHO_TEST_OOD_DATA_PATH)
        traj = data[0]
        plt.plot(traj[:,0], traj[:,1])
        plt.axis("equal")
        plt.show()

def main():
    GenerateSHO_Test_IID_Data(samples=15)
    GenerateSHO_Test_OOD_Data(samples=10)

if __name__ == "__main__":
    main()
