import numpy as np
import matplotlib.pyplot as plt

SHO_DATA_PATH = "data/sho_trajectories.npy" 

def main():
    data = np.load(SHO_DATA_PATH)
    print(f"Loaded data with shape: {data.shape}")  
    # Expected: (num_trajectories, steps, 2)

    traj = data[0]  # shape: (steps, 2)

    print("\nFirst trajectory coordinate points (q, p):")
    for i, (q, p) in enumerate(traj):
        print(f"Step {i:03d}: q = {q:.6f}, p = {p:.6f}")

    plt.figure()
    plt.plot(traj[:, 0], traj[:, 1])
    plt.xlabel("q (position)")
    plt.ylabel("p (momentum)")
    plt.title("SHO Phase-Space Trajectory (First Sample)")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
