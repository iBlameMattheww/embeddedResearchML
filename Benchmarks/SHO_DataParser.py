import numpy as np
import pandas as pd
import os

from SimpleHarmonicOscillator.SHO_visualizer import SHO_DATA_PATH

SHO_TEST_DATA_PATH = "SimpleHarmonicOscillator/data/sho_trajectories.npy"

def DataLoader(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    data = np.load(path)
    return data

def ParseTrajectory(data):
    numTrajectories, numSteps, coordinates = data.shape
    records = []
    for trajectoryIndex in range(numTrajectories):
        for stepIndex in range(numSteps):
            q, p = data[trajectoryIndex, stepIndex]
            records.append({
                'trajectory_index': trajectoryIndex,
                'step_index': stepIndex,
                'q': q,
                'p': p
            })
    df = pd.DataFrame.from_records(records)
    return df

def main():
    print(f"Loading data from {SHO_TEST_DATA_PATH}")
    data = DataLoader(SHO_TEST_DATA_PATH)
    print(f"Data shape: {data.shape}") 

if __name__ == "__main__":
    main()