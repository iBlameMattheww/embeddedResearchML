import pandas as pd

SHO_TRAINING_INFERENCE_CSV = "Benchmarks/SHO_Results/sho_training_inference_results.csv"
SHO_TEST_OOD_INFERENCE_CSV = "Benchmarks/SHO_Results/sho_test_OOD_inference_results.csv"
SHO_TEST_IID_INFERENCE_CSV = "Benchmarks/SHO_Results/sho_test_IID_inference_results.csv"

def AverageInferenceTime(csv_path):
    df = pd.read_csv(csv_path)

    trajectory_ids = df["trajectory_index"].unique()
    inferenceTimes = []

    for traj_id in trajectory_ids:
        traj_df = df[df["trajectory_index"] == traj_id]
        totalTime = traj_df["inference_time_sec"].iloc[-1]
        inferenceTimes.append(totalTime)

    averageTime = sum(inferenceTimes) / len(inferenceTimes)
    print(f"Average inference time over {len(trajectory_ids)} trajectories: {averageTime:.6f} seconds")
    return averageTime

def main():
    print("Training Inference Time:")
    AverageInferenceTime(SHO_TRAINING_INFERENCE_CSV)
    print("Test IID Inference Time:")
    AverageInferenceTime(SHO_TEST_IID_INFERENCE_CSV)
    print("Test OOD Inference Time:")
    AverageInferenceTime(SHO_TEST_OOD_INFERENCE_CSV)

if __name__ == "__main__":
    main()