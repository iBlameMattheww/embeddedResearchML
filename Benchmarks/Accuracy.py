import pandas as pd
import numpy as np

SHO_TRAINING_TRUTH_NPY = "SimpleHarmonicOscillator/data/sho_trajectories.npy"
SHO_TRAINING_INFERENCE_CSV = "Benchmarks/SHO_Results/sho_training_inference_results.csv"

SHO_TEST_OOD_TRUTH_NPY = "SimpleHarmonicOscillator/data/sho_Test_OOD_Trajectories.npy"
SHO_TEST_OOD_INFERENCE_CSV = "Benchmarks/SHO_Results/sho_test_OOD_inference_results.csv"

SHO_TEST_IID_TRUTH_NPY = "SimpleHarmonicOscillator/data/sho_Test_IID_Trajectories.npy"
SHO_TEST_IID_INFERENCE_CSV = "Benchmarks/SHO_Results/sho_test_IID_inference_results.csv"

import numpy as np

def MeanRelativeStateError(inference, groundTruth, steps, eps=1e-8):
    # Fixed normalization scale (trajectory radius)
    q0, p0 = groundTruth[0]
    scale = np.sqrt(q0**2 + p0**2) + eps

    total_rel_error = 0.0

    for step in range(steps):
        q_inf, p_inf = inference[step]
        q_gt, p_gt = groundTruth[step]

        num = np.sqrt((q_inf - q_gt)**2 + (p_inf - p_gt)**2)
        rel_error = num / scale

        total_rel_error += rel_error

    return total_rel_error / steps


def MRSE_Percentage(mean_rel_error):
    mrse_percentage = mean_rel_error * 100
    return mrse_percentage

def AbsoluteStateError(inference, groundTruth, steps):
    total_abs_error = 0.0

    for step in range(steps):
        q_inf, p_inf = inference[step]
        q_gt, p_gt = groundTruth[step]

        abs_error = np.sqrt((q_inf - q_gt)**2 + (p_inf - p_gt)**2)
        total_abs_error += abs_error

    return total_abs_error / steps

def ErrorPercentage(absolute_error, normalization_scale):
    error_percentage = (absolute_error / normalization_scale) * 100
    return error_percentage

def AccuracyPipeline(inference_csv_path, ground_truth_data_path):
    df = pd.read_csv(inference_csv_path)
    ground_truth_data = np.load(ground_truth_data_path)

    trajectory_ids = df["trajectory_index"].unique()
    mrse_list = []
    absoluteErrors = []

    for traj_id in trajectory_ids:
        traj_df = (
            df[df["trajectory_index"] == traj_id]
            .sort_values("step_index")
        )

        inferred_traj = traj_df[["q", "p"]].to_numpy()
        step_idx = traj_df["step_index"].to_numpy()

        gt_traj = ground_truth_data[traj_id]

        # ALIGN BY STEP INDEX
        valid_mask = step_idx < len(gt_traj)
        inferred_traj = inferred_traj[valid_mask]
        step_idx = step_idx[valid_mask]

        gt_aligned = gt_traj[step_idx]

        if len(inferred_traj) == 0:
            raise ValueError(f"No valid aligned steps for trajectory {traj_id}")

        mean_rel_error = MeanRelativeStateError(
            inferred_traj,
            gt_aligned,
            steps=len(inferred_traj)
        )

        absoluteEroor = AbsoluteStateError(
            inferred_traj,
            gt_aligned,
            steps=len(inferred_traj)
        )

        mrse_list.append(mean_rel_error)
        absoluteErrors.append(absoluteEroor)

    average_mrse = np.mean(mrse_list)
    print(f"Average MRSE over {len(mrse_list)} trajectories: {average_mrse * 100:.4f}%")
    print(f"Average Absolute Error over {len(mrse_list)} trajectories: {np.mean(absoluteErrors) * 100:.6f}%")
    return average_mrse * 100


def main():
    print("Training Accuracy:")
    AccuracyPipeline(SHO_TRAINING_INFERENCE_CSV, SHO_TRAINING_TRUTH_NPY)
    print("Test IID Accuracy:")
    AccuracyPipeline(SHO_TEST_IID_INFERENCE_CSV, SHO_TEST_IID_TRUTH_NPY)
    print("Test OOD Accuracy:")
    AccuracyPipeline(SHO_TEST_OOD_INFERENCE_CSV, SHO_TEST_OOD_TRUTH_NPY)

if __name__ == "__main__":
    main()