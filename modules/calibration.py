from re import DEBUG

import numpy as np
import pandas as pd

from modules.sofa_sim_launcher import run_forward_simulation

DEBUG = True


def read_dataset(dataset, from_real):
    """
    Read dataset and return pairs of (motor commands, end-effector positions).

    Args:
        dataset: Path to the CSV file
        from_real: Whether to use real positions or simulated effector positions

    Returns:
        List of tuples (motor_commands, end_effector_position)
    """
    print(f"Loading dataset from {dataset}")

    # Load the dataset
    df = pd.read_csv(dataset, delimiter=";", skiprows=8)

    # Helper function to parse list strings
    def clean_and_eval_list_string(list_string):
        import ast
        import re

        # Add commas between numbers in the string
        cleaned_string = re.sub(r"(?<=\d)\s+(?=[-\d])", ",", list_string)
        return ast.literal_eval(cleaned_string)

    # Extract motor angles (input) and effector positions (output)
    motor_commands = np.array([clean_and_eval_list_string(angle) for angle in df["Motor angle"].tolist()])

    if from_real and "Real Position" in df.columns:
        # Use real measured positions if available
        end_effector_positions = np.array([clean_and_eval_list_string(pos) for pos in df["Real Position"].tolist()])
    else:
        # Use simulated effector positions
        end_effector_positions = np.array([clean_and_eval_list_string(pos) for pos in df["Effector position"].tolist()])

    # Return as list of (motor_command, end_effector_position) pairs
    dataset_pairs = list(zip(motor_commands, end_effector_positions))

    return dataset_pairs


def calibrate_young(dataset, from_real=False):
    delta = 1e2  # finite-diff parameter
    alpha = 1e7  # stepsize

    E = 2.7e4  # starting value of E
    
    converged = False
    msg = "reached maximum number of iterations"

    tol = 1e-7
    max_iter = 100

    data = []
    
    # plot E
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    subplot_E = fig.add_subplot(1, 3, 1)
    subplot_E.set_title("Calibrated Young's Modulus over Iterations")
    subplot_E.set_xlabel("Iteration")
    subplot_E.set_ylabel("Young's Modulus (E)")
    subplot_error = fig.add_subplot(1, 3, 2)
    subplot_error.set_title("Simulation Error over Iterations")
    subplot_error.set_xlabel("Iteration")
    subplot_error.set_ylabel("Simulation Error (f_sim)")
    subplot_gradient = fig.add_subplot(1, 3, 3)
    subplot_gradient.set_title("Gradient over Iterations")
    subplot_gradient.set_xlabel("Iteration")
    subplot_gradient.set_ylabel("Gradient (dE)")
    
    # read dataset of motor angle - end effector position pairs
    dataset_pairs = read_dataset(dataset, from_real)
    print("Done reading dataset")


    for i in range(max_iter):
        print("-"*40)
        print("Batch iteration:", i)
        print(f"Current Young's modulus: {E:.3f}")
        print("-"*10)

        # select random minibatch
        pair_indices = np.random.choice(len(dataset_pairs), size=10)
        gradient = 0
        error = 0
        motor_angles_batch = np.array([dataset_pairs[j][0] for j in pair_indices])
        p_batch = np.array([dataset_pairs[j][1] for j in pair_indices])

        print(f"Running forward simulation with motor angles:\n {motor_angles_batch}")

        p_sim_batch = run_forward_simulation(E, motor_angles_batch)
        f_sim_batch = np.linalg.norm(p_sim_batch - p_batch, axis=1)

        p_sim_delta_batch = run_forward_simulation(E + delta, motor_angles_batch)
        f_sim_delta_batch = np.linalg.norm(p_sim_delta_batch - p_batch, axis=1)

        if DEBUG:
                print("target position:\n", p_batch)
                print("simulated position:\n", p_sim_batch)
                print("simulated position with E-delta:\n", p_sim_delta_batch)

        gradient = np.mean((f_sim_delta_batch - f_sim_batch) / delta)
        error = np.mean(f_sim_batch)

        # update Young modulus
        E -= alpha * gradient 

        # check convergence
        data.append({"iteration": i, "E": E, "error": error, "gradient": gradient})

        print("-"*10)
        print(f"Results for iteration {i}\n datapoints {pair_indices}\n E {E:.3f}\t error {error:.4f}, gradient {gradient:.4e}")

        subplot_E.plot([d["iteration"] for d in data], [d["E"] for d in data], marker='o')
        subplot_error.plot([d["iteration"] for d in data], [d["error"] for d in data], marker='x')
        subplot_gradient.plot([d["iteration"] for d in data], [d["gradient"] for d in data], marker='v')

        if np.abs(gradient) <= tol:
            converged = True
            msg = "converged in gradient norm"
            break
        
        plt.pause(0.1)
        plt.show(block=False)

    print(f"Calibration finished after {i+1} iterations with Young's modulus: {E:.3f} and error: {error:.4f}")
    plt.show()
    results = {"msg": msg, "success": converged, "data": pd.DataFrame(data)}
    return E, results
