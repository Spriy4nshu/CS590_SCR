import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_ep_rew_mean(log_dir, timestep_limit=200000):
    """
    Plots the rollout/ep_rew_mean from TensorBoard logs up to a specified timestep.

    Args:
    - log_dir (str): Path to the directory containing TensorBoard log files.
    - timestep_limit (int): The maximum timestep to include in the plot.
    """
    # List all subdirectories in the log directory
    runs = [os.path.join(log_dir, run) for run in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, run))]

    plt.figure(figsize=(10, 6))

    # Process each run
    for run_dir in runs:
        ea = event_accumulator.EventAccumulator(run_dir)
        ea.Reload()  # Loads the log data

        # Check if 'rollout/ep_rew_mean' is in the scalars
        if 'rollout/ep_rew_mean' in ea.scalars.Keys():
            # Extract ScalarEvents for 'rollout/ep_rew_mean'
            scalar_events = ea.Scalars('rollout/ep_rew_mean')
            
            # Initialize lists to hold the filtered steps and values
            filtered_steps = []
            filtered_vals = []

            # Iterate through ScalarEvents to extract and filter data
            for event in scalar_events:
                if event.step <= timestep_limit:
                    filtered_steps.append(event.step)
                    filtered_vals.append(event.value)

            # Plot the filtered data
            plt.plot(filtered_steps, filtered_vals, label=os.path.basename(run_dir))

    plt.title('Episode Reward Mean over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Episode Reward Mean')
    plt.legend()
    plt.grid(True)
    plt.show()



log_dir = 'logs'
plot_ep_rew_mean(log_dir, 200000)
