import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def plot_time_series(
    observations, X_true, X_sim, ensemble=None, subsample_rate=30, threshold=None, color_threshold="red",
    alpha=0.1
):
    n_obs = observations.shape[0]
    n_members, n_sim, _ = X_sim.shape  # Ensure X_sim is 3D
    n_true = X_true.shape[0]

    # Create time indices
    obs_time = np.arange(0, n_obs)
    true_time = np.arange(n_obs, n_obs + n_true)
    sim_time = np.arange(n_obs, n_obs + n_sim)

    # Subsample
    true_time_sub = true_time[::subsample_rate]
    sim_time_sub = sim_time[::subsample_rate]
    h_true_sub = X_true[::subsample_rate, 0]

    # Compute 5th and 95th percentiles for simulation ensemble
    h_sim_5th = np.percentile(X_sim[:, ::subsample_rate, 0], alpha, axis=0)
    h_sim_95th = np.percentile(X_sim[:, ::subsample_rate, 0], 100-alpha, axis=0)

    # Compute the ensemble mean
    ensemble_mean = np.mean(X_sim, axis=0)  # Shape (n_sim, 1)
    ensemble_mean_sub = ensemble_mean[::subsample_rate, 0]  # Subsample for animation

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot observed values
    ax.plot(obs_time, observations[:, 0], color="black")
    
    # Plot threshold if provided
    if threshold is not None:
        ax.hlines(y=threshold, xmin=0, xmax=n_obs + n_sim, linestyles="-.", color=color_threshold, label="Storm threshold")

    # Initialize empty lines for animation
    true_line, = ax.plot([], [], color="black", label="True value")
    mean_line, = ax.plot([], [], color="red", label="Ensemble Mean")  # Red line for mean
    
    # Use a list to store fill_between object
    sim_fill = [ax.fill_between([], [], [], color="grey", alpha=0.3, label="5-95% range")]

    # Set plot limits
    ax.set_xlim(0, n_obs + n_sim)
    ax.set_ylim(
        min(observations[:, 0].min(), X_true[:, 0].min(), X_sim[:, :, 0].min()) - 0.1,
        max(observations[:, 0].max(), X_true[:, 0].max(), X_sim[:, :, 0].max()) + 0.1
    )

    # Improve legend positioning
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    # Animation update function
    def update(frame):
        true_line.set_data(true_time_sub[:frame], h_true_sub[:frame])
        mean_line.set_data(sim_time_sub[:frame], ensemble_mean_sub[:frame])  # Update mean line
        
        # Remove the previous fill_between
        sim_fill[0].remove()
        
        # Add updated fill_between
        sim_fill[0] = ax.fill_between(
            sim_time_sub[:frame], h_sim_5th[:frame], h_sim_95th[:frame], color="grey", alpha=0.3
        )
        
        return (true_line, mean_line)

    # Create animation (blit=False because fill_between does not support blitting)
    ani = animation.FuncAnimation(fig, update, frames=len(true_time_sub), interval=30, blit=False)

    # Close static plot to prevent double display
    plt.close(fig)

    # Display animation in Jupyter Notebook
    return HTML(ani.to_jshtml())



import numpy as np

def generate_positive_definite_matrix(k: int, decay_factor: float = 0.5) -> np.ndarray:
    C_X = np.fromfunction(lambda i, j: np.exp(-decay_factor * np.abs(i - j)), (k, k))
    return C_X/np.linalg.norm(C_X)