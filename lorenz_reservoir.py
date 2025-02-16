#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Button
from sklearn.linear_model import Ridge
from tqdm import tqdm

# -------------------------------------------------------------------
# 1) Lorenz Data Generator
# -------------------------------------------------------------------
def generate_lorenz_data(n_steps=10000, dt=0.01, 
                         sigma=10.0, rho=28.0, beta=8.0/3.0,
                         initial_state=(1.0, 1.0, 1.0)):
    """
    Generate a Lorenz time series (x,y,z) of length n_steps.
    Chaotic system governed by:
        dx/dt = sigma*(y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z
    
    Returns:
      data: shape (n_steps, 3)
    """
    xs = np.zeros(n_steps)
    ys = np.zeros(n_steps)
    zs = np.zeros(n_steps)
    
    x, y, z = initial_state
    for i in range(n_steps):
        xs[i] = x
        ys[i] = y
        zs[i] = z
        
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
    
    return np.column_stack([xs, ys, zs])

# -------------------------------------------------------------------
# 2) Multi‐Layer Memristive Reservoir
# -------------------------------------------------------------------
class MultiLayerMemristiveReservoir:
    """
    A multi-layer memristive (echo-state) network. Each layer i:
      new_state[i] = (1 - alpha[i])*old_state[i] + alpha[i]*(W[i]@old_state[i] + Win[i]@input_i)
      state[i] = tanh(new_state[i])
    """
    def __init__(self,
                 layer_sizes,           # list of ints, e.g. [500, 500]
                 alphas,                # list of floats (decay/mixing rates)
                 betas,                 # input scaling divisors
                 spectral_radii,        # list of spectral radii
                 densities,             # list of connection densities
                 input_dim=3,           
                 seed=None):
        
        self.n_layers = len(layer_sizes)
        assert all(len(lst) == self.n_layers 
                   for lst in [alphas, betas, spectral_radii, densities]), \
            "All hyperparameter lists must match #layers."
        
        self.layer_sizes    = layer_sizes
        self.alphas         = alphas
        self.betas          = betas
        self.spectral_radii = spectral_radii
        self.densities      = densities
        self.input_dim      = input_dim
        
        self.rng = np.random.default_rng(seed)
        
        self.W_list   = []
        self.Win_list = []
        self.states   = []
        
        prev_dim = input_dim
        
        for i in range(self.n_layers):
            n_units = layer_sizes[i]
            # Recurrent weights
            W = self.rng.standard_normal((n_units, n_units))
            mask = self.rng.random((n_units, n_units)) < densities[i]
            W *= mask
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            if radius > 0:
                W *= spectral_radii[i] / radius
            self.W_list.append(W)
            
            # Input weights
            Win = self.rng.standard_normal((n_units, prev_dim)) / betas[i]
            self.Win_list.append(Win)
            
            # State init
            self.states.append(np.zeros(n_units))
            prev_dim = n_units
    
    def reset(self):
        """Zero out all layer states."""
        for i in range(self.n_layers):
            self.states[i] = np.zeros(self.layer_sizes[i])
    
    def run(self, X, desc="Running reservoir"):
        """
        Pass a time series X of shape (n_steps, input_dim) through each layer,
        returning (n_steps, layer_sizes[-1]) for the final layer's states.
        
        We wrap the loop with tqdm for a progress bar.
        """
        n_steps, _ = X.shape
        outputs = []
        
        for t in tqdm(range(n_steps), desc=desc, ncols=80):
            current_input = X[t]  # shape (input_dim,)
            
            for i in range(self.n_layers):
                W = self.W_list[i]
                Win = self.Win_list[i]
                alpha = self.alphas[i]
                
                pre_activation = W @ self.states[i] + Win @ current_input
                new_state = (1 - alpha)*self.states[i] + alpha*pre_activation
                self.states[i] = np.tanh(new_state)
                
                current_input = self.states[i]  # feed forward
            
            outputs.append(self.states[-1].copy())
        
        return np.stack(outputs, axis=0)

# -------------------------------------------------------------------
# 3) 3D Plot with Toggle Button and Zooming (via scroll event)
# -------------------------------------------------------------------
def plot_3d_lorenz_with_toggle(Y_true, Y_pred, n_points=10000):
    """
    Plot the 3D Lorenz trajectory with an interactive button that toggles
    the predicted trajectory (dashed, jet-colored) on and off.
    Additionally, a scroll event handler is attached so that zooming in/out
    (which on touch devices is often triggered by a two-finger drag) is supported.
    
    - True trajectory is plotted as a continuous black line.
    - Predicted trajectory is plotted as a dashed line with the jet colormap.
    - A colorbar indicates the start and stop of the predicted trajectory.
    """
    N = min(len(Y_true), n_points)
    true_data = Y_true[:N]
    pred_data = Y_pred[:N]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the continuous true trajectory (solid black line)
    ax.plot(true_data[:, 0], true_data[:, 1], true_data[:, 2],
            color="black", lw=2, label="True")
    
    # Prepare predicted data: create line segments for the predicted trajectory
    points = pred_data.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a Line3DCollection for the predicted trajectory
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(0, len(pred_data)-1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linestyles="dashed", linewidth=2.0)
    lc.set_array(np.linspace(0, len(pred_data)-1, len(pred_data)))
    ax.add_collection3d(lc)
    
    # Set axis limits based on all data
    all_data = np.concatenate((true_data, pred_data), axis=0)
    ax.set_xlim(np.min(all_data[:, 0]), np.max(all_data[:, 0]))
    ax.set_ylim(np.min(all_data[:, 1]), np.max(all_data[:, 1]))
    ax.set_zlim(np.min(all_data[:, 2]), np.max(all_data[:, 2]))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("Memristor Reservoir Inference: 1503 Trainable Parameters [n_steps = 25,000]")
    
    # Add a colorbar for the predicted trajectory
    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_ticks([0, len(pred_data)-1])
    cbar.set_ticklabels(["Start", "Stop"])
    
    plt.legend()
    
    # Add a button to toggle the predicted trajectory's visibility
    ax_button = plt.axes([0.81, 0.05, 0.15, 0.075])
    button = Button(ax_button, 'Toggle Predicted')
    
    def toggle(event):
        vis = not lc.get_visible()
        lc.set_visible(vis)
        plt.draw()
    
    button.on_clicked(toggle)
    
    # Zoom in/out via scroll event (often mapped from two-finger pinch)
    def on_scroll(event):
        # Only proceed if event.inaxes is our axis
        if event.inaxes != ax:
            return
        # Determine scale factor: scroll up -> zoom in, scroll down -> zoom out.
        # You may adjust these values as needed.
        if event.button == 'up':
            scale_factor = 0.9
        elif event.button == 'down':
            scale_factor = 1.1
        else:
            scale_factor = 1.0
        
        # Get current limits
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        # Compute midpoints
        xmid = np.mean(xlim)
        ymid = np.mean(ylim)
        zmid = np.mean(zlim)
        new_xlim = [xmid + (x - xmid)*scale_factor for x in xlim]
        new_ylim = [ymid + (y - ymid)*scale_factor for y in ylim]
        new_zlim = [zmid + (z - zmid)*scale_factor for z in zlim]
        ax.set_xlim3d(new_xlim)
        ax.set_ylim3d(new_ylim)
        ax.set_zlim3d(new_zlim)
        plt.draw()
    
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    
    plt.show()

# -------------------------------------------------------------------
# 4) Utility for RMSE
# -------------------------------------------------------------------
def rmse_3d(true, pred):
    return np.sqrt(np.mean((true - pred)**2))

# -------------------------------------------------------------------
# 5) Main
# -------------------------------------------------------------------
def main():
    # 1) Generate Lorenz Data
    n_total = 10000
    data = generate_lorenz_data(n_steps=n_total)
    
    # Train/Test split (one-step-ahead)
    train_len = 7000
    warmup = 200
    X_train = data[:train_len]
    Y_train = data[1:train_len+1]
    X_test = data[train_len:-1]
    Y_test = data[train_len+1:]
    
    # 2) Build a multi-layer reservoir 
    layer_sizes    = [500, 500]
    alphas         = [0.1, 0.1]
    betas          = [1.0, 1.0]
    spectral_radii = [0.9, 0.9]   
    densities      = [0.1, 0.1]
    
    reservoir = MultiLayerMemristiveReservoir(
        layer_sizes=layer_sizes,
        alphas=alphas,
        betas=betas,
        spectral_radii=spectral_radii,
        densities=densities,
        input_dim=3,
        seed=42
    )
    
    # 3) Training
    print(">> Running reservoir on training data...")
    reservoir.reset()
    states_train = reservoir.run(X_train, desc="Train Reservoir")
    
    # Discard initial 'warmup' states and targets so we only train on the region
    states_train = states_train[warmup:]
    Y_train_trim = Y_train[warmup:]
    
    print(">> Fitting Ridge readout...")
    readout = Ridge(alpha=1e-5)
    readout.fit(states_train, Y_train_trim)
    
    # 4) Test "warm-up" before main test run
    print(">> Warming up the reservoir on the first 200 test steps...")
    reservoir.reset()
    warmup_test_steps = 200
    _ = reservoir.run(X_test[:warmup_test_steps], desc="Test Warm-up")
    
    # 5) Run the remainder of test data (post-warmup)
    print(">> Running reservoir on test data (post-warmup)...")
    states_test_main = reservoir.run(X_test[warmup_test_steps:], 
                                     desc="Test Reservoir (main)")
    
    # Predict on the final states using the trained readout
    Y_pred_main = readout.predict(states_test_main)
    
    # Corresponding ground truth for that slice
    Y_test_main = Y_test[warmup_test_steps:]
    
    # Evaluate RMSE
    err = rmse_3d(Y_test_main, Y_pred_main)
    print(f"Test RMSE (3D, multi-layer memristive), post-warmup: {err:.4f}")
    
    # 6) Plot 3D trajectories with toggle button and zoom capability
    plot_3d_lorenz_with_toggle(Y_test_main, Y_pred_main, n_points=10000)

if __name__ == '__main__':
    main()

