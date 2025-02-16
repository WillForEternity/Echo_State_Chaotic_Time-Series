#!/usr/bin/env python3
import time
import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from tqdm import tqdm
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.optim as optim

#########################################################
# 1) Lorenz Data Generator (shared for both models)
#########################################################
def generate_lorenz_data(n_steps=10000, dt=0.01, 
                         sigma=10.0, rho=28.0, beta=8.0/3.0,
                         initial_state=(1.0, 1.0, 1.0)):
    """
    Generate Lorenz time series (x, y, z) of length n_steps.
    Returns: data of shape (n_steps, 3)
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

#########################################################
# 2) RMSE Utility (shared)
#########################################################
def rmse_3d(true, pred):
    return np.sqrt(np.mean((true - pred)**2))

#########################################################
# 3) Multi-Layer Memristive Reservoir (Echo State Net)
#########################################################
class MultiLayerMemristiveReservoir:
    """
    A multi-layer memristive (echo-state) network.
    new_state[i] = (1 - alpha[i])*old_state[i] + alpha[i]*(W[i]@old_state[i] + Win[i]@input_i)
    state[i] = tanh(new_state[i])
    """
    def __init__(self,
                 layer_sizes,           
                 alphas,                
                 betas,                 
                 spectral_radii,        
                 densities,             
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
        Pass a time series X: shape (n_steps, input_dim) through each layer,
        returning (n_steps, layer_sizes[-1]) for the final layer's states.
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

#########################################################
# 4) Reservoir Experiment (CPU usage + runtime)
#########################################################
def run_reservoir_experiment(n_total=20000):
    # Prepare data
    data = generate_lorenz_data(n_steps=n_total)
    
    # Training / test split
    train_len = 15000
    warmup = 200
    X_train = data[:train_len-1]
    Y_train = data[1:train_len]
    X_test = data[train_len:-1]
    Y_test = data[train_len+1:]
    
    # Build reservoir
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
    
    # Measure time & CPU usage
    proc = psutil.Process(os.getpid())
    start_cpu_times = proc.cpu_times()
    wall_t0 = time.time()
    
    # Train
    reservoir.reset()
    states_train = reservoir.run(X_train, desc="Reservoir Train")
    states_train = states_train[warmup:]
    Y_train_trim = Y_train[warmup:]
    
    readout = Ridge(alpha=1e-5)
    readout.fit(states_train, Y_train_trim)
    
    # Test warm-up
    reservoir.reset()
    warmup_test_steps = 200
    _ = reservoir.run(X_test[:warmup_test_steps], desc="Test Warm-up (Reservoir)")
    
    # Main test
    states_test_main = reservoir.run(X_test[warmup_test_steps:], desc="Test Reservoir (main)")
    Y_pred_main = readout.predict(states_test_main)
    Y_test_main = Y_test[warmup_test_steps:]
    
    wall_t1 = time.time()
    end_cpu_times = proc.cpu_times()
    
    cpu_user = end_cpu_times.user - start_cpu_times.user
    cpu_sys  = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = cpu_user + cpu_sys
    wall_time = wall_t1 - wall_t0
    avg_cpu_percent = 100.0 * (total_cpu_time / wall_time) if wall_time > 0 else 0
    
    error = rmse_3d(Y_test_main, Y_pred_main)
    
    return {
        "model": "Multi-Layer Reservoir",
        "time_sec": wall_time,
        "cpu_time_sec": total_cpu_time,
        "avg_cpu_pct": avg_cpu_percent,
        "rmse": error,
        # We'll store the test data and predictions for plotting later
        "Y_test_main": Y_test_main,
        "Y_pred_main": Y_pred_main
    }

#########################################################
# 5) Simple RNN (PyTorch)
#########################################################
class SimpleRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=3):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None):
        out, hn = self.rnn(x, h0)
        y = self.fc(out)
        return y, hn

#########################################################
# 6) RNN Experiment (CPU usage + runtime)
#########################################################
def run_rnn_experiment(n_total=20000, device='cpu'):
    # Prepare data
    data = generate_lorenz_data(n_steps=n_total)
    
    train_len = 15000
    train_data = data[:train_len]
    train_target = data[1:train_len+1]
    test_data = data[train_len:-1]
    test_target = data[train_len+1:]
    
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)
    
    X_train_seq = train_data_tensor.unsqueeze(0) # shape: (1, train_len, 3)
    Y_train_seq = train_target_tensor.unsqueeze(0) # shape: (1, train_len, 3)
    
    input_size = 3
    hidden_size = 100
    output_size = 3
    model = SimpleRNN(input_size, hidden_size, output_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    truncation_length = 100
    num_steps = X_train_seq.size(1)
    num_epochs = 10
    
    # Measure time & CPU usage
    proc = psutil.Process(os.getpid())
    start_cpu_times = proc.cpu_times()
    wall_t0 = time.time()
    
    model.train()
    h = None
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for start in tqdm(range(0, num_steps, truncation_length),
                          desc=f"RNN Epoch {epoch+1}/{num_epochs}",
                          ncols=80):
            end = min(start + truncation_length, num_steps)
            X_chunk = X_train_seq[:, start:end, :]
            Y_chunk = Y_train_seq[:, start:end, :]
            
            output, h = model(X_chunk, h)
            # Detach hidden state so we don't backprop through entire history
            h = h.detach()
            
            loss = criterion(output, Y_chunk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * (end - start)
        
        epoch_loss /= num_steps
        print(f"[RNN] Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    
    model.eval()
    with torch.no_grad():
        X_test_seq = test_data_tensor.unsqueeze(0)
        Y_test_seq = test_target_tensor.unsqueeze(0)
        test_output, _ = model(X_test_seq)
        test_loss = criterion(test_output, Y_test_seq).item()
    
    wall_t1 = time.time()
    end_cpu_times = proc.cpu_times()
    
    cpu_user = end_cpu_times.user - start_cpu_times.user
    cpu_sys  = end_cpu_times.system - start_cpu_times.system
    total_cpu_time = cpu_user + cpu_sys
    wall_time = wall_t1 - wall_t0
    avg_cpu_percent = 100.0 * (total_cpu_time / wall_time) if wall_time > 0 else 0
    
    Y_test_np = Y_test_seq.squeeze(0).cpu().numpy()
    pred_np = test_output.squeeze(0).cpu().numpy()
    test_rmse_val = rmse_3d(Y_test_np, pred_np)
    print(f"[RNN] Test MSE: {test_loss:.6f}, Test RMSE: {test_rmse_val:.6f}")
    
    return {
        "model": "Traditional RNN (PyTorch)",
        "time_sec": wall_time,
        "cpu_time_sec": total_cpu_time,
        "avg_cpu_pct": avg_cpu_percent,
        "rmse": test_rmse_val,
        # For plotting
        "Y_test_main": Y_test_np,
        "Y_pred_main": pred_np
    }

#########################################################
# 7) Side-by-side plotting
#########################################################
def plot_3d_lorenz(ax, Y_true, Y_pred, title="", n_points=10000):
    """
    Plots a Lorenz trajectory (Y_true in black, Y_pred in dashed color)
    onto an existing Axes3D object.
    """
    N = min(len(Y_true), n_points)
    true_data = Y_true[:N]
    pred_data = Y_pred[:N]
    
    # True trajectory (solid black)
    ax.plot(true_data[:, 0], true_data[:, 1], true_data[:, 2],
            color="black", lw=2, label="True")
    
    # Predicted trajectory (dashed multi-color)
    points = pred_data.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(0, len(pred_data)-1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linestyles="dashed", linewidth=2.0)
    lc.set_array(np.linspace(0, len(pred_data)-1, len(pred_data)))
    ax.add_collection3d(lc)
    
    # Determine axis limits
    combined = np.concatenate((true_data, pred_data), axis=0)
    ax.set_xlim(np.min(combined[:, 0]), np.max(combined[:, 0]))
    ax.set_ylim(np.min(combined[:, 1]), np.max(combined[:, 1]))
    ax.set_zlim(np.min(combined[:, 2]), np.max(combined[:, 2]))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    
    # Horizontal colorbar below plot
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, orientation='horizontal', shrink=0.7, pad=0.15)
    cbar.set_label("Prediction Step")
    
    # Legend without a box
    ax.legend(loc='upper center', bbox_to_anchor=(0.7, -0.1), ncol=2, fontsize='small', frameon=False)

def plot_cpu_comparison(ax, reservoir_results, rnn_results):
    """
    Plots a bar chart comparing CPU usage and time metrics for the two models
    on a provided Matplotlib axis.
    """
    metrics = ["Wall-clock Time (s)", "CPU Time (s)", "Avg CPU Usage (%)"]
    reservoir_values = [
        reservoir_results["time_sec"],
        reservoir_results["cpu_time_sec"],
        reservoir_results["avg_cpu_pct"]
    ]
    rnn_values = [
        rnn_results["time_sec"],
        rnn_results["cpu_time_sec"],
        rnn_results["avg_cpu_pct"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35  # Bar width
    
    ax.bar(x - width/2, reservoir_values, width, label='Reservoir', color='darkblue')
    ax.bar(x + width/2, rnn_values, width, label='RNN', color='limegreen')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylabel('Value')
    ax.set_title('CPU Usage and Runtime Comparison')
    ax.legend(fontsize='small', frameon=False)

#########################################################
# 8) Main
#########################################################
def main():
    print("=== Running Reservoir Experiment on CPU ===")
    reservoir_results = run_reservoir_experiment(n_total=20000)
    
    print("\n=== Running RNN Experiment on CPU ===")
    rnn_results = run_rnn_experiment(n_total=20000, device='cpu')
    
    # Create a figure with 3 subplots.
    # Explicitly create the 3D axes to avoid the extra 2D border.
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133)
    
    # 1) Reservoir 3D plot
    plot_3d_lorenz(ax1,
                   reservoir_results["Y_test_main"],
                   reservoir_results["Y_pred_main"],
                   title="Reservoir ESN",
                   n_points=10000)
    
    # 2) RNN 3D plot
    plot_3d_lorenz(ax2,
                   rnn_results["Y_test_main"],
                   rnn_results["Y_pred_main"],
                   title="Traditional RNN",
                   n_points=10000)
    
    # 3) CPU bar chart
    plot_cpu_comparison(ax3, reservoir_results, rnn_results)
    
    # Adjust spacing so the plots extend closer to the window edges.
    plt.subplots_adjust(left=0.00, right=0.98, top=0.95, bottom=0.1, wspace=0.4)
    plt.show(block=False)
    
    # Print final comparison statistics
    print("\n===========================")
    print(" COMPARISON OF BOTH MODELS ")
    print("===========================")
    print("{:<30}  {:>16}  {:>16}".format("Metric", "Reservoir", "RNN"))
    print("-"*65)
    print("{:<30}  {:>16.4f}  {:>16.4f}".format(
        "Test RMSE", reservoir_results["rmse"], rnn_results["rmse"]))
    print("{:<30}  {:>16.2f}  {:>16.2f}".format(
        "Wall-clock time (sec)",
        reservoir_results["time_sec"], rnn_results["time_sec"]))
    print("{:<30}  {:>16.2f}  {:>16.2f}".format(
        "CPU time (sec)",
        reservoir_results["cpu_time_sec"], rnn_results["cpu_time_sec"]))
    print("{:<30}  {:>15.1f}%  {:>15.1f}%".format(
        "Approx Avg CPU Usage",
        reservoir_results["avg_cpu_pct"], rnn_results["avg_cpu_pct"]))
    print("===========================")

    input("\nPress ENTER to close plots and exit...") 

if __name__ == "__main__":
    main()
