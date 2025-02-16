#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Button

# -------------------------------
# 1) Lorenz Data Generator
# -------------------------------
def generate_lorenz_data(n_steps=10000, dt=0.01, 
                         sigma=10.0, rho=28.0, beta=8.0/3.0,
                         initial_state=(1.0, 1.0, 1.0)):
    """
    Generate a Lorenz time series (x, y, z) of length n_steps.
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

# -------------------------------
# 2) Define a Stateful Simple RNN Model
# -------------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=100, output_size=3):
        super(SimpleRNN, self).__init__()
        # One-layer RNN with tanh activation (non-LSTM, non-GRU)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # Modify the model so that we apply the linear readout at every time step.
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h0=None):
        # x: shape (batch, seq_len, input_size)
        out, hn = self.rnn(x, h0)  # out: (batch, seq_len, hidden_size)
        y = self.fc(out)           # y: (batch, seq_len, output_size)
        return y, hn

# -------------------------------
# 3) Utility Function: RMSE
# -------------------------------
def rmse(true, pred):
    return np.sqrt(np.mean((true - pred)**2))

# -------------------------------
# 4) Interactive 3D Plot Function
# -------------------------------
def plot_3d_lorenz_with_toggle(Y_true, Y_pred, n_points=10000):
    """
    Creates an interactive 3D plot of the Lorenz trajectories.
    - True trajectory: solid black line.
    - Predicted trajectory: dashed, jet-colored line.
    Clicking the "Toggle Predicted" button hides/shows the predicted line.
    Zoom in/out is enabled via scroll events (mapped to two-finger gestures on your Mac).
    """
    N = min(len(Y_true), n_points)
    true_data = Y_true[:N]
    pred_data = Y_pred[:N]
    
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot true trajectory as a solid black line.
    ax.plot(true_data[:, 0], true_data[:, 1], true_data[:, 2],
            color="black", lw=2, label="Ground-Truth")
    
    # Prepare predicted trajectory segments.
    points = pred_data.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a Line3DCollection for the predicted trajectory.
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(0, len(pred_data)-1)
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linestyles="dashed", linewidth=2.0)
    lc.set_array(np.linspace(0, len(pred_data)-1, len(pred_data)))
    ax.add_collection3d(lc)
    
    # Set axis limits.
    all_data = np.concatenate((true_data, pred_data), axis=0)
    ax.set_xlim(np.min(all_data[:, 0]), np.max(all_data[:, 0]))
    ax.set_ylim(np.min(all_data[:, 1]), np.max(all_data[:, 1]))
    ax.set_zlim(np.min(all_data[:, 2]), np.max(all_data[:, 2]))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title("RNN Inference: 10,803 Trainable Parameters [n_steps = 10,000]")
    
    # Add a colorbar for the predicted trajectory.
    cbar = fig.colorbar(lc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_ticks([0, len(pred_data)-1])
    cbar.set_ticklabels(["Start", "Stop"])
    
    plt.legend()
    
    # Add a button to toggle predicted trajectory visibility.
    ax_button = plt.axes([0.81, 0.05, 0.15, 0.075])
    button = Button(ax_button, 'Toggle Predicted')
    
    def toggle(event):
        vis = not lc.get_visible()
        lc.set_visible(vis)
        plt.draw()
    
    button.on_clicked(toggle)
    
    # Enable zooming in/out via scroll events.
    def on_scroll(event):
        if event.inaxes != ax:
            return
        scale_factor = 0.9 if event.button == 'up' else 1.1 if event.button == 'down' else 1.0
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        xmid = np.mean(xlim)
        ymid = np.mean(ylim)
        zmid = np.mean(zlim)
        new_xlim = [xmid + (x - xmid) * scale_factor for x in xlim]
        new_ylim = [ymid + (y - ymid) * scale_factor for y in ylim]
        new_zlim = [zmid + (z - zmid) * scale_factor for z in zlim]
        ax.set_xlim3d(new_xlim)
        ax.set_ylim3d(new_ylim)
        ax.set_zlim3d(new_zlim)
        plt.draw()
    
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    plt.show()

# -------------------------------
# 5) Main Training Script with Stateful, Truncated BPTT
# -------------------------------
def main():
    # Generate Lorenz data (same as reservoir).
    n_total = 10000
    data = generate_lorenz_data(n_steps=n_total)
    
    # Train/Test split (one-step-ahead prediction)
    # Use the same splitting as in your reservoir code.
    train_len = 8000
    # For training: X_train = data[0:train_len], Y_train = data[1:train_len+1]
    train_data = data[:train_len]
    train_target = data[1:train_len+1]
    # For testing: X_test = data[train_len:-1], Y_test = data[train_len+1:]
    test_data = data[train_len:-1]
    test_target = data[train_len+1:]
    
    # Convert training data to torch tensors.
    train_data_tensor = torch.tensor(train_data, dtype=torch.float32)
    train_target_tensor = torch.tensor(train_target, dtype=torch.float32)
    
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)
    test_target_tensor = torch.tensor(test_target, dtype=torch.float32)
    
    # Reshape training data as a single long sequence (batch size 1, seq_len = train_len)
    X_train_seq = train_data_tensor.unsqueeze(0)        # shape: (1, train_len, 3)
    Y_train_seq = train_target_tensor.unsqueeze(0)        # shape: (1, train_len, 3)
    
    # Define model: use a small hidden size to keep compute comparable.
    input_size = 3
    hidden_size = 100  # modest hidden size
    output_size = 3
    model = SimpleRNN(input_size, hidden_size, output_size)
    
    # Loss and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Truncated BPTT parameters.
    truncation_length = 100  # similar to the RNN script from before.
    num_steps = X_train_seq.size(1)  # should be train_len
    model.train()
    
    h = None
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # Process the long training sequence in chunks.
        for start in tqdm(range(0, num_steps, truncation_length), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=80):
            end = min(start + truncation_length, num_steps)
            X_chunk = X_train_seq[:, start:end, :]  # shape: (1, chunk_len, 3)
            Y_chunk = Y_train_seq[:, start:end, :]  # shape: (1, chunk_len, 3)
            # Forward pass; note: we output predictions at every time step.
            output, h = model(X_chunk, h)
            # Detach hidden state to avoid backpropagating through entire history.
            h = h.detach()
            loss = criterion(output, Y_chunk)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * (end - start)
        epoch_loss /= num_steps
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}")
    
    # Evaluate on the test set.
    # For testing, we similarly process the test sequence as one long sequence.
    X_test_seq = test_data_tensor.unsqueeze(0)      # shape: (1, test_len, 3)
    Y_test_seq = test_target_tensor.unsqueeze(0)      # shape: (1, test_len, 3)
    
    model.eval()
    with torch.no_grad():
        test_output, _ = model(X_test_seq)
        test_loss = criterion(test_output, Y_test_seq).item()
    
    # Convert predictions and targets to numpy.
    Y_test_np = Y_test_seq.squeeze(0).numpy()  # shape: (test_len, 3)
    pred_np = test_output.squeeze(0).numpy()   # shape: (test_len, 3)
    test_rmse = rmse(Y_test_np, pred_np)
    print(f"Test MSE Loss: {test_loss:.6f} - Test RMSE: {test_rmse:.6f}")
    
    # Plot interactive 3D plot using the test target and predictions.
    plot_3d_lorenz_with_toggle(Y_test_np, pred_np, n_points=10000)

if __name__ == '__main__':
    main()
