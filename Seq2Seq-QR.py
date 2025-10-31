import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
# Matplotlib is imported for plotting

# --- Configuration ---
HORIZON = 20  # Forecast steps: t+1, t+2, t+3, t+4, t+5, t+6
WINDOW_SIZE = 10 # Look-back steps
QUANTILES = [0.05, 0.50, 0.95] # 90% PI (5th, 50th, 95th percentiles)
N_QUANTILES = len(QUANTILES)
OUTPUT_SIZE = HORIZON * N_QUANTILES # 6 * 3 = 18 output nodes

# --- 1. The Custom Pinball Loss Function ---
def pinball_loss(y_true, y_pred, quantiles):
    """
    Calculates the Pinball Loss (Quantile Loss) for multi-quantile, multi-horizon output.
    """
    
    y_pred = y_pred.view(-1, HORIZON, N_QUANTILES)
    y_true_expanded = y_true.unsqueeze(-1)
    errors = y_true_expanded - y_pred
    
    loss = 0
    for i, q in enumerate(quantiles):
        qloss_i = torch.max(q * errors[:,:,i], (q - 1) * errors[:,:,i])
        loss += qloss_i.mean()
        
    return loss / N_QUANTILES

# --- 2. The Seq2Seq Quantile Model ---
class Seq2SeqQuantileForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2SeqQuantileForecaster, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # --- FIX APPLIED ---
        self.output_linear = nn.Linear(hidden_size * HORIZON, output_size)
        
        self.horizon = HORIZON
        self.n_quantiles = N_QUANTILES

    def forward(self, x):
        _, (hn, cn) = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), self.horizon, x.size(2)).to(x.device)
        decoder_output, _ = self.decoder(decoder_input, (hn, cn))
        
        # Reshape to [BATCH_SIZE, HORIZON * HIDDEN_SIZE]
        output = self.output_linear(decoder_output.reshape(decoder_output.size(0), -1))
        
        return output


# --- 3. Simulated Data & Training ---

# Dummy Data: A simple rising trend with noise
def generate_data(size):
    X = []
    Y = []
    t = np.arange(0, size + WINDOW_SIZE + HORIZON)
    # Generate noisy sine wave with trend
    series = 100 + 0.5 * t + 10 * np.sin(t / 5) + np.random.randn(len(t)) * 5
    
    for i in range(len(series) - WINDOW_SIZE - HORIZON):
        x = series[i : i + WINDOW_SIZE]
        y = series[i + WINDOW_SIZE : i + WINDOW_SIZE + HORIZON]
        X.append(x)
        Y.append(y)
    
    # Return the full series as well for plotting the history
    full_series = series[:size + WINDOW_SIZE + HORIZON]

    X = np.array(X).reshape(-1, WINDOW_SIZE, 1) # [N, 10, 1]
    Y = np.array(Y) # [N, 3]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32), full_series


X_train, Y_train, full_series = generate_data(500)

# Initialize Model
INPUT_SIZE = 1 
HIDDEN_SIZE = 64
model = Seq2SeqQuantileForecaster(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred_all = model(X_train) 
    loss = pinball_loss(Y_train, y_pred_all, QUANTILES)
    loss.backward()
    optimizer.step()
    # Optional: Print loss
    # if (epoch + 1) % 50 == 0:
    #     print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- 4. Prediction and Plotting ---

# Get the historical data used for the final forecast
hist_end_idx = len(full_series) - HORIZON
historical_data = full_series[:hist_end_idx]

# Get the true future data
true_future = full_series[hist_end_idx:]

# Run the single forward pass
current_input = X_train[-1].unsqueeze(0) # [1, 10, 1]
with torch.no_grad():
    prediction = model(current_input).squeeze().numpy()

# Reshape the 9 outputs into a readable table
forecast_table = prediction.reshape(HORIZON, N_QUANTILES)
lower_bound = forecast_table[:, 0] # 5th percentile
median_forecast = forecast_table[:, 1] # 50th percentile
upper_bound = forecast_table[:, 2] # 95th percentile

# Create time indices for plotting
hist_time = np.arange(len(historical_data))
future_time = np.arange(len(historical_data), len(historical_data) + HORIZON)

# --- PLOTTING ---
plt.figure(figsize=(12, 6))

# 1. Plot Historical Data
plt.plot(hist_time, historical_data, label='Historical Sales', color='blue', linewidth=2)

# 2. Plot True Future Data
plt.plot(future_time, true_future, 'o--', label='True Future Sales', color='red', alpha=0.7)

# 3. Plot Point Forecast (Median)
plt.plot(future_time, median_forecast, 'D-', label='Point Forecast (Median)', color='orange', linewidth=2, markersize=5)

# 4. Plot Prediction Interval (90% PI)
plt.fill_between(
    future_time,
    lower_bound,
    upper_bound,
    color='orange',
    alpha=0.2,
    label='90% Prediction Interval (5%-95%)'
)

# Vertical line to separate history and forecast
plt.axvline(x=hist_end_idx - 1, color='gray', linestyle='--', linewidth=1)
plt.text(hist_end_idx - 1, plt.ylim()[0], 'Forecast Start', rotation=90, verticalalignment='bottom')

# Labels and Title
plt.title('Multi-Step Ahead Quantile Regression Forecast')
plt.xlabel('Time Step')
plt.ylabel('Sales/Value')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Display the numerical results again for comparison
print("\n--- Final Numerical Prediction ---")
print(f"Forecast Horizon: {HORIZON} steps")
print(f"Quantiles: {QUANTILES}")
print("-" * 50)
print(f"{'Step':<5} | {'True Value':<12} | {'Lower (5%)':<12} | {'Median (50%)':<12} | {'Upper (95%)':<12} | {'Interval Width':<15}")
print("-" * 50)
for i in range(HORIZON):
    true_val = true_future[i]
    lower = lower_bound[i]
    median = median_forecast[i]
    upper = upper_bound[i]
    width = upper - lower
    print(f"t+{i+1:<4} | {true_val:.3f}      | {lower:.3f}      | {median:.3f}      | {upper:.3f}      | {width:.3f}")

print("\n--- Plot Interpretation ---")
print("1. **Single Forecast Run:** The orange shaded area and line were generated by ONE single forward pass.")
print("2. **Uncertainty Growth:** The orange shaded area (the PI) should visibly get wider from t+1 to t+20, demonstrating that the Pinball Loss successfully trained the model to account for error accumulation.")
print("3. **Coverage:** Ideally, the red true values should mostly fall within the orange shaded area (the 90% PI).")