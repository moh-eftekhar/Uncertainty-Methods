import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# Use keras.layers via the imported keras to avoid editor linter issues with tensorflow.keras
layers = keras.layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. Define Model Parameters ---
N_PAST_STEPS = 24       # How many past hours to use as input
N_HORIZONS = 20        # How many future hours to predict
N_FEATURES = 1        # We are using 1 feature (the energy value)
QUANTILES = [0.05, 0.5, 0.95] # The 3 quantiles we want to predict
N_QUANTILES = len(QUANTILES)

# --- 2. Create Sample Data (Sine Wave with Noise) ---
def create_time_series_data(n_samples, n_past, n_future):
    """Creates a sine wave dataset for time series forecasting."""
    X, y = [], []
    for i in range(n_samples):
        # Create a sine wave segment
        x_start = i * 0.1
        t = np.linspace(x_start, x_start + (n_past + n_future) * 0.1, n_past + n_future)
        wave = np.sin(t) + 0.1 * np.random.randn(n_past + n_future)
        
        # Split into past (X) and future (y)
        X.append(wave[:n_past])
        y.append(wave[n_past:])
        
    X = np.array(X).reshape(-1, n_past, 1)
    y = np.array(y).reshape(-1, n_future)
    return X, y

print("Creating sample data...")
X_data, y_data = create_time_series_data(1000, N_PAST_STEPS, N_HORIZONS)

# Normalize the data (very important for neural nets)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X_data.reshape(-1, N_PAST_STEPS)).reshape(X_data.shape)
y_scaled = scaler_y.fit_transform(y_data)

# Split data
X_train, y_train = X_scaled[:800], y_scaled[:800]
X_test, y_test = X_scaled[800:], y_scaled[800:]

print(f"X_train shape: {X_train.shape}") # (800, 24, 1)
print(f"y_train shape: {y_train.shape}") # (800, 6)


# --- 3. Define the Pinball Loss Function ---
# This is the "magic" that trains the model to predict quantiles.
# y_pred will have shape (batch, horizons, quantiles)
# y_true will have shape (batch, horizons)

def pinball_loss(y_true, y_pred):
    # y_true is (batch_size, N_HORIZONS)
    # y_pred is (batch_size, N_HORIZONS, N_QUANTILES)
    
    # Expand y_true to match y_pred's shape for easier calculation
    # Shape becomes (batch_size, N_HORIZONS, 1)
    y_true_expanded = tf.expand_dims(y_true, -1)
    
    # Calculate the error
    # Shape is (batch_size, N_HORIZONS, N_QUANTILES)
    error = y_true_expanded - y_pred
    
    # Calculate the pinball loss for each quantile
    loss = 0.0
    for i, q in enumerate(QUANTILES):
        # Get the error for the i-th quantile
        q_error = error[..., i] 
        
        # This is the core pinball loss formula
        q_loss = tf.reduce_mean(tf.maximum(q * q_error, (q - 1) * q_error))
        loss += q_loss
        
    return loss

# --- 4. Build the Neural Network Model ---
# We will build ONE model that outputs all quantiles for all horizons.
N_OUTPUTS = N_HORIZONS * N_QUANTILES # 6 horizons * 3 quantiles = 18 outputs

model = keras.Sequential([
    layers.LSTM(64, input_shape=(N_PAST_STEPS, N_FEATURES), return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    # The crucial output layer: 18 neurons
    layers.Dense(N_OUTPUTS), 
    # Reshape the 18 outputs into a (6, 3) structure for the loss function
    layers.Reshape((N_HORIZONS, N_QUANTILES)) 
])

model.summary()

# --- 5. Compile and Train the Model ---
model.compile(loss=pinball_loss, optimizer='adam')

print("\nTraining the model...")
history = model.fit(
    X_train, 
    y_train,  # y_train is (800, 6), matching y_true in the loss
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# --- 6. Make Predictions and Plot ---
print("\nMaking predictions...")
# Get predictions (shape: 200, 6, 3)
y_pred_scaled = model.predict(X_test)

# We must un-normalize both the predictions and the true values
# Un-normalize y_true (the real data)
y_test_unscaled = scaler_y.inverse_transform(y_test)

# Un-normalize y_pred (the forecasts)
# We have to reshape to 2D for the scaler, then reshape back
y_pred_unscaled = scaler_y.inverse_transform(
    y_pred_scaled.reshape(-1, N_HORIZONS)
).reshape(y_pred_scaled.shape)


# --- 7. Plot the Results for one sample ---
sample_idx = 42
plt.figure(figsize=(12, 6))

# Plot the 6-hour future (true value)
plt.plot(
    range(N_HORIZONS), 
    y_test_unscaled[sample_idx], 
    'o-', 
    label="Actual Future Value", 
    color='blue'
)

# Plot the 50% quantile (median)
plt.plot(
    range(N_HORIZONS), 
    y_pred_unscaled[sample_idx, :, QUANTILES.index(0.5)], 
    'x--', 
    label="Predicted Median (0.5 Quantile)", 
    color='orange'
)

# Plot the 10% and 90% quantiles
q_low = y_pred_unscaled[sample_idx, :, QUANTILES.index(0.05)]
q_high = y_pred_unscaled[sample_idx, :, QUANTILES.index(0.95)]

# Fill the 80% confidence interval
plt.fill_between(
    range(N_HORIZONS), 
    q_low, 
    q_high, 
    alpha=0.2, 
    color='orange', 
    label="5%-95% Prediction Interval"
)

plt.title("Multi-Step Quantile Forecast (Keras)")
plt.xlabel("Forecast Horizon (Hours Ahead)")
plt.ylabel("Scaled Energy Value")
plt.legend()
plt.grid(True)
plt.show()