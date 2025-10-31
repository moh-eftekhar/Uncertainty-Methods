import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.multioutput import MultiOutputRegressor

## ------------------------------------------------------------------
## 1. CREATE SYNTHETIC TIME SERIES DATA
## ------------------------------------------------------------------
# We'll create a sine wave with some noise. This is easy to learn.
total_points = 500
time = np.arange(total_points)
data = np.sin(time * 0.1) + np.random.randn(total_points) * 0.2
data = data.astype(np.float32)

# Plot our data to see what it looks like
plt.figure(figsize=(15, 3))
plt.title("Original Time Series Data")
plt.plot(time, data)
plt.show()

## ------------------------------------------------------------------
## 2. DEFINE FORECASTING PARAMETERS
## ------------------------------------------------------------------
# We will look at the past 10 steps to predict the next 5 steps.
N_LAGS = 10       # Number of past steps to use as features (X)
N_FORECASTS = 5   # Number of future steps to predict (y)

## ------------------------------------------------------------------
## 3. HELPER FUNCTION TO FORMAT THE DATA
## ------------------------------------------------------------------
# This is the most important part of the concept.
# We need to transform our 1D array [1, 2, 3, ...] into
# an (X, y) matrix for machine learning.
#
# X will be the "lags" (e.g., [1, 2, 3, 4, 5])
# y will be the "forecasts" (e.g., [6, 7])

def create_dataset(data, n_lags, n_forecasts):
    """
    Creates a supervised learning dataset from a time series.
    """
    X, y = [], []
    # We slide a "window" across the data
    for i in range(len(data) - n_lags - n_forecasts + 1):
        # The 'X' part is the n_lags of historical data
        X.append(data[i : i + n_lags])
        # The 'y' part is the n_forecasts of future data
        y.append(data[i + n_lags : i + n_lags + n_forecasts])
    
    return np.array(X), np.array(y)

# Create the dataset
X, y = create_dataset(data, N_LAGS, N_FORECASTS)

print(f"Original data shape: {data.shape}")
print(f"Transformed X shape: {X.shape}")
print(f"Transformed y shape: {y.shape}")
print("\nExample:")
print(f"First X sample (lags):\n {X[0]}")
print(f"First y sample (forecasts):\n {y[0]}")

## ------------------------------------------------------------------
## 4. SPLIT DATA AND TRAIN MODELS
## ------------------------------------------------------------------

# We'll train on all data except the very last sample,
# which we will use for our final prediction.
X_train = X[:-1]
y_train = y[:-1]

# This is the single input we will use to make a new forecast
X_test_input = X[-1:] 
# This is the "ground truth" for that input, so we can compare
y_test_true = y[-1]   

print(f"\nInput for prediction (shape {X_test_input.shape}):\n {X_test_input[0]}")
print(f"True future to compare (shape {y_test_true.shape}):\n {y_test_true}")


# --- Create the models ---
# We need to create a model for the mean, and for the upper/lower bounds.
# We use MultiOutputRegressor to "wrap" our simple linear models.
# This wrapper handles the multi-step magic for us.

# Model 1: Mean prediction
lr = LinearRegression()
model_mean = MultiOutputRegressor(lr)

# Model 2: Lower bound (5th percentile)
# We set alpha=0 to turn off regularization for a "purer" quantile model
qr_low = QuantileRegressor(quantile=0.05, alpha=0, solver='highs')
model_low = MultiOutputRegressor(qr_low)

# Model 3: Upper bound (95th percentile)
qr_high = QuantileRegressor(quantile=0.95, alpha=0, solver='highs')
model_high = MultiOutputRegressor(qr_high)

# --- Fit all three models ---
print("\nTraining models...")
model_mean.fit(X_train, y_train)
model_low.fit(X_train, y_train)
model_high.fit(X_train, y_train)
print("Models trained.")

## ------------------------------------------------------------------
## 5. MAKE MULTI-STEP PREDICTIONS
## ------------------------------------------------------------------

# Use our single test sample to predict the next 5 steps
y_pred_mean = model_mean.predict(X_test_input)
y_pred_low = model_low.predict(X_test_input)
y_pred_high = model_high.predict(X_test_input)

# The shape of these outputs is (1, 5) -- one prediction for 5 steps
print(f"\nMean prediction output:\n {y_pred_mean.flatten()}")
print(f"Lower bound output:\n {y_pred_low.flatten()}")
print(f"Upper bound output:\n {y_pred_high.flatten()}")


## ------------------------------------------------------------------
## 6. PLOT THE RESULTS
## ------------------------------------------------------------------

# Create time axes for plotting
# time_axis_lags will be steps 0 to 9
# time_axis_forecast will be steps 10 to 14
time_axis_lags = np.arange(0, N_LAGS)
time_axis_forecast = np.arange(N_LAGS, N_LAGS + N_FORECASTS)

plt.figure(figsize=(15, 7))

# Plot the historical data that was used as input
plt.plot(time_axis_lags, X_test_input.flatten(), 'ko-', label='Input Data (Last 10 steps)')

# Plot the "true" future data that actually happened
plt.plot(time_axis_forecast, y_test_true.flatten(), 'bo--', 
         label=f'True Future (Next {N_FORECASTS} steps)')

# Plot the model's mean prediction
plt.plot(time_axis_forecast, y_pred_mean.flatten(), 'g^-', 
         label='Mean Prediction (Multi-step)')

# Plot the prediction interval (the quantiles)
plt.plot(time_axis_forecast, y_pred_low.flatten(), 'r:', label='5th Percentile')
plt.plot(time_axis_forecast, y_pred_high.flatten(), 'r:', label='95th Percentile')

# Fill the area between the quantiles to show the 90% prediction interval
plt.fill_between(time_axis_forecast, y_pred_low.flatten(), y_pred_high.flatten(), 
                 color='red', alpha=0.1, label='90% Prediction Interval')

plt.title("Multi-Step Forecasting with QuantileRegressor", fontsize=16)
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend(fontsize=11)
plt.grid(True)
plt.show()