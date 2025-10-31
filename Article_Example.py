import numpy as np
from sklearn.linear_model import QuantileRegressor
from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt  # Import for plotting

def get_errors_and_k(series, max_lead_time):
    """
    [cite_start]This function implements the data collection from Section 4 [cite: 161-170].
    It fits a model ONCE, then collects all k-step-ahead
    fit errors (y) and their corresponding k-features (X).
    """
    model = SimpleExpSmoothing(series, initialization_method="estimated").fit()
    smoothed_level = model.level
    
    y_errors = []
    X_features = []
    
    for k in range(1, max_lead_time + 1):
        for t in range(len(series) - k):
            forecast = smoothed_level[t] 
            actual = series[t + k]
            error = actual - forecast
            
            y_errors.append(error)
            # Our features are k and k^2
            X_features.append([k, k**2]) 
            
    return np.array(X_features), np.array(y_errors)

# --- 1. Create a sample time series ---
np.random.seed(42)
series = np.arange(100) + np.random.normal(0, 5, 100) + 20
max_k = 18

# --- 2. Get the X and y data ---
X_data, y_data = get_errors_and_k(series, max_k)

# --- 3. Run the Quantile Regressions ---
qr_95 = QuantileRegressor(quantile=0.95, alpha=0, solver='highs')
qr_95.fit(X_data, y_data)

qr_05 = QuantileRegressor(quantile=0.05, alpha=0, solver='highs')
qr_05.fit(X_data, y_data)

# --- 4. Print the final formulas (the models) ---
a_95, (b_95, c_95) = qr_95.intercept_, qr_95.coef_
a_05, (b_05, c_05) = qr_05.intercept_, qr_05.coef_

print("--- Your Final Prediction Interval Formulas ---")
print(f"Upper Bound (95%): Q_fe(0.95) = {a_95:.2f} + {b_95:.2f}*k + {c_95:.2f}*k^2")
print(f"Lower Bound ( 5%): Q_fe(0.05) = {a_05:.2f} + {b_05:.2f}*k + {c_05:.2f}*k^2")
print("\n" + "="*30 + "\n")


# --- 5. NEW: Calculate Bounds for a New Forecast ---

# Let's say we have a new forecast for k=12
# And our "best guess" (point forecast) is 150
k_new = 12
point_forecast = 150

# # === DYNAMIC PART ===

# # In your real code, you would get these values from
# # your own forecasting process.

# # 1. Decide what lead time you want to forecast for.
# k_new = 6  # Example: We want to forecast 6 steps ahead

# # 2. Get the "best guess" forecast from your *original* smoothing model.
# #    (We'll re-use the 'model' we fit in Step 1)
# model_from_step1 = SimpleExpSmoothing(series, initialization_method="estimated").fit()
# point_forecast = model_from_step1.forecast(k_new)[-1] # Get the last value of the forecast

# # === END DYNAMIC PART ===


# Create the X data for k=12. It must be a 2D array: [[k, k^2]]
X_new = np.array([[k_new, k_new**2]])

# Use the fitted models to predict the error bounds
upper_error = qr_95.predict(X_new)[0]
lower_error = qr_05.predict(X_new)[0]

# Add the error bounds to our point forecast
upper_bound = point_forecast + upper_error
lower_bound = point_forecast + lower_error

print("--- New Forecast Calculation (k=12) ---")
print(f"Point Forecast:   {point_forecast:.2f}")
print(f"Predicted Upper Error: {upper_error:+.2f}")
print(f"Predicted Lower Error: {lower_error:+.2f}")
print(f"90% Prediction Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
print("\n" + "="*30 + "\n")


# --- 6. NEW: Plot the Results ---
print("Generating plot...")

# To draw smooth lines, we need many k-points
k_plot = np.linspace(1, max_k, 100)
X_plot = np.vstack([k_plot, k_plot**2]).T

# Get the predicted quantile lines for all these k-points
y_plot_95 = qr_95.predict(X_plot)
y_plot_05 = qr_05.predict(X_plot)

# Create the plot 📈
plt.figure(figsize=(10, 6))

# Plot the raw error data as a scatter plot
# X_data[:, 0] is just the 'k' column
plt.scatter(X_data[:, 0], y_data, alpha=0.1, label="Raw Fit Errors")

# Plot the 95th Percentile line
plt.plot(k_plot, y_plot_95, color='red', linestyle='--', 
         label="95th Percentile Model")

# Plot the 5th Percentile line
plt.plot(k_plot, y_plot_05, color='blue', linestyle='--', 
         label="5th Percentile Model")

plt.title("Quantile Regression Model of Forecast Errors")
plt.xlabel("Lead Time (k)")
plt.ylabel("Forecast Error (Actual - Forecast)")
plt.legend()
plt.grid(True, which='both', linestyle=':', linewidth=0.5)
plt.show()

print("Plot complete.")