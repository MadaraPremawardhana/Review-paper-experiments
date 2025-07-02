import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler

# --- Dempster-Shafer fusion function ---
def dempster_shafer_fusion(m1, m2):
    """
    Simple DS fusion assuming m1 and m2 are model predictions (beliefs)
    m1, m2: numpy arrays of shape (n,)
    Returns combined belief (numpy array)
    """
    # Normalize inputs between 0 and 1 (optional if already scaled)
    m1 = np.clip(m1, 0, 1)
    m2 = np.clip(m2, 0, 1)
    
    # Conflict coefficient
    K = m1 * (1 - m2) + (1 - m1) * m2
    # Avoid division by zero
    denominator = 1 - K
    denominator[denominator == 0] = 1e-6
    
    combined = (m1 * m2) / denominator
    return combined

# --- Load training data (2023) ---
data_2023 = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])
power_2023 = data_2023['Average actual panel production'].values.reshape(-1,1)

# --- Load 2024 data (to predict) ---
data_2024 = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])
power_2024 = data_2024['Average actual panel production'].values.reshape(-1,1) if 'Panel power(watts)' in data_2024 else None

# --- Normalize using scaler fit on 2023 ---
scaler = MinMaxScaler()
power_2023_scaled = scaler.fit_transform(power_2023)

# --- Create lag features function ---
def create_lag_features(data, lag=3):
    X = []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i].flatten())
    return np.array(X)

lag = 3

# Prepare training features and labels from 2023
X_train = create_lag_features(power_2023_scaled, lag)
y_train = power_2023_scaled[lag:]

# Train physical model (Linear Regression)
physical_model = LinearRegression()
physical_model.fit(X_train, y_train)

# Train data-driven model (XGBoost)
data_model = XGBRegressor(objective='reg:squarederror', random_state=42)
data_model.fit(X_train, y_train.ravel())

# Prepare 2024 data for prediction
# If 2024 power missing, just prepare lagged sequences from previous known data (need last lag values of 2023 + 2024 time steps)
# We'll do recursive prediction over 2024 timestamps

# Concatenate 2023 and 2024 power scaled for lagging:
if power_2024 is None:
    # Create empty array to hold predictions
    full_power_scaled = np.concatenate([power_2023_scaled, np.zeros((len(data_2024),1))])
else:
    power_2024_scaled = scaler.transform(power_2024)
    full_power_scaled = np.concatenate([power_2023_scaled, power_2024_scaled])

# Recursive prediction for 2024
predictions_scaled = []

# Start from last lag values of 2023
history = list(power_2023_scaled[-lag:].flatten())

for i in range(len(data_2024)):
    X_input = np.array(history[-lag:]).reshape(1, -1)
    pred_phys = physical_model.predict(X_input)[0][0]
    pred_data = data_model.predict(X_input)[0]
    
    # Fuse with Dempster-Shafer theory
    pred_fused = dempster_shafer_fusion(np.array([pred_phys]), np.array([pred_data]))[0]
    
    # Append prediction
    predictions_scaled.append(pred_fused)
    # Add to history for next step
    history.append(pred_fused)

predictions_scaled = np.array(predictions_scaled).reshape(-1,1)

# Inverse scale to watts
predictions_watts = scaler.inverse_transform(predictions_scaled)

# Save predictions
results_df = pd.DataFrame({
    'Date/Time': data_2024['Date/Time'],
    'Predicted Panel power(watts)': predictions_watts.flatten()
})
results_df.to_csv('2024_power_predictions.csv', index=False)
print("✅ 2024 predictions saved to '2024_power_predictions.csv'")
