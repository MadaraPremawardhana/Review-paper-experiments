import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers

# --- Load dataset ---
data = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])

# Extract panel power values
power = data['Average actual panel production'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
power_scaled = scaler.fit_transform(power)

# --- Supervised framing (1-step lookback) ---
X = power_scaled[:-1]
y = power_scaled[1:]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MLP Model ---
mlp = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

mlp.compile(optimizer='adam', loss='mse')
mlp.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=0)

# --- Predict and inverse transform ---
y_pred_scaled = mlp.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test)

# --- Optional: Save predictions with timestamps ---
# Offset Date/Time to match y values
timestamps = data['Date/Time'][1:]  # y is 1-step ahead of X
_, X_test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
timestamps_test = timestamps.iloc[X_test_idx].reset_index(drop=True)

# Save to CSV
results_df = pd.DataFrame({
    'Date/Time': timestamps_test,
    'Actual Panel Power (watts)': y_test_actual.flatten(),
    'Predicted Panel Power (watts)': y_pred_actual.flatten()
})

results_df.to_csv('mlp_panel_power_predictions.csv', index=False)
print("✅ MLP predictions saved to 'mlp_panel_power_predictions.csv'")
