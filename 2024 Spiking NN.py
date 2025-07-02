import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge

# --- Load 2023 data ---
data_2023 = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])
power_2023 = data_2023['Average actual panel production'].values.reshape(-1, 1)

# --- Load 2024 data (dates only, no power) ---
data_2024 = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])

# --- Scale 2023 power ---
scaler = MinMaxScaler()
power_2023_scaled = scaler.fit_transform(power_2023)

# --- Simple Echo State Network implementation ---
class SimpleESN:
    def __init__(self, n_reservoir=100, sparsity=0.1, random_state=42):
        np.random.seed(random_state)
        self.n_reservoir = n_reservoir
        self.Win = np.random.rand(n_reservoir, 1) - 0.5
        self.W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        mask = np.random.rand(*self.W.shape) < sparsity
        self.W *= mask

    def fit(self, X, y):
        states = []
        x = np.zeros((self.n_reservoir,))
        for u in X:
            x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, x))
            states.append(x.copy())
        self.ridge = Ridge(alpha=1e-6)
        self.ridge.fit(states, y)

    def predict(self, X):
        preds = []
        x = np.zeros((self.n_reservoir,))
        for u in X:
            x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, x))
            preds.append(self.ridge.predict(x.reshape(1, -1))[0])
        return np.array(preds)

# --- Prepare supervised data (one-step lag) from 2023 ---
X_2023 = power_2023_scaled[:-1]
y_2023 = power_2023_scaled[1:]

# --- Train ESN ---
esn = SimpleESN()
esn.fit(X_2023, y_2023)

# --- Recursive prediction for 2024 ---

# Start from last known 2023 scaled value
last_input = power_2023_scaled[-1].reshape(1, 1)  # shape (1,1)
predictions_scaled = []

# For each time step in 2024, predict next power, then use it as input for next step
for _ in range(len(data_2024)):
    pred_scaled = esn.predict(last_input.reshape(1,1))[0]
    predictions_scaled.append(pred_scaled)
    last_input = np.array([[pred_scaled]])

predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)

# --- Inverse scale to original units ---
predictions_watts = scaler.inverse_transform(predictions_scaled)

# --- Save 2024 predictions ---
results_df = pd.DataFrame({
    'Date/Time': data_2024['Date/Time'],
    'Predicted Panel power(watts)': predictions_watts.flatten()
})

results_df.to_csv('2024_esn_power_predictions.csv', index=False)
print("✅ 2024 ESN predictions saved to '2024_esn_power_predictions.csv'")
