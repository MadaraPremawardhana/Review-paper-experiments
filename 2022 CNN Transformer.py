import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers, Input

# ========================
# --- Utility function ---
# ========================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# ==============================
# --- Load and preprocess 2023 data ---
# ==============================
train_data = pd.read_csv(
    '2023 data for ML benchmark training.csv',
    parse_dates=['Date/Time']
)

train_power = train_data['Average actual panel production'].values.reshape(-1, 1)

# Scale data (fit ONLY on training data)
scaler = MinMaxScaler()
train_power_scaled = scaler.fit_transform(train_power)

# Create sequences
seq_length = 8
X_seq, y_seq = create_sequences(train_power_scaled, seq_length)

# =========================================
# --- TIME-BASED train/test split (FIXED) ---
# =========================================
split_index = int(len(X_seq) * 0.8)

X_train = X_seq[:split_index]
X_test = X_seq[split_index:]

y_train = y_seq[:split_index]
y_test = y_seq[split_index:]

# Reshape for Conv1D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ========================
# --- Build model ---
# ========================
input_layer = Input(shape=(seq_length, 1))

x = layers.Conv1D(32, kernel_size=3, activation='relu')(input_layer)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
output = layers.Dense(1)(x)

model = models.Model(inputs=input_layer, outputs=output)

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test)
)

# ========================
# --- Predict for 2024 ---
# ========================

# IMPORTANT FIX: use actual 2024 file here
predict_data = pd.read_csv(
    '2024 data for ML benchmark.csv',
    parse_dates=['Date/Time']
)

predict_power = predict_data['Average actual panel production'].values.reshape(-1, 1)

# Use SAME scaler (no refit!)
predict_power_scaled = scaler.transform(predict_power)

# Create sequences
X_2024_seq, _ = create_sequences(predict_power_scaled, seq_length=seq_length)

X_2024_seq = X_2024_seq.reshape((X_2024_seq.shape[0], seq_length, 1))

# Predict
pred_scaled = model.predict(X_2024_seq)

# Convert back to real values
pred_actual = scaler.inverse_transform(pred_scaled)

# Save results
timestamps = predict_data['Date/Time'][seq_length:].reset_index(drop=True)

results_df = pd.DataFrame({
    'Date/Time': timestamps,
    'Predicted Panel Power (watts)': pred_actual.flatten()
})

results_df.to_csv('predicted_2024_panel_power.csv', index=False)

print("✅ Predictions saved to 'predicted_2024_panel_power.csv'")
