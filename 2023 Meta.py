import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models, layers

# --- Load dataset ---
data = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])

# Extract target feature
power = data['Average actual panel production'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
power_scaled = scaler.fit_transform(power)

# --- Create sequences ---
def create_sequences(X, seq_length=4):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(X[i + seq_length])
    return np.array(Xs), np.array(ys)

seq_length = 4
X_seq, y_seq = create_sequences(power_scaled, seq_length)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# --- Define LSTM model builder ---
def build_lstm():
    model = models.Sequential([
        layers.LSTM(32, input_shape=(seq_length, 1)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Train multiple LSTM models (meta-learning ensemble) ---
models_list = [build_lstm(), build_lstm(), build_lstm()]

for i, model in enumerate(models_list):
    print(f"Training model {i+1}...")
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test), verbose=0)

# --- Ensemble prediction: average of all model predictions ---
predictions = np.array([model.predict(X_test).flatten() for model in models_list])
meta_prediction_scaled = np.mean(predictions, axis=0)

# --- Inverse transform to get predicted watt values ---
meta_prediction_actual = scaler.inverse_transform(meta_prediction_scaled.reshape(-1, 1))

# --- Optional: Save predictions with corresponding timestamps ---
timestamps = data['Date/Time'][seq_length + len(X_train):].reset_index(drop=True)
results_df = pd.DataFrame({
    'Date/Time': timestamps[:len(meta_prediction_actual)],  # ensure length match
    'Predicted Panel Power (watts)': meta_prediction_actual.flatten()
})

results_df.to_csv('meta_lstm_predictions.csv', index=False)
print("✅ Predictions saved to 'meta_lstm_predictions.csv'")
