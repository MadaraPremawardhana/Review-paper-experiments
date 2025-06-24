import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('sampleInput.csv', parse_dates=['Date/Time'])

# Only use 'Panel power(watts)'
power = data['Panel power(watts)'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
power_scaled = scaler.fit_transform(power)

# meta_lstm_power_prediction.py
from tensorflow.keras import models, layers

# Create sequences
def create_sequences(X, seq_length=4):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(X[i+seq_length])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(power_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build LSTM
def build_lstm():
    model = models.Sequential([
        layers.LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

models_list = [build_lstm(), build_lstm(), build_lstm()]

for model in models_list:
    model.fit(X_train, y_train, epochs=30, batch_size=8, validation_data=(X_test, y_test))

# Meta-prediction: average
predictions = np.array([model.predict(X_test).flatten() for model in models_list])
meta_prediction = np.mean(predictions, axis=0)
