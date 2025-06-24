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

# cnn_transformer_power_prediction.py
from tensorflow.keras import models, layers

# Create sequences
X_seq, y_seq = create_sequences(power_scaled, seq_length=8)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Expand dims for Conv1D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

input_layer = layers.Input(shape=(X_train.shape[1], 1))
cnn_out = layers.Conv1D(32, kernel_size=3, activation='relu')(input_layer)
cnn_out = layers.MaxPooling1D(pool_size=2)(cnn_out)
cnn_out = layers.Flatten()(cnn_out)

transformer_out = layers.Dense(64, activation='relu')(cnn_out)
transformer_out = layers.Dense(32, activation='relu')(transformer_out)

output = layers.Dense(1)(transformer_out)

model = models.Model(inputs=input_layer, outputs=output)

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
