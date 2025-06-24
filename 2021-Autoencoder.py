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

# autoencoder_power_prediction.py
from tensorflow.keras import models, layers

# Input features are historical 'Panel power(watts)'
X = power_scaled

# Train/test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Autoencoder
input_dim = X_train.shape[1]
encoding_dim = input_dim // 2 or 1  # avoid 0

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(encoding_dim, activation='relu'),
    layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=8, validation_data=(X_test, X_test))

# Encoded feature output
encoder = models.Sequential([autoencoder.layers[0]])
X_encoded = encoder.predict(X_scaled)

print("Encoded features shape:", X_encoded.shape)

