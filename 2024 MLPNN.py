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

# mlp_power_prediction.py
from tensorflow.keras import models, layers

# Supervised framing: use previous value to predict next
X = power_scaled[:-1]
y = power_scaled[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLP Model
mlp = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

mlp.compile(optimizer='adam', loss='mse')
mlp.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

y_pred = mlp.predict(X_test)
