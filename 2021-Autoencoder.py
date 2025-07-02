import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- Load training data (2023) ---
train_data = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])
train_power = train_data['Average actual panel production'].values.reshape(-1, 1)

# --- Normalize training data ---
scaler = MinMaxScaler()
train_power_scaled = scaler.fit_transform(train_power)

# --- Split for training/validation ---
X_train, X_test = train_test_split(train_power_scaled, test_size=0.2, random_state=42)

# --- Autoencoder using Functional API ---
input_dim = X_train.shape[1]
encoding_dim = max(input_dim // 2, 1)  # ensure at least 1

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='linear')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

# --- Train autoencoder ---
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=8,
                validation_data=(X_test, X_test))

# --- Encoder model to get compressed features ---
encoder = Model(inputs=input_layer, outputs=encoded)

# --- Load 2024 data ---
predict_data = pd.read_csv('2023 data for ML benchmark training.csv', parse_dates=['Date/Time'])
predict_power = predict_data['Average actual panel production'].values.reshape(-1, 1)

# --- Normalize 2024 data using 2023 scaler ---
predict_power_scaled = scaler.transform(predict_power)

# --- Get encoded/compressed features ---
encoded_2024 = encoder.predict(predict_power_scaled)

print("Encoded features shape for 2024 data:", encoded_2024.shape)

# --- Save output ---
predict_data['Encoded_feature'] = encoded_2024
predict_data.to_csv('2024_data_with_encoded_features.csv', index=False)
