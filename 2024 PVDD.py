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

# pv_dt_power_model.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Use past values to predict next
X = np.arange(len(power_scaled)).reshape(-1,1)  # Time index as input
y = power_scaled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Physical model: linear trend
physical_model = LinearRegression()
physical_model.fit(X_train, y_train)

# Data model: random forest
data_model = RandomForestRegressor()
data_model.fit(X_train, y_train.ravel())

# Combine
physical_pred = physical_model.predict(X_test)
data_pred = data_model.predict(X_test)

final_pred = (0.5 * physical_pred) + (0.5 * data_pred)
