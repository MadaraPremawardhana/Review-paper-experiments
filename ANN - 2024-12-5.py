import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# Load and preprocess data
file_path = 'D:/pre-digital-twin-v4/Data/2023_Jan to June_KB_ hourly.csv'
data = pd.read_csv(file_path)

data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%d/%m/%Y %H:%M')
data.sort_values(by='Date/Time', inplace=True)

data = data.dropna()
data['Average panel production'] = data['Average panel production'].clip(lower=0)

data['Hour'] = data['Date/Time'].dt.hour
data['Day'] = data['Date/Time'].dt.day
data['Month'] = data['Date/Time'].dt.month

predefined_weather = [
    # Thunderstorm group (2xx)
    'thunderstorm with light rain', 'thunderstorm with rain', 'thunderstorm with heavy rain',
    'light thunderstorm', 'thunderstorm', 'heavy thunderstorm', 'ragged thunderstorm',
    'thunderstorm with light drizzle', 'thunderstorm with drizzle', 'thunderstorm with heavy drizzle',

    # Drizzle group (3xx)
    'light intensity drizzle', 'drizzle', 'heavy intensity drizzle', 'light intensity drizzle rain',
    'drizzle rain', 'heavy intensity drizzle rain', 'shower rain and drizzle',
    'heavy shower rain and drizzle', 'shower drizzle',

    # Rain group (5xx)
    'light rain', 'moderate rain', 'heavy intensity rain', 'very heavy rain', 'extreme rain',
    'freezing rain', 'light intensity shower rain', 'shower rain', 'heavy intensity shower rain',
    'ragged shower rain',

    # Snow group (6xx)
    'light snow', 'snow', 'heavy snow', 'sleet', 'light shower sleet', 'shower sleet',
    'light rain and snow', 'rain and snow', 'light shower snow', 'shower snow', 'heavy shower snow',

    # Atmosphere group (7xx)
    'mist', 'smoke', 'haze', 'sand/dust whirls', 'fog', 'sand', 'dust', 'volcanic ash',
    'squalls', 'tornado',

    # Clear group (800)
    'clear sky',

    # Clouds group (80x)
    'few clouds: 11-25%', 'scattered clouds: 25-50%', 'broken clouds: 51-84%', 'overcast clouds', 'few clouds', 'scattered clouds', 'broken clouds', 'overcast clouds'
]

label_encoder = LabelEncoder()
label_encoder.fit(predefined_weather)

def encode_weather(weather, label_encoder):
    try:
        return label_encoder.transform([weather.lower()])[0]
    except ValueError:
        return -1  

data['WeatherDescription'] = data['WeatherDescription'].apply(lambda x: encode_weather(x, label_encoder))

features = ['Hour', 'Day', 'Month', 'Latitude', 'Longitude', 'WeatherDescription',
            'Wind velcity (m/s)', 'Wind direction(degrees from north)', 'Cloud cover percentage',
            'Atmospheric temperature(Kelvin)', 'GHI']
target = 'Average panel production'

X = data[features].values
y = data[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = y.reshape(-1, 1)
y = scaler_y.fit_transform(y)

# Split data for ANN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

with SuppressOutput():
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the ANN model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

# Make predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Prediction function
def predict_instance(instance):
    instance = np.array(instance).reshape(1, -1)
    instance_scaled = scaler_X.transform(instance)
    prediction = model.predict(instance_scaled, verbose=0)
    predicted_output = scaler_y.inverse_transform(prediction)
    return predicted_output[0][0]

encode_weather = lambda weather: label_encoder.transform([weather])[0]

# Example usage
instances = [[1,29,0,51.353036,-2.132106,encode_weather('overcast clouds'),3.17,188.0,100.0,283.92,0.0],[1,29,1,51.353036,-2.132106,encode_weather('overcast clouds'),3.9,193.0,100.0,284.01,0.0],[1,29,2,51.353036,-2.132106,encode_weather('overcast clouds'),4.85,202.0,100.0,284.22,0.0],[1,29,3,51.353036,-2.132106,encode_weather('overcast clouds'),6.06,209.0,100.0,284.34,0.0],[1,29,4,51.353036,-2.132106,encode_weather('overcast clouds'),4.21,198.0,100.0,284.3,0.0],[1,29,5,51.353036,-2.132106,encode_weather('overcast clouds'),3.37,174.0,100.0,284.29,0.0],[1,29,6,51.353036,-2.132106,encode_weather('overcast clouds'),3.85,179.0,100.0,284.27,0.0],[1,29,7,51.353036,-2.132106,encode_weather('overcast clouds'),4.58,177.0,100.0,284.32,0.0],[1,29,8,51.353036,-2.132106,encode_weather('overcast clouds'),4.59,176.0,100.0,284.23,0.0],[1,29,9,51.353036,-2.132106,encode_weather('overcast clouds'),5.25,173.0,100.0,284.37,0.0],[1,29,10,51.353036,-2.132106,encode_weather('overcast clouds'),5.93,181.0,100.0,284.28,0.0],[1,29,11,51.353036,-2.132106,encode_weather('overcast clouds'),6.38,189.0,100.0,284.28,0.0],[1,29,12,51.353036,-2.132106,encode_weather('overcast clouds'),7.02,180.0,98.0,284.7,0.0],[1,29,13,51.353036,-2.132106,encode_weather('overcast clouds'),6.68,184.0,100.0,284.9,0.0],[1,29,14,51.353036,-2.132106,encode_weather('overcast clouds'),6.67,188.0,100.0,284.95,0.0],[1,29,15,51.353036,-2.132106,encode_weather('overcast clouds'),6.71,188.0,100.0,284.82,0.0],[1,29,16,51.353036,-2.132106,encode_weather('overcast clouds'),6.47,188.0,99.0,284.72,0.0],[1,29,17,51.353036,-2.132106,encode_weather('overcast clouds'),6.14,182.0,99.0,284.55,0.0],[1,29,18,51.353036,-2.132106,encode_weather('overcast clouds'),6.61,178.0,99.0,284.53,0.0],[1,29,19,51.353036,-2.132106,encode_weather('overcast clouds'),6.73,180.0,100.0,284.99,0.0],[1,29,20,51.353036,-2.132106,encode_weather('overcast clouds'),7.38,188.0,100.0,285.24,0.0],[1,29,21,51.353036,-2.132106,encode_weather('overcast clouds'),6.98,198.0,100.0,285.41,0.0],[1,29,22,51.353036,-2.132106,encode_weather('overcast clouds'),7.74,201.0,100.0,284.95,0.0],[1,29,23,51.353036,-2.132106,encode_weather('overcast clouds'),8.26,201.0,100.0,284.27,0.0],[1,30,0,51.353036,-2.132106,encode_weather('overcast clouds'),7.38,203.0,100.0,283.58,0.0],[1,30,1,51.353036,-2.132106,encode_weather('overcast clouds'),7.26,223.0,100.0,282.97,0.0],[1,30,2,51.353036,-2.132106,encode_weather('overcast clouds'),7.84,246.0,100.0,282.05,0.0],[1,30,3,51.353036,-2.132106,encode_weather('overcast clouds'),9.07,274.0,98.0,280.86,0.0],[1,30,4,51.353036,-2.132106,encode_weather('overcast clouds'),8.44,276.0,94.0,280.62,0.0],[1,30,5,51.353036,-2.132106,encode_weather('overcast clouds'),8.22,283.0,95.0,280.53,0.0],[1,30,6,51.353036,-2.132106,encode_weather('overcast clouds'),7.87,283.0,98.0,280.43,0.0],[1,30,7,51.353036,-2.132106,encode_weather('overcast clouds'),6.4,279.0,100.0,280.22,0.0],[1,30,8,51.353036,-2.132106,encode_weather('overcast clouds'),5.47,277.0,100.0,280.07,0.0],[1,30,9,51.353036,-2.132106,encode_weather('overcast clouds'),5.5,284.0,100.0,280.17,0.0],[1,30,10,51.353036,-2.132106,encode_weather('overcast clouds'),5.38,289.0,100.0,280.77,0.0],[1,30,11,51.353036,-2.132106,encode_weather('overcast clouds'),4.78,288.0,100.0,280.75,0.0],[1,30,12,51.353036,-2.132106,encode_weather('overcast clouds'),4.3,295.0,100.0,280.83,0.0],[1,30,13,51.353036,-2.132106,encode_weather('overcast clouds'),4.5,299.0,100.0,280.93,0.0],[1,30,14,51.353036,-2.132106,encode_weather('overcast clouds'),3.39,313.0,100.0,281.22,0.0],[1,30,15,51.353036,-2.132106,encode_weather('overcast clouds'),3.12,302.0,99.0,281.0,0.0],[1,30,16,51.353036,-2.132106,encode_weather('overcast clouds'),2.16,309.0,99.0,280.84,0.0],[1,30,17,51.353036,-2.132106,encode_weather('overcast clouds'),1.2,318.0,99.0,280.69,0.0],[1,30,18,51.353036,-2.132106,encode_weather('overcast clouds'),0.87,352.0,99.0,280.37,0.0],[1,30,19,51.353036,-2.132106,encode_weather('overcast clouds'),1.42,253.0,100.0,280.07,0.0],[1,30,20,51.353036,-2.132106,encode_weather('overcast clouds'),1.53,242.0,100.0,280.01,0.0],[1,30,21,51.353036,-2.132106,encode_weather('overcast clouds'),0.97,228.0,100.0,279.47,0.0],[1,30,22,51.353036,-2.132106,encode_weather('overcast clouds'),1.78,246.0,98.0,279.32,0.0],[1,30,23,51.353036,-2.132106,encode_weather('broken clouds'),1.5,268.0,80.0,278.34,0.0],[1,31,0,51.353036,-2.132106,encode_weather('broken clouds'),1.29,190.0,69.0,278.2,0.0],[1,31,1,51.353036,-2.132106,encode_weather('clear sky'),2.03,190.0,10.0,278.08,0.0],[1,31,2,51.353036,-2.132106,encode_weather('scattered clouds'),2.34,195.0,45.0,278.22,0.0],[1,31,3,51.353036,-2.132106,encode_weather('scattered clouds'),2.53,199.0,34.0,277.27,0.0],[1,31,4,51.353036,-2.132106,encode_weather('scattered clouds'),2.91,189.0,27.0,276.43,0.0],[1,31,5,51.353036,-2.132106,encode_weather('scattered clouds'),2.82,201.0,33.0,276.58,0.0],[1,31,6,51.353036,-2.132106,encode_weather('scattered clouds'),2.78,189.0,33.0,276.6,0.0],[1,31,7,51.353036,-2.132106,encode_weather('overcast clouds'),2.83,210.0,98.0,277.39,0.0],[1,31,8,51.353036,-2.132106,encode_weather('broken clouds'),3.07,209.0,59.0,277.67,0.0],[1,31,9,51.353036,-2.132106,encode_weather('scattered clouds'),3.83,205.0,47.0,278.81,0.0],[1,31,10,51.353036,-2.132106,encode_weather('broken clouds'),5.3,215.0,57.0,280.14,0.0],[1,31,11,51.353036,-2.132106,encode_weather('broken clouds'),6.29,216.0,66.0,280.79,0.0],[1,31,12,51.353036,-2.132106,encode_weather('broken clouds'),6.75,223.0,78.0,281.88,0.0],[1,31,13,51.353036,-2.132106,encode_weather('overcast clouds'),7.33,219.0,100.0,282.43,0.0],[1,31,14,51.353036,-2.132106,encode_weather('overcast clouds'),7.71,221.0,100.0,283.46,0.0],[1,31,15,51.353036,-2.132106,encode_weather('overcast clouds'),7.76,218.0,100.0,283.79,0.0],[1,31,16,51.353036,-2.132106,encode_weather('overcast clouds'),7.78,225.0,100.0,284.11,0.0],[1,31,17,51.353036,-2.132106,encode_weather('overcast clouds'),7.73,228.0,100.0,284.37,0.0],[1,31,18,51.353036,-2.132106,encode_weather('overcast clouds'),7.39,240.0,100.0,284.35,0.0],[1,31,19,51.353036,-2.132106,encode_weather('overcast clouds'),8.24,248.0,100.0,284.28,0.0],[1,31,20,51.353036,-2.132106,encode_weather('overcast clouds'),7.85,256.0,100.0,284.13,0.0],[1,31,21,51.353036,-2.132106,encode_weather('light rain'),7.33,257.0,100.0,283.68,0.0],[1,31,22,51.353036,-2.132106,encode_weather('light rain'),7.31,258.0,100.0,283.34,0.0],[1,31,23,51.353036,-2.132106,encode_weather('overcast clouds'),6.5,272.0,100.0,283.39,0.0],[2,1,0,51.353036,-2.132106,encode_weather('overcast clouds'),4.69,288.0,100.0,283.36,0.0],[2,1,1,51.353036,-2.132106,encode_weather('overcast clouds'),3.29,328.0,100.0,283.12,0.0],[2,1,2,51.353036,-2.132106,encode_weather('overcast clouds'),3.18,3.0,100.0,281.96,0.0],[2,1,3,51.353036,-2.132106,encode_weather('overcast clouds'),2.56,14.0,100.0,280.69,0.0],[2,1,4,51.353036,-2.132106,encode_weather('overcast clouds'),1.78,1.0,100.0,279.31,0.0],[2,1,5,51.353036,-2.132106,encode_weather('overcast clouds'),1.64,337.0,100.0,278.21,0.0],[2,1,6,51.353036,-2.132106,encode_weather('overcast clouds'),2.19,324.0,100.0,277.3,0.0],[2,1,7,51.353036,-2.132106,encode_weather('overcast clouds'),2.09,322.0,100.0,276.25,0.0],[2,1,8,51.353036,-2.132106,encode_weather('overcast clouds'),1.51,316.0,100.0,275.51,0.0],[2,1,9,51.353036,-2.132106,encode_weather('overcast clouds'),1.14,305.0,100.0,276.32,0.0],[2,1,10,51.353036,-2.132106,encode_weather('overcast clouds'),1.07,286.0,100.0,277.89,0.0],[2,1,11,51.353036,-2.132106,encode_weather('overcast clouds'),1.39,279.0,100.0,279.71,0.0],[2,1,12,51.353036,-2.132106,encode_weather('overcast clouds'),1.81,271.0,98.0,280.83,0.0],[2,1,13,51.353036,-2.132106,encode_weather('broken clouds'),2.34,244.0,84.0,281.88,0.0],[2,1,14,51.353036,-2.132106,encode_weather('broken clouds'),2.86,241.0,74.0,281.78,0.0],[2,1,15,51.353036,-2.132106,encode_weather('broken clouds'),3.29,250.0,79.0,281.98,0.0],[2,1,16,51.353036,-2.132106,encode_weather('broken clouds'),3.16,246.0,77.0,280.89,0.0],[2,1,17,51.353036,-2.132106,encode_weather('broken clouds'),3.04,236.0,80.0,279.82,0.0],[2,1,18,51.353036,-2.132106,encode_weather('broken clouds'),2.96,235.0,76.0,278.15,0.0],[2,1,19,51.353036,-2.132106,encode_weather('overcast clouds'),3.18,230.0,100.0,277.94,0.0],[2,1,20,51.353036,-2.132106,encode_weather('overcast clouds'),3.04,228.0,100.0,278.23,0.0],[2,1,21,51.353036,-2.132106,encode_weather('overcast clouds'),3.26,223.0,100.0,278.47,0.0],[2,1,22,51.353036,-2.132106,encode_weather('overcast clouds'),3.27,221.0,100.0,278.47,0.0],[2,1,23,51.353036,-2.132106,encode_weather('overcast clouds'),3.15,215.0,100.0,278.88,0.0],[2,2,0,51.353036,-2.132106,encode_weather('overcast clouds'),3.18,211.0,97.0,279.35,0.0],[2,2,1,51.353036,-2.132106,encode_weather('broken clouds'),3.42,209.0,71.0,279.68,0.0],[2,2,2,51.353036,-2.132106,encode_weather('broken clouds'),3.58,214.0,80.0,280.11,0.0],[2,2,3,51.353036,-2.132106,encode_weather('overcast clouds'),3.49,221.0,87.0,280.48,0.0],[2,2,4,51.353036,-2.132106,encode_weather('overcast clouds'),4.01,230.0,90.0,280.97,0.0],[2,2,5,51.353036,-2.132106,encode_weather('overcast clouds'),4.99,239.0,91.0,281.46,0.0],[2,2,6,51.353036,-2.132106,encode_weather('overcast clouds'),5.93,241.0,91.0,281.75,0.0],[2,2,7,51.353036,-2.132106,encode_weather('overcast clouds'),6.72,249.0,100.0,281.91,0.0],[2,2,8,51.353036,-2.132106,encode_weather('overcast clouds'),7.09,255.0,100.0,281.97,0.0],[2,2,9,51.353036,-2.132106,encode_weather('overcast clouds'),6.97,254.0,100.0,282.52,0.0],[2,2,10,51.353036,-2.132106,encode_weather('overcast clouds'),6.93,255.0,100.0,283.03,0.0],[2,2,11,51.353036,-2.132106,encode_weather('overcast clouds'),6.96,259.0,100.0,283.58,0.0],[2,2,12,51.353036,-2.132106,encode_weather('overcast clouds'),8.16,263.0,100.0,284.23,0.0],[2,2,13,51.353036,-2.132106,encode_weather('overcast clouds'),8.5,266.0,100.0,284.28,0.0],[2,2,14,51.353036,-2.132106,encode_weather('overcast clouds'),8.19,268.0,100.0,284.73,0.0],[2,2,15,51.353036,-2.132106,encode_weather('overcast clouds'),7.75,268.0,100.0,284.61,0.0],[2,2,16,51.353036,-2.132106,encode_weather('overcast clouds'),7.43,269.0,100.0,284.55,0.0],[2,2,17,51.353036,-2.132106,encode_weather('overcast clouds'),7.09,270.0,100.0,284.34,0.0],[2,2,18,51.353036,-2.132106,encode_weather('overcast clouds'),6.96,267.0,100.0,284.23,0.0],[2,2,19,51.353036,-2.132106,encode_weather('overcast clouds'),6.57,264.0,100.0,284.14,0.0],[2,2,20,51.353036,-2.132106,encode_weather('overcast clouds'),6.09,264.0,100.0,284.01,0.0],[2,2,21,51.353036,-2.132106,encode_weather('overcast clouds'),5.97,266.0,100.0,284.06,0.0],[2,2,22,51.353036,-2.132106,encode_weather('overcast clouds'),5.98,260.0,100.0,284.16,0.0],[2,2,23,51.353036,-2.132106,encode_weather('overcast clouds'),5.98,259.0,97.0,284.06,0.0],[2,3,0,51.353036,-2.132106,encode_weather('overcast clouds'),6.46,262.0,93.0,283.84,0.0],[2,3,1,51.353036,-2.132106,encode_weather('overcast clouds'),7.19,262.0,99.0,283.85,0.0],[2,3,2,51.353036,-2.132106,encode_weather('overcast clouds'),7.36,263.0,100.0,283.75,0.0],[2,3,3,51.353036,-2.132106,encode_weather('overcast clouds'),7.5,263.0,100.0,283.85,0.0],[2,3,4,51.353036,-2.132106,encode_weather('overcast clouds'),6.98,260.0,100.0,283.75,0.0],[2,3,5,51.353036,-2.132106,encode_weather('overcast clouds'),6.88,259.0,100.0,283.7,0.0],[2,3,6,51.353036,-2.132106,encode_weather('overcast clouds'),6.29,258.0,100.0,283.73,0.0],[2,3,7,51.353036,-2.132106,encode_weather('overcast clouds'),6.01,254.0,100.0,283.54,0.0],[2,3,8,51.353036,-2.132106,encode_weather('overcast clouds'),6.25,253.0,100.0,283.63,0.0],[2,3,9,51.353036,-2.132106,encode_weather('overcast clouds'),6.14,256.0,100.0,283.79,0.0],[2,3,10,51.353036,-2.132106,encode_weather('overcast clouds'),6.5,262.0,100.0,283.82,0.0],[2,3,11,51.353036,-2.132106,encode_weather('overcast clouds'),6.28,262.0,100.0,284.35,0.0],[2,3,12,51.353036,-2.132106,encode_weather('overcast clouds'),6.32,260.0,100.0,284.57,0.0],[2,3,13,51.353036,-2.132106,encode_weather('overcast clouds'),6.19,264.0,100.0,284.85,0.0],[2,3,14,51.353036,-2.132106,encode_weather('overcast clouds'),5.72,262.0,100.0,284.85,0.0],[2,3,15,51.353036,-2.132106,encode_weather('overcast clouds'),5.82,257.0,100.0,284.87,0.0],[2,3,16,51.353036,-2.132106,encode_weather('overcast clouds'),6.26,256.0,100.0,284.76,0.0],[2,3,17,51.353036,-2.132106,encode_weather('overcast clouds'),6.68,259.0,100.0,284.61,0.0],[2,3,18,51.353036,-2.132106,encode_weather('overcast clouds'),7.52,259.0,100.0,284.28,0.0],[2,3,19,51.353036,-2.132106,encode_weather('overcast clouds'),7.97,263.0,100.0,284.25,0.0],[2,3,20,51.353036,-2.132106,encode_weather('overcast clouds'),7.24,268.0,100.0,284.2,0.0],[2,3,21,51.353036,-2.132106,encode_weather('overcast clouds'),6.02,273.0,100.0,284.14,0.0],[2,3,22,51.353036,-2.132106,encode_weather('overcast clouds'),5.17,267.0,100.0,284.15,0.0],[2,3,23,51.353036,-2.132106,encode_weather('overcast clouds'),4.93,262.0,100.0,284.07,0.0],[2,4,0,51.353036,-2.132106,encode_weather('overcast clouds'),4.56,259.0,100.0,283.93,0.0],[2,4,1,51.353036,-2.132106,encode_weather('overcast clouds'),5.08,258.0,100.0,283.88,0.0],[2,4,2,51.353036,-2.132106,encode_weather('overcast clouds'),5.75,259.0,100.0,283.88,0.0],[2,4,3,51.353036,-2.132106,encode_weather('overcast clouds'),5.88,258.0,100.0,283.82,0.0],[2,4,4,51.353036,-2.132106,encode_weather('overcast clouds'),6.72,256.0,100.0,283.78,0.0],[2,4,5,51.353036,-2.132106,encode_weather('overcast clouds'),7.29,259.0,100.0,283.89,0.0],[2,4,6,51.353036,-2.132106,encode_weather('overcast clouds'),8.05,263.0,100.0,283.82,0.0],[2,4,7,51.353036,-2.132106,encode_weather('overcast clouds'),8.58,265.0,100.0,284.07,0.0],[2,4,8,51.353036,-2.132106,encode_weather('overcast clouds'),7.98,266.0,100.0,284.09,0.0],[2,4,9,51.353036,-2.132106,encode_weather('overcast clouds'),6.99,266.0,100.0,284.3,0.0],[2,4,10,51.353036,-2.132106,encode_weather('overcast clouds'),6.91,261.0,100.0,284.38,0.0],[2,4,11,51.353036,-2.132106,encode_weather('overcast clouds'),7.55,256.0,100.0,284.83,0.0],[2,4,12,51.353036,-2.132106,encode_weather('overcast clouds'),8.7,256.0,100.0,285.26,0.0],[2,4,13,51.353036,-2.132106,encode_weather('overcast clouds'),9.38,257.0,100.0,285.56,0.0],[2,4,14,51.353036,-2.132106,encode_weather('overcast clouds'),9.71,261.0,100.0,285.88,0.0],[2,4,15,51.353036,-2.132106,encode_weather('overcast clouds'),9.59,260.0,100.0,285.82,0.0],[2,4,16,51.353036,-2.132106,encode_weather('overcast clouds'),9.31,262.0,99.0,285.49,0.0],[2,4,17,51.353036,-2.132106,encode_weather('overcast clouds'),9.05,263.0,99.0,285.0,0.0],[2,4,18,51.353036,-2.132106,encode_weather('overcast clouds'),8.57,266.0,96.0,284.53,0.0],[2,4,19,51.353036,-2.132106,encode_weather('overcast clouds'),8.54,264.0,92.0,284.14,0.0],[2,4,20,51.353036,-2.132106,encode_weather('overcast clouds'),8.48,264.0,96.0,283.76,0.0],[2,4,21,51.353036,-2.132106,encode_weather('overcast clouds'),9.01,264.0,97.0,283.44,0.0],[2,4,22,51.353036,-2.132106,encode_weather('overcast clouds'),8.29,260.0,98.0,283.37,0.0],[2,4,23,51.353036,-2.132106,encode_weather('overcast clouds'),8.76,261.0,98.0,283.36,0.0],[2,5,0,51.353036,-2.132106,encode_weather('overcast clouds'),9.07,260.0,98.0,283.3,0.0],]

for i in range(len(instances)):
    instance = instances[i]    
    predicted_output = predict_instance(instance)
    print(predicted_output)