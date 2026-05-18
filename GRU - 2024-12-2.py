import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from datetime import datetime

# GPU-specific configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth to avoid allocating all memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Explicitly set GPU as the visible device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        # Optional: Log device placement to confirm GPU usage
        tf.debugging.set_log_device_placement(True)
        
        print(f"GPU is available and set as primary device: {gpus[0]}")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found. CPU will be used instead.")
    
# Load the dataset

df = pd.read_csv('D:/pre-digital-twin-v4/Data/2023_Jan to June_KB_ hourly.csv')

# Data preprocessing
def preprocess_data(df):
    if df.empty:
      return None, None, None, None

    # Convert Date/Time to datetime objects and extract relevant features
    df['Date/Time'] = pd.to_datetime(df['Date/Time'], format='%d/%m/%Y %H:%M')
    df['Hour'] = df['Date/Time'].dt.hour
    df['Dayofweek'] = df['Date/Time'].dt.dayofweek
    df['Month'] = df['Date/Time'].dt.month

    # One-hot encode 'WeatherDescription'
    df = pd.get_dummies(df, columns=['WeatherDescription'], prefix='Weather')

    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['Latitude', 'Longitude', 'Wind velcity (m/s)', 'Wind direction(degrees from north)',
                      'Cloud cover percentage', 'Atmospheric temperature(Kelvin)', 'GHI', 'Hour', 'Dayofweek', 'Month'] + [col for col in df.columns if 'Weather' in col]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler, numerical_cols


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length, -1]  # Target is 'Average panel production'
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Preprocess data
df, scaler, numerical_cols = preprocess_data(df)

if df is not None:
    # Split into features (X) and target (y)
    X = df[numerical_cols]
    y = df['Average panel production']

    # Create sequences
    seq_length = 10 # Example sequence length. Adjust as needed
    X_seq, y_seq = create_sequences(X.values, seq_length)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=None)

    # Build and train the GRU model
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(GRU(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, batch_size=32) # adjust epochs and batch size

instances = []

for i in range(len(instances)):
    numerical_cols = instances[i]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])    
    prediction = model.predict(np.zeros((1, seq_length, len(numerical_cols))))
    print("Predicted average panel production:", prediction[0][0])
