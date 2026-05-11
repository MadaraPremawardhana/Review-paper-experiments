import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    '2023 data for ML benchmark training.csv',
    parse_dates=['Date/Time']
)

# ============================================================
# SORT BY TIME
# ============================================================

data = data.sort_values('Date/Time').reset_index(drop=True)

# ============================================================
# SPLIT:
# TRAIN ON 2023
# PREDICT ON 2024
# ============================================================

train_data = data[data['Date/Time'].dt.year == 2023].copy()
test_data  = data[data['Date/Time'].dt.year == 2024].copy()

print(f"Training samples: {len(train_data)}")
print(f"Testing samples : {len(test_data)}")

# ============================================================
# TARGET VARIABLE
# ============================================================

target_col = 'Average actual panel production'

# ============================================================
# NORMALIZATION
# FIT ONLY ON TRAIN DATA
# ============================================================

scaler = MinMaxScaler()

train_power = scaler.fit_transform(
    train_data[[target_col]]
)

test_power = scaler.transform(
    test_data[[target_col]]
)

# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(data_array, seq_length=4):

    Xs = []
    ys = []

    for i in range(len(data_array) - seq_length):

        Xs.append(data_array[i:i + seq_length])
        ys.append(data_array[i + seq_length])

    return np.array(Xs), np.array(ys)

seq_length = 4

X_train, y_train = create_sequences(train_power, seq_length)
X_test, y_test   = create_sequences(test_power, seq_length)

# ============================================================
# MODEL BUILDERS
# ============================================================

# ------------------------------------------------------------
# 1. STACKED LSTM
# ------------------------------------------------------------

def build_stacked_lstm():

    model = models.Sequential([

        layers.LSTM(
            32,
            return_sequences=True,
            input_shape=(seq_length, 1)
        ),

        layers.LSTM(64),

        layers.Dense(1)

    ])

    model.compile(
        optimizer=Adam(),
        loss='mse'
    )

    return model

# ------------------------------------------------------------
# 2. BIDIRECTIONAL LSTM
# ------------------------------------------------------------

def build_bilstm():

    model = models.Sequential([

        layers.Bidirectional(
            layers.LSTM(
                64,
                input_shape=(seq_length, 1)
            )
        ),

        layers.Dense(1)

    ])

    model.compile(
        optimizer=Adam(),
        loss='mse'
    )

    return model

# ------------------------------------------------------------
# 3. CNN-LSTM
# ------------------------------------------------------------

def build_cnn_lstm():

    model = models.Sequential([

        layers.Conv1D(
            filters=64,
            kernel_size=2,
            activation='relu',
            input_shape=(seq_length, 1)
        ),

        layers.MaxPooling1D(pool_size=2),

        layers.LSTM(64),

        layers.Dense(1)

    ])

    model.compile(
        optimizer=Adam(),
        loss='mse'
    )

    return model

# ============================================================
# CREATE MODELS
# ============================================================

stacked_lstm = build_stacked_lstm()
bilstm       = build_bilstm()
cnn_lstm     = build_cnn_lstm()

models_list = [
    ('Stacked_LSTM', stacked_lstm),
    ('BiLSTM', bilstm),
    ('CNN_LSTM', cnn_lstm)
]

# ============================================================
# TRAIN MODELS USING 2023 DATA
# ============================================================

for name, model in models_list:

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=8,
        verbose=1
    )

# ============================================================
# VALIDATION PREDICTIONS FOR META LEARNER
# USING TRAIN DATA
# ============================================================

train_predictions = []

for name, model in models_list:

    pred = model.predict(X_train).flatten()

    train_predictions.append(pred)

meta_X_train = np.vstack(train_predictions).T

# ============================================================
# TRAIN META-LEARNER (SVR)
# ============================================================

meta_model = SVR(
    kernel='rbf',
    C=10,
    epsilon=0.01
)

meta_model.fit(
    meta_X_train,
    y_train.flatten()
)

# ============================================================
# GENERATE 2024 PREDICTIONS
# ============================================================

test_predictions = []

for name, model in models_list:

    pred = model.predict(X_test).flatten()

    test_predictions.append(pred)

meta_X_test = np.vstack(test_predictions).T

# ============================================================
# META PREDICTIONS
# ============================================================

meta_prediction_scaled = meta_model.predict(meta_X_test)

# ============================================================
# INVERSE SCALE
# ============================================================

meta_prediction_actual = scaler.inverse_transform(
    meta_prediction_scaled.reshape(-1, 1)
)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# ============================================================
# EVALUATION
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        meta_prediction_actual
    )
)

print(f"\n✅ 2024 Prediction RMSE: {rmse:.4f}")

# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

timestamps = test_data['Date/Time'].iloc[seq_length:].reset_index(drop=True)

results_df = pd.DataFrame({

    'Date/Time': timestamps,

    'Actual Panel Power (watts)': y_test_actual.flatten(),

    'Predicted Panel Power (watts)': meta_prediction_actual.flatten()

})

# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    'meta_learning_2024_predictions.csv',
    index=False
)

print("\n✅ Predictions saved to:")
print("meta_learning_2024_predictions.csv")
