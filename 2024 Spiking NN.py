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

# esn_power_prediction.py
from sklearn.linear_model import Ridge

class SimpleESN:
    def __init__(self, n_reservoir=100, sparsity=0.1, random_state=42):
        np.random.seed(random_state)
        self.n_reservoir = n_reservoir
        self.Win = np.random.rand(n_reservoir, 1) - 0.5
        self.W = np.random.rand(n_reservoir, n_reservoir) - 0.5
        mask = np.random.rand(*self.W.shape) < sparsity
        self.W *= mask

    def fit(self, X, y):
        states = []
        x = np.zeros((self.n_reservoir,))
        for u in X:
            x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, x))
            states.append(x.copy())
        self.ridge = Ridge(alpha=1e-6)
        self.ridge.fit(states, y)

    def predict(self, X):
        preds = []
        x = np.zeros((self.n_reservoir,))
        for u in X:
            x = np.tanh(np.dot(self.Win, u) + np.dot(self.W, x))
            preds.append(self.ridge.predict(x.reshape(1, -1))[0])
        return np.array(preds)

# Univariate supervised learning
X = power_scaled[:-1]
y = power_scaled[1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

esn = SimpleESN()
esn.fit(X_train, y_train)
y_pred = esn.predict(X_test)
