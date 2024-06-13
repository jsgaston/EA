import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, LSTM
from keras.metrics import RootMeanSquaredError as rmse
from keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from alpha_vantage.timeseries import TimeSeries

# Alpha Vantage API parameters
api_key = 
symbol = 'MSFT'#'EWP'  # Stock symbol, e.g., Apple Inc.
interval = 'daily'
outputsize = 'full'

# Fetch data from Alpha Vantage
def fetch_data(api_key, symbol, interval, outputsize):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+symbol+'&apikey='+api_key + '&outputsize='+outputsize
    r = requests.get(url)
    data1 = r.json()
    #data = response.json()
    if 'Time Series (Daily)' not in data1:
        raise ValueError("Invalid API response. Check your API key and parameters.")
    df = pd.DataFrame.from_dict(data1['Time Series (Daily)'], orient='index')
    df = df.rename(columns={'5. adjusted close': 'close'}).astype(float)
    return df

# Fetch the data
df = fetch_data(api_key, symbol, interval, outputsize)
print(df)
df = df.sort_index()  # Ensure the data is in chronological order

# Get close prices
data = df.filter(['4. close']).values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define training size
training_size = int(len(scaled_data) * 0.80)
train_data_initial = scaled_data[0:training_size, :]
test_data_initial = scaled_data[training_size:, :1]

# Function to split sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Define sequence length
time_step = 120

# Split data into sequences
x_train, y_train = split_sequence(train_data_initial, time_step)
x_test, y_test = split_sequence(test_data_initial, time_step)

# Reshape data for LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Build the model
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same', input_shape=(time_step, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=[rmse()])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=32, callbacks=[early_stop], verbose=2)

# Evaluate the model
train_loss, train_rmse = model.evaluate(x_train, y_train, batch_size=32)
print(f"train_loss={train_loss:.3f}")
print(f"train_rmse={train_rmse:.3f}")

test_loss, test_rmse = model.evaluate(x_test, y_test, batch_size=32)
print(f"test_loss={test_loss:.3f}")
print(f"test_rmse={test_rmse:.3f}")

# Predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Inverse scaling
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_train_pred = scaler.inverse_transform(y_train_pred)
y_test_pred = scaler.inverse_transform(y_test_pred)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_corr, _ = pearsonr(y_train.flatten(), y_train_pred.flatten())
test_corr, _ = pearsonr(y_test.flatten(), y_test_pred.flatten())

print(f"train_mse={train_mse:.3f}")
print(f"test_mse={test_mse:.3f}")
print(f"train_r2={train_r2:.3f}")
print(f"test_r2={test_r2:.3f}")
print(f"train_corr={train_corr:.3f}")
print(f"test_corr={test_corr:.3f}")

# Make prediction for the next value
last_sequence = scaled_data[-time_step:].reshape((1, time_step, 1))
next_value_scaled = model.predict(last_sequence)
next_value = scaler.inverse_transform(next_value_scaled)

# Compare with the last actual value
last_real_value = data[-1][0]

percentage_change = ((next_value - last_real_value) / last_real_value) * 100

if next_value > last_real_value:
    print(f"The prediction indicates that the value will go up. Predicted next value: {next_value[0][0]:.5f}, Last real value: {last_real_value:.5f}")
else:
    print(f"The prediction indicates that the value will go down. Predicted next value: {next_value[0][0]:.5f}, Last real value: {last_real_value:.5f}")

print("the symbol is: {symbol}")
print(f"Percentage change: {percentage_change[0][0]:.2f}%")
