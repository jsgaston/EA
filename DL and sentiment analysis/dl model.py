import ccxt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Flatten
import tf2onnx
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

# Función para descargar datos de Binance
def descargar_datos(symbol, timeframe='1d', start_date='2000-01-01T00:00:00Z', end_date='2024-05-22T00:00:00Z'):
    exchange = ccxt.binance({'enableRateLimit': False})
    since = exchange.parse8601(start_date)
    end_date_timestamp = pd.to_datetime(end_date, utc=True)
    all_data = []

    while since < end_date_timestamp.timestamp() * 1000:
        ohlc = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
        all_data.extend(ohlc)
        since = ohlc[-1][0] + 1  # increment the `since` parameter by one millisecond

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Verify both are timezone-aware or convert if necessary
    if df.index.tz is None:
        df.index = df.index.tz_localize('utc')
    
    df = df[df.index <= end_date_timestamp]
    print(df)
    return df['close'].values

# Load data
data = descargar_datos('ETH/USDT')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Function to create samples from the sequence
def crear_muestras(dataset, pasos_de_tiempo=120):
    X, y = [], []
    for i in range(pasos_de_tiempo, len(dataset)):
        X.append(dataset[i-pasos_de_tiempo:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Prepare training and test data
pasos_de_tiempo = 120
X, y = crear_muestras(data, pasos_de_tiempo)
X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

# Split the data (80% for training)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train the model

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu',padding = 'same',input_shape=(X_train.shape[1],1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss= 'mse' , metrics = [tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Set up early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)
# Checkpoint to save the best model
checkpoint = ModelCheckpoint(
    'best_model.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=False
)


# model training for 300 epochs
history = model.fit(X_train, y_train, epochs = 300 , validation_data = (X_test,y_test), batch_size=32, callbacks=[early_stopping, checkpoint], verbose=2)

# Plot the training history
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('Model Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss/RMSE')
plt.legend()
plt.savefig('ETHUSDT.png')  # Save the plot as an image file

# Convert the model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13, output_path="model_ethusdt.onnx")
print("ONNX model saved as 'model_ethusdt.onnx'.")

# Evaluate the model
train_loss, train_rmse = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_rmse = model.evaluate(X_test, y_test, verbose=0)
print(f"train_loss={train_loss:.3f}, train_rmse={train_rmse:.3f}")
print(f"test_loss={test_loss:.3f}, test_rmse={test_rmse:.3f}")