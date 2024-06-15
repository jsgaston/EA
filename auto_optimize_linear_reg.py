import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator
from sklearn.linear_model import LinearRegression
from itertools import product

# Download historical stock data
def download_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Close'] = data['Adj Close']
    data = data.dropna()
    return data

# Moving average crossover strategy function
def moving_average_strategy(data, short_window, long_window):
    data = data.copy()
    data['SMA_short'] = SMAIndicator(data['Close'], window=short_window).sma_indicator()
    data['SMA_long'] = SMAIndicator(data['Close'], window=long_window).sma_indicator()
    data = data.dropna()
    
    if data.empty:
        return np.nan
    
    data['signal'] = 0
    data.loc[data.index[short_window:], 'signal'] = np.where(
        data['SMA_short'].iloc[short_window:] > data['SMA_long'].iloc[short_window:], 1, 0
    )
    data['positions'] = data['signal'].diff()
    
    if data.empty:
        return np.nan

    data['returns'] = data['Close'].pct_change().shift(-1)
    data['strategy_returns'] = data['returns'] * data['positions'].shift(1)
    data = data.dropna(subset=['strategy_returns'])
    
    if data.empty:
        return np.nan

    mean_return = data['strategy_returns'].mean()
    std_return = data['strategy_returns'].std()
    
    if std_return == 0:
        return np.nan
    
    sharpe_ratio = mean_return / std_return
    return sharpe_ratio

# Prepare the data for regression
def prepare_data_for_regression(data, short_windows, long_windows):
    results = []
    for short_window, long_window in product(short_windows, long_windows):
        if short_window >= long_window:
            continue
        sharpe = moving_average_strategy(data, short_window, long_window)
        if not np.isnan(sharpe):
            results.append((short_window, long_window, sharpe))
    return pd.DataFrame(results, columns=['short_window', 'long_window', 'sharpe_ratio'])

# Train linear regression model
def train_linear_regression(df):
    X = df[['short_window', 'long_window']]
    y = df['sharpe_ratio']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict optimal parameters
def predict_optimal_parameters(model):
    short_windows = np.arange(5, 50, 1)
    long_windows = np.arange(50, 200, 1)
    best_params = None
    best_sharpe = -np.inf
    for short_window, long_window in product(short_windows, long_windows):
        if short_window >= long_window:
            continue
        input_df = pd.DataFrame([[short_window, long_window]], columns=['short_window', 'long_window'])
        predicted_sharpe = model.predict(input_df)[0]
        if predicted_sharpe > best_sharpe:
            best_sharpe = predicted_sharpe
            best_params = (short_window, long_window)
    return best_params, best_sharpe

# Rolling optimization
def rolling_optimization(symbol, start_date, end_date, look_back_period, optimization_interval):
    data = download_data(symbol, start_date, end_date)
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    while current_date <= end_date:
        look_back_end = current_date
        look_back_start = current_date - pd.DateOffset(months=look_back_period)
        if look_back_start < pd.to_datetime(start_date):
            look_back_start = pd.to_datetime(start_date)
        optimization_data = data[look_back_start:look_back_end]
        if optimization_data.empty:
            current_date += pd.DateOffset(months=optimization_interval)
            continue
        short_windows = range(5, 50, 5)
        long_windows = range(50, 200, 10)
        df = prepare_data_for_regression(optimization_data, short_windows, long_windows)
        if df.empty:
            current_date += pd.DateOffset(months=optimization_interval)
            continue
        model = train_linear_regression(df)
        best_params, best_sharpe = predict_optimal_parameters(model)
        print(f"Date: {look_back_end.date()} | Best parameters: short={best_params[0]}, long={best_params[1]} | Predicted Sharpe Ratio: {best_sharpe}")
        current_date += pd.DateOffset(months=optimization_interval)

# Example usage
rolling_optimization("AAPL", "2018-01-01", "2023-01-01", look_back_period=12, optimization_interval=1)
