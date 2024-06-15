import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator
from itertools import product

# Download historical stock data
data = yf.download("EURUSD=X", start="2016-01-01", end="2022-06-01")
data['Close'] = data['Adj Close']
print(data)

# Remove rows with NaN values
data = data.dropna()

# Moving average crossover strategy function
def moving_average_strategy(data, short_window, long_window):
    data = data.copy()  # Make a copy of the data to avoid modifying the original DataFrame
    data['SMA_short'] = SMAIndicator(data['Close'], window=short_window).sma_indicator()  # Calculate short-term SMA
    data['SMA_long'] = SMAIndicator(data['Close'], window=long_window).sma_indicator()  # Calculate long-term SMA
    
    # Remove rows with NaN values after calculating SMAs
    data = data.dropna()
    
    # Generate trading signals
    data['signal'] = 0
    data.loc[data.index[short_window:], 'signal'] = np.where(
        data['SMA_short'].iloc[short_window:] > data['SMA_long'].iloc[short_window:], 1, 0
    )
    data['positions'] = data['signal'].diff()  # Calculate positions (buy/sell signals)
    
    # Calculate returns
    data['returns'] = data['Close'].pct_change().shift(-1)
    data['strategy_returns'] = data['returns'] * data['positions'].shift(1)
    
    # Remove rows with NaN values in strategy returns
    data = data.dropna(subset=['strategy_returns'])
    
    # Calculate Sharpe Ratio
    mean_return = data['strategy_returns'].mean()
    std_return = data['strategy_returns'].std()
    sharpe_ratio = mean_return / std_return
    
    return sharpe_ratio

# Parameter optimization using Grid Search
short_windows = range(5, 50, 5)
long_windows = range(20, 200, 5)

best_params = None
best_sharpe = -np.inf

# Grid search over parameter space
for short_window, long_window in product(short_windows, long_windows):
    if short_window >= long_window:
        continue
    
    sharpe = moving_average_strategy(data, short_window, long_window)
    print(f"Evaluating short={short_window}, long={long_window}: Sharpe Ratio={sharpe}")
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = (short_window, long_window)

# Print the best parameters and Sharpe Ratio
if best_params is not None:
    print(f"Best parameters: short={best_params[0]}, long={best_params[1]}")
    print(f"Best Sharpe Ratio: {best_sharpe}")
else:
    print("No optimal parameters found.")
