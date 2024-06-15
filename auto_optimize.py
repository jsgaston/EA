import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator
from itertools import product

# Descargar datos históricos
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")
data['Close'] = data['Adj Close']

# Eliminar filas con valores NaN
data = data.dropna()

# Estrategia de cruce de medias móviles
def moving_average_strategy(data, short_window, long_window):
    data = data.copy()
    data['SMA_short'] = SMAIndicator(data['Close'], window=short_window).sma_indicator()
    data['SMA_long'] = SMAIndicator(data['Close'], window=long_window).sma_indicator()
    
    # Eliminar filas con valores NaN después de calcular las medias móviles
    data = data.dropna()
    
    data['signal'] = 0
    data.loc[data.index[short_window:], 'signal'] = np.where(
        data['SMA_short'].iloc[short_window:] > data['SMA_long'].iloc[short_window:], 1, 0
    )
    data['positions'] = data['signal'].diff()
    
    data['returns'] = data['Close'].pct_change().shift(-1)
    data['strategy_returns'] = data['returns'] * data['positions'].shift(1)
    
    # Eliminar filas con valores NaN en los retornos de la estrategia
    data = data.dropna(subset=['strategy_returns'])
    
    return data['strategy_returns'].cumsum().iloc[-1]

# Optimización con Grid Search
short_windows = range(5, 50, 5)
long_windows = range(50, 200, 10)

best_params = None
best_performance = -np.inf

for short_window, long_window in product(short_windows, long_windows):
    if short_window >= long_window:
        continue
    
    performance = moving_average_strategy(data, short_window, long_window)
    print(f"Evaluando corto={short_window}, largo={long_window}: desempeño={performance}")
    if performance > best_performance:
        best_performance = performance
        best_params = (short_window, long_window)

if best_params is not None:
    print(f"Mejores parámetros: corto={best_params[0]}, largo={best_params[1]}")
    print(f"Mejor desempeño: {best_performance}")
else:
    print("No se encontraron parámetros óptimos.")
