import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# Generar datos de ejemplo (simulando precios de cierre)
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2021-01-01')
prices = np.cumsum(np.random.randn(len(dates))) + 100
data = pd.DataFrame({'Close': prices}, index=dates)

# Función para calcular el rendimiento de la estrategia
def backtest_strategy(short_window, long_window, data):
    data = data.copy()  # Crear una copia del DataFrame original
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    data['Signal'] = 0
    data.loc[data.index[short_window:], 'Signal'] = np.where(
        data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()
    
    initial_capital = 10000.0
    positions = 100 * data['Position']
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Holdings'] = positions.cumsum() * data['Close']
    portfolio['Cash'] = initial_capital - (positions * data['Close']).cumsum()
    portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']
    portfolio['Returns'] = portfolio['Total'].pct_change()
    
    return portfolio['Total'].iloc[-1]

# Definir rangos de parámetros
short_window_range = range(5, 20)
long_window_range = range(20, 50)

# Optimización por fuerza bruta
best_params = (0, 0)
best_performance = -np.inf

for short_window, long_window in product(short_window_range, long_window_range):
    if short_window < long_window:
        performance = backtest_strategy(short_window, long_window, data)
        if performance > best_performance:
            best_performance = performance
            best_params = (short_window, long_window)

print(f"Mejores parámetros: Corto = {best_params[0]}, Largo = {best_params[1]}")
print(f"Mejor rendimiento: {best_performance}")
