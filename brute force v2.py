import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product

# Descargar datos históricos de una criptomoneda (por ejemplo, Bitcoin)
data = yf.download('BTC-USD', start='2020-01-01', end='2021-01-01')
data = data[['Close']]  # Usar solo el precio de cierre

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
    
    return portfolio['Total'].iloc[-1], portfolio['Returns'].std()

# Definir rangos de parámetros
short_window_range = range(5, 20)
long_window_range = range(20, 50)

# Optimización por fuerza bruta
results = []

for short_window, long_window in product(short_window_range, long_window_range):
    if short_window < long_window:
        performance, risk = backtest_strategy(short_window, long_window, data)
        results.append((short_window, long_window, performance, risk))

# Convertir resultados a DataFrame para análisis
results_df = pd.DataFrame(results, columns=['Short_Window', 'Long_Window', 'Performance', 'Risk'])

# Visualizar resultados
plt.figure(figsize=(14, 7))

# Gráfico de Performance
plt.subplot(1, 2, 1)
plt.scatter(results_df['Short_Window'], results_df['Long_Window'], c=results_df['Performance'], cmap='viridis')
plt.colorbar(label='Performance')
plt.xlabel('Short Window')
plt.ylabel('Long Window')
plt.title('Performance of Strategy')

# Gráfico de Riesgo
plt.subplot(1, 2, 2)
plt.scatter(results_df['Short_Window'], results_df['Long_Window'], c=results_df['Risk'], cmap='viridis')
plt.colorbar(label='Risk')
plt.xlabel('Short Window')
plt.ylabel('Long Window')
plt.title('Risk of Strategy')

plt.tight_layout()
plt.show()

# Encontrar los mejores parámetros basados en la performance
best_params = results_df.loc[results_df['Performance'].idxmax()]
print(f"Mejores parámetros: Corto = {best_params['Short_Window']}, Largo = {best_params['Long_Window']}")
print(f"Mejor rendimiento: {best_params['Performance']}, Riesgo asociado: {best_params['Risk']}")
