import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Descargar datos históricos de Bitcoin
data = yf.download('BTC-USD', start='2020-01-01', end='2021-01-01')
data = data[['Close']].copy()  # Usar solo el precio de cierre y asegurarse de que es una copia

# Función para calcular el rendimiento de la estrategia
def backtest_strategy(short_window, long_window, data):
    # Asegúrate de que 'data' es un DataFrame y no una vista de otro DataFrame
    data = data.copy()

    if short_window <= 0 or long_window <= 0 or short_window >= long_window:
        return None  # Valores no válidos
    
    # Utiliza .loc para realizar las asignaciones de manera segura
    data.loc[:, 'Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data.loc[:, 'Long_MA'] = data['Close'].rolling(window=long_window).mean()
        
    # Genera las señales de trading
    data['Signal'] = 0
    data.loc[data.index[short_window:], 'Signal'] = np.where(
        data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()
    
    # Calcula el rendimiento de la estrategia
    initial_capital = 10000.0
    positions = 100 * data['Position']
    portfolio = pd.DataFrame(index=data.index)
    portfolio['Holdings'] = positions.cumsum() * data['Close']
    portfolio['Cash'] = initial_capital - (positions * data['Close']).cumsum()
    portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']
    portfolio['Returns'] = portfolio['Total'].pct_change()
    
    return portfolio  # Devuelve el DataFrame completo

# Optimización de Estrategias Evolutivas
def evolution_strategies(population_size, generations, sigma, data):
    population = np.random.randint(5, 50, size=(population_size, 2))
    for gen in range(generations):
        fitness = np.array([backtest_strategy(ind[0], ind[1], data)['Total'].iloc[-1] if backtest_strategy(ind[0], ind[1], data) is not None else -np.inf for ind in population])
        best_idx = np.argmax(fitness)
        best_individual = population[best_idx]
        new_population = [best_individual + sigma * np.random.randn(2) for _ in range(population_size)]
        new_population = np.clip(new_population, 5, 50).astype(int)
        population = new_population
    return best_individual

# Parámetros
population_size = 50
generations = 20
sigma = 5

# Ejecutar la optimización
best_params = evolution_strategies(population_size, generations, sigma, data)
best_short_window, best_long_window = best_params

# Ejecutar backtesting con los mejores parámetros
portfolio = backtest_strategy(best_short_window, best_long_window, data)
best_performance = portfolio['Total'].iloc[-1]  # Define best_performance

# Rendimiento Acumulado, Volatilidad y Ratio de Sharpe
initial_capital = 10000.0
portfolio['Cumulative Returns'] = (portfolio['Total'] / initial_capital) - 1
portfolio['Daily Returns'] = portfolio['Total'].pct_change()
volatility = portfolio['Daily Returns'].std() * np.sqrt(252)  # Anualizada
risk_free_rate = 0.01  # Tasa libre de riesgo, ajusta esto a la tasa actual
sharpe_ratio = (portfolio['Daily Returns'].mean() - risk_free_rate) / volatility


####################################################################################
# Optimización de Estrategias Evolutivas con registro de fitness
def evolution_strategies(population_size, generations, sigma, data):
    population = np.random.randint(5, 50, size=(population_size, 2))
    fitness_history = []  # Para registrar el mejor fitness en cada generación

    for gen in range(generations):
        fitness = np.array([backtest_strategy(ind[0], ind[1], data)['Total'].iloc[-1] if backtest_strategy(ind[0], ind[1], data) is not None else -np.inf for ind in population])
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        fitness_history.append(best_fitness)  # Registrar el mejor fitness
        best_individual = population[best_idx]
        new_population = [best_individual + sigma * np.random.randn(2) for _ in range(population_size)]
        new_population = np.clip(new_population, 5, 50).astype(int)
        population = new_population

    return best_individual, fitness_history

# Ejecutar la optimización y obtener el historial de fitness
best_params, fitness_history = evolution_strategies(population_size, generations, sigma, data)

# Graficar la convergencia de la optimización
plt.figure(figsize=(10, 5))
plt.plot(fitness_history, label='Mejor Fitness por Generación')
plt.xlabel('Generación')
plt.ylabel('Fitness')
plt.title('Convergencia de la Optimización')
plt.legend()
plt.show()

##################################################################################

# Imprimir los resultados
print(f"Mejores Parámetros: Ventana Corta = {best_short_window}, Ventana Larga = {best_long_window}")
print(f"Mejor Rendimiento: {best_performance}")
print(f"Rendimiento Acumulado: {portfolio['Cumulative Returns'].iloc[-1]}")
print(f"Volatilidad Anualizada: {volatility}")
print(f"Ratio de Sharpe: {sharpe_ratio}")

# Graficar los resultados
plt.figure(figsize=(14, 7))
# Asegúrate de que 'data' es una copia antes de modificarla
data = data.copy()
data.loc[:, 'Short_MA'] = data['Close'].rolling(window=best_short_window).mean()
data.loc[:, 'Long_MA'] = data['Close'].rolling(window=best_long_window).mean()
plt.plot(data['Close'], label='Precio de Cierre')
plt.plot(data['Short_MA'], label=f'MA Corta ({best_short_window})')
plt.plot(data['Long_MA'], label=f'MA Larga ({best_long_window})')
plt.legend()
plt.show()

