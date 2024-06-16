import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# Descargar datos históricos de una criptomoneda (por ejemplo, Bitcoin)
data = yf.download('BTC-USD', start='2020-01-01', end='2021-01-01')
data = data[['Close']]  # Usar solo el precio de cierre

# Función para calcular el rendimiento de la estrategia
def backtest_strategy(short_window, long_window, data):
    if short_window <= 0 or long_window <= 0 or short_window >= long_window:
        return -np.inf, np.inf  # Valores no válidos
    data = data.copy().assign(
        Short_MA=data['Close'].rolling(window=short_window).mean(),
        Long_MA=data['Close'].rolling(window=long_window).mean()
    )
    data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1, 0)
    data['Position'] = data['Signal'].diff().fillna(0)
    
    initial_capital = 10000.0
    positions = 100 * data['Position']
    portfolio = pd.DataFrame(index=data.index).assign(
        Holdings=positions.cumsum() * data['Close'],
        Cash=initial_capital - (positions * data['Close']).cumsum()
    )
    portfolio['Total'] = portfolio['Holdings'] + portfolio['Cash']
    portfolio['Returns'] = portfolio['Total'].pct_change().fillna(0)
    
    return portfolio['Total'].iloc[-1], portfolio['Returns'].std()

# Configuración del Algoritmo Genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 5, 50)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función de evaluación mejorada
def evaluate(individual):
    short_window, long_window = map(int, individual)
    performance, risk = backtest_strategy(short_window, long_window, data)
    fitness = performance - risk
    print(f"Evaluating individual: {individual}, Fitness: {fitness}")  # Impresión de depuración
    return (fitness,)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=5, up=50, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

##################################################################################
# Define el objeto stats para recopilar estadísticas
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Define el objeto log para registrar la información de cada generación
log = tools.Logbook()
log.header = ["gen", "avg", "std", "min", "max"]

# Ejecutar el Algoritmo Genético
try:
    population = toolbox.population(n=50)
    num_generations = 20
    
    for gen in range(num_generations):
        # Evaluar la población
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Registrar las estadísticas de la generación actual
        record = stats.compile(population)
        log.record(gen=gen, **record)
        
        # Seleccionar y reproducir la próxima generación
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Reemplazar la población con los descendientes
        population[:] = offspring
    
    # Extraer el mejor individuo
    best_individual = tools.selBest(population, k=1)[0]
    print(f"Mejor individuo: {best_individual}, Fitness: {best_individual.fitness.values[0]}")
    
except Exception as e:
    print(f"Se ha producido un error durante la ejecución del algoritmo genético: {e}")

# Graficar los resultados
generations = log.select("gen")
avg_fitness = log.select("avg")
max_fitness = log.select("max")
min_fitness = log.select("min")

# Asegúrate de que los valores de fitness son finitos
avg_fitness = np.nan_to_num(avg_fitness, nan=np.nanmin(avg_fitness), posinf=np.nanmax(avg_fitness), neginf=np.nanmin(avg_fitness))
max_fitness = np.nan_to_num(max_fitness, nan=np.nanmin(max_fitness), posinf=np.nanmax(max_fitness), neginf=np.nanmin(max_fitness))
min_fitness = np.nan_to_num(min_fitness, nan=np.nanmin(min_fitness), posinf=np.nanmax(min_fitness), neginf=np.nanmin(min_fitness))

# Asegúrate de que los valores de fitness son finitos
avg_fitness = [x if np.isfinite(x) else np.nanmin(avg_fitness) for x in avg_fitness]
max_fitness = [x if np.isfinite(x) else np.nanmin(max_fitness) for x in max_fitness]
min_fitness = [x if np.isfinite(x) else np.nanmin(min_fitness) for x in min_fitness]

# Graficar los resultados
plt.figure(figsize=(14, 7))

# Gráfico de rendimiento
plt.subplot(1, 2, 1)
plt.plot(generations, avg_fitness, label="Fitness Promedio", linestyle='--', marker='o')
plt.plot(generations, max_fitness, label="Fitness Máximo", linestyle='-', marker='x')
plt.plot(generations, min_fitness, label="Fitness Mínimo", linestyle=':', marker='s')
plt.xlabel("Generación")
plt.ylabel("Fitness")
plt.title("Fitness a lo largo de las Generaciones")
plt.legend()

# Establecer límites del eje Y manualmente si es necesario
plt.ylim([min(filter(np.isfinite, min_fitness)) - 0.1, max(filter(np.isfinite, max_fitness)) + 0.1])

# ... (el resto del código para graficar permanece igual)


# ... (el resto del código para graficar permanece igual)


# Gráfico de dispersión de la población final
short_windows = [ind[0] for ind in population]
long_windows = [ind[1] for ind in population]
performances = [evaluate(ind)[0] for ind in population]

# Impresiones de depuración
print("Ventanas Cortas:", short_windows)
print("Ventanas Largas:", long_windows)
print("Rendimientos:", performances)

plt.subplot(1, 2, 2)
plt.scatter(short_windows, long_windows, c=performances, cmap='viridis')
plt.colorbar(label='Rendimiento')
plt.xlabel('Ventana Corta')
plt.ylabel('Ventana Larga')
plt.title('Rendimiento de la Población Final')

plt.tight_layout()
plt.show()
