import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import SMAIndicator
from deap import base, creator, tools, algorithms
import random

# Download historical stock data
def download_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Close'] = data['Adj Close']
    data = data.dropna()
    return data

# Moving average crossover strategy function
def moving_average_strategy(data, short_window, long_window):
    short_window = int(short_window)
    long_window = int(long_window)

    if not isinstance(short_window, int) or not isinstance(long_window, int):
        raise ValueError("Window lengths must be integers")
    if short_window <= 0 or long_window <= 0:
        raise ValueError("Window lengths must be positive integers")

    data = data.copy()
    data['SMA_short'] = SMAIndicator(data['Close'], window=short_window, fillna=True).sma_indicator()
    data['SMA_long'] = SMAIndicator(data['Close'], window=long_window, fillna=True).sma_indicator()
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

# Genetic Algorithm setup
def setup_ga(data):
    # Define the evaluation function
    def evaluate(individual):
        short_window, long_window = individual
        if short_window >= long_window:
            return -np.inf,
        try:
            sharpe = moving_average_strategy(data, short_window, long_window)
            if np.isnan(sharpe):
                return -np.inf,
            return sharpe,
        except ValueError:
            return -np.inf,

    # Register the individual and population
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_short_window", random.randint, 5, 50)
    toolbox.register("attr_long_window", random.randint, 50, 200)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_short_window, toolbox.attr_long_window), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[5, 50], up=[50, 200], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox

# Running the Genetic Algorithm
def run_ga(toolbox, population_size=50, n_generations=20):
    population = toolbox.population(n=population_size)
    hall_of_fame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2,
                                              ngen=n_generations, stats=stats,
                                              halloffame=hall_of_fame, verbose=True)
    return hall_of_fame[0]

# Main function
def main(symbol, start_date, end_date):
    data = download_data(symbol, start_date, end_date)
    toolbox = setup_ga(data)
    best_individual = run_ga(toolbox)
    best_short_window, best_long_window = best_individual
    print(f"Best parameters: short_window={best_short_window}, long_window={best_long_window}")

# Example usage
if __name__ == "__main__":
    main("AAPL", "2018-01-01", "2023-01-01")
