"""
Calculate the normalised fitness of a strategy
"""

from candle import get_candles
import statistics
from typing import Literal
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from strategy import Strategy
import matplotlib.pyplot as plt
import copy
import math


class Fitness:
    def __init__(self, strats=[], batches=1) -> None:
        self.strats = copy.deepcopy(strats)
        self.generation = 0
        self.batches = batches

        self.fitness = {}
        self.portfolio = {}

        self.single = not len(strats)

    def get_fitness(self, strat):
        fitnesses = []
        for n in range(self.batches):
            fitnesses.append(self.get_fitness_with_batch(strat, n))

        fitness = np.average(fitnesses)

        if not self.single:
            self.update_values(fitness, strat.portfolio)

        return fitness

    def get_fitness_with_batch(self, strat, n):
        s = copy.deepcopy(strat)
        l = len(strat.close)
        start = int(l / self.batches * n)
        end = int(l / self.batches * (n + 1))

        s.close_prices = s.close_prices[start:end]
        s.close = s.close[start:end]

        # fitness = self.get_sortino_raw(strat) + self.get_sharpe_raw(strat) + 0.01*self.get_ROI_fitness_normalised(strat)

        fitness = self.get_sortino_raw(strat)

        return fitness

    def update_values(self, fitness, portfolio):
        self.fitness[self.generation].append(fitness)
        self.portfolio[self.generation].append(portfolio)

    def update_generation(self, strats):
        self.generation += 1
        self.strats = copy.deepcopy(strats)
        self.fitness[self.generation] = []
        self.portfolio[self.generation] = []

    def ROI(self):
        scaler = MinMaxScaler()
        portfolios = [s.portfolio for s in self.strats]
        scaled_quotes = scaler.fit_transform(np.array(portfolios).reshape(-1, 1))
        self.roi_quotes = {s.id: q[0] for s, q in zip(self.strats, scaled_quotes)}

    def get_ROI_fitness_normalised(self, strat):
        try:
            return self.roi_quotes[strat.id]
        except BaseException:
            self.ROI()
            return self.roi_quotes[strat.id]

    def get_ROI_raw(self, strat):
        return strat.portfolio

    def get_sharpe_raw(self, strat):
        daily_returns = []
        self.rf = 0.012 / 365

        for i in range(1, len(strat.close_prices)):
            daily_return = (strat.close_prices[i] - strat.close_prices[i - 1]) / (
                strat.close_prices[i - 1]
            )
            daily_returns.append(daily_return)

        avg_daily_return = sum(daily_returns) / len(daily_returns)

        std_dev_daily_return = statistics.stdev(daily_returns)

        sharpe_ratio = 0

        if not std_dev_daily_return == 0:
            sharpe_ratio = (avg_daily_return - self.rf) / std_dev_daily_return

        return sharpe_ratio

    def get_sortino_raw(self, strat):
        daily_returns = []
        self.rf = 0.012 / 365
        target_return = 0

        for i in range(1, len(strat.close_prices)):
            daily_return = (strat.close_prices[i] - strat.close_prices[i - 1]) / (
                strat.close_prices[i - 1]
            )
            daily_returns.append(daily_return)

        avg_daily_return = sum(daily_returns) / len(daily_returns)

        downside_deviation = 0
        for return_value in daily_returns:
            if return_value < target_return:
                downside_deviation += (return_value - target_return) ** 2

        downside_deviation = math.sqrt(downside_deviation / len(daily_returns))

        sortino_ratio = 0

        if not downside_deviation == 0:
            sortino_ratio = (avg_daily_return - self.rf) / downside_deviation

        return sortino_ratio

    def generate_generation_graph(
        self, generation=-1, type: Literal["fitness", "porfolio"] = "fitness"
    ):
        data = self.fitness
        if type == "portfolio":
            data = self.portfolio

        if generation == -1:
            generation = self.generation
        data[generation] = sorted(data[generation])
        plt.plot(data[generation])
        plt.show()

    def generate_average_graph(self, type: Literal["fitness", "porfolio"] = "fitness"):
        data = self.fitness
        if type == "portfolio":
            data = self.portfolio

        averages = []
        generations = []
        for g in range(1, self.generation + 1):
            averages.append(np.average(data[g]))
            generations.append(g)
        plt.plot(generations, averages, marker="o")
        plt.xticks(range(1, len(generations) + 1), map(str, generations))

        plt.show()


candles = get_candles()

if __name__ == "__main__":
    # params from json file
    filename = "results/best_strategies.json"
    s = Strategy.from_json(candles, filename)[0]

    # s.evaluate(graph=True)
    f = Fitness([s])
    f.get_fitness(s)
