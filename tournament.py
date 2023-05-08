"""
Run a tournament to find the best Strategy through evolution.
"""

import pickle
import random
import math

import ta
from candle import get_candles_split
from globals import timer_decorator
from strategy import Strategy
from operators import crossover, selection, mutation
from fitness import Fitness
from chromosome import ChromosomeHandler

import pandas as pd

import os
import sys

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


class Tournament:
  def __init__(
      self,
      candles: pd.DataFrame,
      size: int,
      num_parents: float,
      num_iterations: int,
      mutation_probability: float = 0.09,
      n_best_individuals: int = 3,
      chromosome_handler: ChromosomeHandler = None,
  ) -> None:
    """
    Parameters
    ----------

    size : int
      Size of the population that stays fixed throughout tournament.

    num_parents : int
      Number of parents to select from the population at each iteration.
      Also the size of population (i.e. children) in next iteration.
      If num_parents < size, migration is used to fill the population.

    mutation_probability: float
      Probability the chromosome of each child will be mutated.

    n_best_individuals: int
        Elitism - number of top performing individuals to keep in next iteration.
    """

    self.size = size
    self.num_parents = int(num_parents * size)
    self.num_iterations = num_iterations
    self.mutation_probability = mutation_probability
    self.n_best_individuals = n_best_individuals
    self.candles = candles

    self.chromosome_handler = chromosome_handler or ChromosomeHandler()

    self.strats = [
        Strategy(candles, chromosome_handler=self.chromosome_handler)
        for _ in range(self.size)
    ]

    self.fitness = Fitness(self.strats, batches=1)

  @timer_decorator
  def play(self) -> None:
    """Complete self.num_iterations of the tournament."""

    self.fitness.update_generation(self.strats)

    for s in self.strats:
      s.fitness = self.fitness.get_fitness(s)
    best_individuals = self.best_strategies(self.n_best_individuals)

    for n_iter in range(self.num_iterations):
      best_individuals = self.run_iteration(n_iter, best_individuals)

    # self.fitness.generate_average_graph()
    # self.fitness.generate_average_graph(type="portfolio")

  @timer_decorator
  def run_iteration(
      self, n_iter: int, best_individuals: list[Strategy]
  ) -> list[Strategy]:
    """Run a single iteration of the tournament."""

    print(f"Iteration: {n_iter}")
    new_pop = selection(self.strats, self.num_parents)
    new_pop = crossover(new_pop)

    # Adaptive mutation probability
    mutation_prob = self.mutation_probability * (
        (self.num_iterations - n_iter) / self.num_iterations
    )
    for s in new_pop:
      if random.uniform(0, 1) < mutation_prob:
        mutation(s.buy_chromosome)
        s.set_chromosome(s.buy_chromosome, is_buy=True)
        s.set_chromosome(
            self.chromosome_handler.generate_symmetric_chromosome(
                s.buy_chromosome
            ),
            is_buy=False,
        )

    new_pop.extend(best_individuals)  # Elitism

    n_migrants = self.size - len(new_pop)
    new_pop.extend(
        [
            Strategy(self.candles, chromosome_handler=self.chromosome_handler)
            for _ in range(n_migrants)
        ]
    )
    self.strats = new_pop

    self.fitness.update_generation(self.strats)
    for s in self.strats:
      s.fitness = self.fitness.get_fitness(s)
    return self.best_strategies(self.n_best_individuals)

  def best_strategies(self, n: int = 1) -> list[Strategy]:
    """Return the best n strategies in the current population."""
    return sorted(self.strats, key=lambda s: s.fitness, reverse=True)[:n]

  def write_best_strategies(
      self, filename: str = "results/best_strategies.pkl", n: int = 1
  ) -> None:
    """
    Write the best n strategies in the current population to a json file.
    """

    with open(filename, "wb") as f:
      pickle.dump([s.get_pickle_data() for s in self.best_strategies(n)], f)

  def load_strategies(
      self,
      filename: str = "results/best_strategies.pkl",
      candles: pd.DataFrame = None,
  ) -> list[Strategy]:
    if candles is None:
      candles = self.candles

    with open(filename, "rb") as f:
      data = pickle.load(f)
    strats = []
    for d in data:
      s = Strategy.load_pickle_data(candles, d)
      strats.append(s)
    return strats


if __name__ == "__main__":
  """
  Testing
  """
  train_candles, test_candles = get_candles_split(0.8)
  modules = [ta.momentum, ta.trend]

  ch = ChromosomeHandler(modules)

  for i in range(10):
    filename = f"results/best_strategies_tournament{i}.pkl"

    # CHANGE THESE SETTINGS
    t = Tournament(
        train_candles,
        size=100,
        num_parents=0.7,
        num_iterations=100,
        mutation_probability=0.3,
    )
    t.play()

    t.write_best_strategies(filename, 10)
