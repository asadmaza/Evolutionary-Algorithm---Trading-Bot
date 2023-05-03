"""
Run a tournament to find the best Strategy through evolution.
"""

from candle import get_candles_split
from globals import timer_decorator
from strategy import Strategy
from operators import crossover, selection, mutation
from fitness import Fitness

import pandas as pd

import os
import sys
import json

script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


class Tournament:
  def __init__(
      self,
      candles: pd.DataFrame,
      size: int,
      num_parents: int,
      num_iterations: int,
      mutation_probability: float = 0.5,
      n_best_individuals: int = 3,
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

    if num_parents > size:
      raise ValueError("num_parents must be <= size")

    self.size = size
    self.num_parents = num_parents
    self.num_iterations = num_iterations
    self.mutation_probability = mutation_probability
    self.n_best_individuals = n_best_individuals
    self.candles = candles

    self.strats = [Strategy(candles) for _ in range(self.size)]

    self.fitness = Fitness(self.strats, batches=4)

  @timer_decorator
  def play(self) -> None:
    """Complete self.num_iterations of the tournament."""

    self.fitness.update_generation(self.strats)
    for s in self.strats:
      s.fitness = self.fitness.get_fitness(s)
    best_individuals = self.best_strategies(self.n_best_individuals)

    for n_iter in range(self.num_iterations):
      print(f"Iteration: {n_iter}")
      new_pop = selection(self.strats, self.num_parents)
      new_pop = crossover(new_pop)
      for s in new_pop:
        mutation(
            s,
            self.mutation_probability
            * ((self.num_iterations - n_iter) / self.num_iterations),
        )
      new_pop.extend(best_individuals)  # Elitism

      n_migrants = self.size - len(new_pop)
      new_pop.extend([Strategy(self.candles) for _ in range(n_migrants)])
      self.strats = new_pop

      self.fitness.update_generation(self.strats)
      for s in self.strats:
        s.fitness = self.fitness.get_fitness(s)

      best_individuals = self.best_strategies(self.n_best_individuals)

    self.fitness.generate_average_graph()
    self.fitness.generate_average_graph(type="portfolio")

  def best_strategies(self, n: int = 1) -> list[Strategy]:
    """Return the best n strategies in the current population."""
    return sorted(self.strats, key=lambda s: s.fitness, reverse=True)[:n]

  def write_best(self, filename: str, n: int = 1) -> None:
    """
    Write the best n strategies in the current population to a json file.
    """

    with open(filename, "w") as f:
      json.dump([s.to_json() for s in self.best_strategies(n)], f, indent=2)


if __name__ == "__main__":
  """
  Testing
  """

  train_candles, test_candles = get_candles_split(0.8)

  t = Tournament(train_candles, size=30, num_parents=20, num_iterations=10)
  t.play()

  filename = "results/best_strategies.json"

  t.write_best(filename, 10)
  strat = Strategy.from_json(train_candles, filename)[0]
  strat.evaluate(graph=True)
  print(strat)

  strat = Strategy.from_json(test_candles, filename)[0]
  strat.evaluate(graph=True)
  print(strat)
