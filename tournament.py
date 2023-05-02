"""
Run a tournament to find the best Strategy through evolution.
"""

from strategy import Strategy
import pandas as pd
import json
from operators import crossover, selection, mutation

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

    def play(self) -> None:
        """Complete self.num_iterations of the tournament."""
        best_individuals = self.best_strategies(self.n_best_individuals)
        for n_iter in range(self.num_iterations):
            print(n_iter)
            new_pop = selection(self.strats, self.num_parents)
            new_pop = crossover(new_pop)
            for s in new_pop:
                mutation(
                    s,
                    self.mutation_probability
                    * ((self.num_iterations - n_iter) / self.num_iterations),
                )
            new_pop.extend(best_individuals)  # Elitism
            best_individuals = self.best_strategies(self.n_best_individuals)

            n_migrants = self.size - len(new_pop)
            new_pop.extend([Strategy(self.candles) for _ in range(n_migrants)])
            self.strats = new_pop

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

    from candle import get_candles

    candles = get_candles()

    t = Tournament(candles, size=100, num_parents=40, num_iterations=30)
    t.play()

    filename = "results/best_strategies.json"

    t.write_best(filename, 10)
    strats = Strategy.from_json(candles, filename)

    for s in strats:
        print(s)
        print(s.evaluate(graph=True))
