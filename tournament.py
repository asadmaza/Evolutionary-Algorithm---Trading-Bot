"""
Run a tournament to find the best Strategy through evolution.
"""

import copy
from strategy import Strategy
import pandas as pd
import random
import json
import numpy as np

random.seed()


class Tournament:
    def __init__(
        self, candles: pd.DataFrame, size: int, num_parents: int, num_iterations: int
    ) -> None:
        """
        Parameters
        ----------

        size : int
          Size of the population that stays fixed throughout tournament.

        num_parents : int
          The number of strategies that mutate to create children.
          Also the number of strategies to be discarded each iteration.

        num_iterations : int
          Number of iterations per tournament.
        """

        self.size = size
        self.num_parents = num_parents  # must be less than half the size
        self.num_iterations = num_iterations

        self.strats = []

        buy_weights, sell_weights = [-1, 1], [1, -1]
        for _ in range(self.size // 2):  # assume size is even
            self.strats.append(Strategy(candles))
            self.strats.append(
                Strategy(candles, buy_weights, sell_weights)
            )  # seed half the population with 'good' weights

    def play(self) -> None:
        """
        Complete self.num_iterations of the tournament.

         - evaluate each strategy,
         - mutate the best (self.num_parents) strategies and add to the pool, and
         - 'kill off' the worst (self.num_parents) strategies.
        """

        for _ in range(self.num_iterations):
            new_pop = self.selection(n_selections=self.num_parents)
            new_pop = self.crossover(new_pop)
            new_pop = self.mutation(new_pop)
            self.strats.extend(
                [s.mutate() for s in self.strats[: self.num_parents]]
            )  # add mutations of the best

            # could also add more random strategies back into the population at each iteration

    def best_strategies(self, n: int = 1) -> list[Strategy]:
        """
        Return the best n strategies in the current population.
        """

        return sorted(self.strats, key=lambda s: s.evaluate(), reverse=True)[:n]

    def selection(self, n_selections: int = 1) -> list[Strategy]:
        """
        Roulette wheel selection.
        """
        total_fitness = sum([s.fitness for s in self.strats])
        # Probability of selection is proportional to fitness
        probs = [s.fitness / total_fitness for s in self.strats]
        return np.random.choice(self.strats, size=n_selections, p=probs)

    def crossover(self, parents: list[Strategy]) -> list[Strategy]:
        """
        Given a list of parents, return same number of offspring from crossover.
        """
        offspring = []
        while len(offspring) < len(parents):
            p1, p2 = random.choices(parents, k=2)
            params = copy.deepcopy(p1.params)
            for key in params.keys():
                params[key] = (params[key] + p2.params[key]) / 2
            offspring.append(Strategy(p1.candles, params=params))
        pass

    def mutation(self, parents: list[Strategy]) -> list[Strategy]:
        pass

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

    t = Tournament(candles, size=50, num_parents=20, num_iterations=10)
    t.play()

    filename = "results/best_strategies.json"

    t.write_best(filename, t.size)
    strat = Strategy.from_json(candles, filename)[0]

    print(strat)
    print(strat.evaluate())
