'''
Run a tournament to find the best Strategy through evolution.
'''

from strategy import Strategy
import pandas as pd
import numpy as np
import random
import json
from fitness import Fitness
import time

random.seed()

class Tournament():
  
  def __init__(self, candles: pd.DataFrame, size: int, num_parents: int, num_iterations: int) -> None:
    '''
    Parameters
    ----------

    size : int
      Size of the population that stays fixed throughout tournament.

    num_parents : int
      The number of strategies that mutate to create children.
      Also the number of strategies to be discarded each iteration.

    num_iterations : int
      Number of iterations per tournament.
    '''

    self.size = size
    self.num_parents = num_parents # must be less than half the size
    self.num_iterations = num_iterations

    self.strats = []

    buy_weights, sell_weights = [-1, 1], [1, -1]
    for _ in range(self.size // 2): # assume size is even
      self.strats.append(Strategy(candles))
      self.strats.append(Strategy(candles, buy_weights, sell_weights)) # seed half the population with 'good' weights

    self.fitness = Fitness(self.strats)

  def play(self) -> None:
    '''
    Complete self.num_iterations of the tournament.

     - evaluate each strategy,
     - mutate the best (self.num_parents) strategies and add to the pool, and
     - 'kill off' the worst (self.num_parents) strategies.
    '''


    for n in range(self.num_iterations):

      self.fitness.update_generation(self.strats)

      for s in self.strats:
        s.update_fitness(self.fitness.get_sharpe_raw(s))

      self.strats.sort(key=lambda s: s.fitness, reverse=True) # best strategies sorted to the top

      if n < self.num_iterations-1: # Not the last generation
        self.strats = self.strats[:-self.num_parents] # kill off the worst
        self.strats.extend([s.mutate() for s in self.strats[:self.num_parents]]) # add mutations of the best


      # self.fitness.generate_generation_graph()
    self.fitness.generate_average_graph()

      # could also add more random strategies back into the population at each iteration

  def best_strategies(self, n: int = 1) -> list[Strategy]:
    '''
    Return the best n strategies in the current population.
    '''
    return sorted(self.strats, key=lambda s: s.fitness, reverse=True)[:n]

  def write_best(self, filename: str, n: int = 1) -> None:
    '''
    Write the best n strategies in the current population to a json file.
    '''
    with open(filename, 'w') as f:
      json.dump([s.to_json() for s in self.best_strategies(n)], f, indent=2)



def print_strategy(strategy):
    buy_weights = [round(weight, 3) for weight in strategy.buy_weights]
    sell_weights = [round(weight, 3) for weight in strategy.sell_weights]
    params = [{k: round(v, 3) for k, v in param.items()} for param in strategy.params]
    portfolio = round(strategy.portfolio, 2)
    fitness = round(strategy.fitness, 3)
    
    print(f"--- Best Strategy ---\n"
          f"Buy weights: {buy_weights}\n"
          f"Sell weights: {sell_weights}\n"
          f"Parameters: {params}\n"
          f"Portfolio value: {portfolio}\n"
          f"Fitness: {fitness}\n"
          f"----------------\n")

if __name__ == '__main__':
  '''
  Testing
  '''

  from candle import get_candles

  candles = get_candles()

  start = time.time()
  t = Tournament(candles, size=50, num_parents=20, num_iterations=50)
  t.play()
  print(f"Time taken: {time.time() - start}\n")

  filename = 'results/best_strategies.json'

  t.write_best(filename, t.size)
  strat = Strategy.from_json(candles, filename)[0]

  print_strategy(strat)



  