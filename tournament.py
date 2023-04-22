'''
Run a tournament to find the best Strategy through evolution.
'''

from strategy import Strategy
import pandas as pd
import random
random.seed()

class Tournament():
  
  def __init__(self, candles: pd.DataFrame, size: int, num_parents: int, num_iterations: int):
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

    # possible initial window sizes
    windows = range(100)

    self.strats = []
    for _ in range(self.size):
      params = [
        {'window': random.choice(windows)},
        {'window': random.choice(windows)},
      ]
      buy_weights = [-1, 1]
      sell_weights = [1, -1]
      self.strats.append(Strategy(candles, params, buy_weights, sell_weights))

  def play(self) -> None:
    '''
    Complete self.num_iterations of the tournament.

     - evaluate each strategy,
     - mutate the best (self.num_parents) strategies and add to the pool, and
     - 'kill off' the worst (self.num_parents) strategies.
    '''

    for _ in range(self.num_iterations):
      self.strats.sort(key=lambda s: s.evaluate())
      self.strats.extend([s.mutate() for s in self.strats[self.size-self.num_parents:]]) # add mutations of the best
      self.strats = self.strats[self.num_parents:] # kill off the worst

  def best_strategies(self, n=1) -> Strategy:
    '''
    Return the best strategy in the current population.
    '''

    return sorted(self.strats, key=lambda s: s.evaluate(), reverse=True)[:n]

  def write_best(self, filename, n=1):
    '''
    Write the best n strategies in the current population to a json file
    '''

    with open(filename, 'w') as f:
      json.dump([s.to_json() for s in self.best_strategies(n)], f)

if __name__ == '__main__':
  '''
  Testing
  '''
  
  import json
  from candles import get_candles

  candles = get_candles()

  t = Tournament(candles, size=50, num_parents=10, num_iterations=10)
  t.play()

  filename = 'results/best.json'

  t.write_best(filename)
  strat = Strategy.from_json(candles, filename)[0]

  print(strat)
  print(strat.evaluate())



  