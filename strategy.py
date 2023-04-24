'''
A Strategy that triggers:
- a buy when the buy_weighted sum of indicators turns from negative to positive.
- a sell when the sell_weighted sum of indicators turns from negative to positive.

Questions for lab facilitator:
- Is 300-700, 1 day candle data points all that is required to evolve strategy? Alternatively we can fetch data in many batches and store in a file.
- Why are all the prices between 0.26-1.24 as opposed to around $30,000 USD? if they are normalised, why isn't the min and max of a dataframe 0 and 1?
'''

import pandas as pd
from matplotlib import pyplot as plt
import ta
import random
import copy
import json

random.seed()

class Strategy():

  # callables that take close Series as first parameter and return a indicator Series
  INDICATORS = [
    ta.trend.sma_indicator,
    ta.trend.ema_indicator,
    # ... simply add more indicators
  ]
  NUM_INDICATORS = len(INDICATORS)

  def __init__(self, candles: pd.DataFrame, params: list[dict], buy_weights: list[float], sell_weights: list[float]) -> None:
    '''
    Parameters
    ----------
      candles : pandas.DataFrame
        A DataFrame containing ohlcv data.

      params : list[dict]
        A list of dicts, where each dict contains the keyword arguments to pass to the corresponding indicator.

      buy_weights : list[float]
        A list of weights to be applied to each indicator in the buy sum.

      sell_weights : list[float]
        A list of weights to be applied to each indicator in the sell sum.
    '''

    self.candles = candles
    self.close = self.candles.iloc[:, 4] # 5th column is close price

    self.params = params
    self.buy_weights = buy_weights
    self.sell_weights = sell_weights

    self.indicators = [Strategy.INDICATORS[i](self.close, **self.params[i]) for i in range(Strategy.NUM_INDICATORS)]

    self.fitness = self.evaluate() # evaluate fitness once on init

  def buy_sum(self, t: int) -> float:
    '''
    Return buy_weighted sum of indicators at time period t.

    Parameters
    ----------
      t : int
        The time period to assess. Assumed to be within [1, len(self.close)].
    '''

    return sum([self.buy_weights[i] * self.indicators[i][t] for i in range(Strategy.NUM_INDICATORS)])

  def sell_sum(self, t: int) -> float:
    '''
    Return sell_weighted sum of indicators at time period t.

    Parameters
    ----------
      t : int
        The time period to assess. Assumed to be within [1, len(self.close)].
    '''

    return sum([self.sell_weights[i] * self.indicators[i][t] for i in range(Strategy.NUM_INDICATORS)])

  def buy_trigger(self, t: int) -> bool:
    '''
    Return True if should buy at time period t, else False.

    Parameters
    ----------
      t : int
        The time period to assess. Assumed to be within [1, len(self.close)].
    '''

    return self.buy_sum(t) > 0 and self.buy_sum(t-1) <= 0
  
  def sell_trigger(self, t: int) -> bool:
    '''
    Return True if should sell at time period t, else False.

    Parameters
    ----------
      t : int
        The time period to assess. Assumed to be within [1, len(self.close)].
    '''

    return self.sell_sum(t) > 0 and self.sell_sum(t-1) <= 0

  def evaluate(self, graph: bool = False) -> float:
    '''
    Return the fitness of the Strategy, which is defined as the USD remaining after:
    - starting with $1 USD,
    - buying and selling at each trigger in the timeframe, and
    - selling in the last time period.

    Parameters
    ----------
      graph : bool
        Also plot the close price, indicators, and buy and sell points and block execution
    '''

    if graph:
      plt.plot(self.close, label='Close price')
      for i in range(Strategy.NUM_INDICATORS): plt.plot(self.indicators[i], label=Strategy.INDICATORS[i].__name__)

    usd = 1
    bitcoin = 0
    bought, sold = 0, 0

    for t in range(1, len(self.close)):

      if self.buy_trigger(t):
        bitcoin += usd / self.close[t]
        if graph:
          print(f'Bought {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at time {t:3d}, price {self.close[t]:4.2f}')
          plt.plot((t), (self.close[t]), 'o', color='red', label='Buy' if not bought else '')
        usd = 0
        bought += 1

      elif bitcoin and self.sell_trigger(t): # must buy before selling
        usd += bitcoin * self.close[t]
        if graph:
          print(f'Sold   {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at time {t:3d}, price {self.close[t]:4.2f}')
          plt.plot((t), (self.close[t]), 'o', color='green', label='Sell' if not sold else '')
        bitcoin = 0
        sold += 1

    # if haven't sold, sell in last time period
    if bitcoin:
      usd += bitcoin * self.close.iloc[-1]
      if graph:
        print(f'Sold   {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at time {t:3d}, price {self.close.iloc[-1]:4.2f}')
        plt.plot((len(self.close)-1), (self.close.iloc[-1]), 'o', color='green', label='Sell' if not sold else '')

    if graph:
      plt.legend()
      plt.show(block=True)
    
    return usd
  
  def mutate(self) -> 'Strategy':
    '''
    Return a new Strategy with randomly mutated params.
    
    Initial strategy:
    - multiply each param by either 0.9 or 1.1 and cast to int.
    '''

    mults = [0.9, 1.1]
    params = copy.deepcopy(self.params)

    for kwargs in params:
      for k, v in kwargs.items():
        kwargs[k] = int(v * random.choice(mults))

    return Strategy(self.candles, params, self.buy_weights, self.sell_weights)
  
  def to_json(self) -> dict:
    '''
    Return a dict of the minimum data needed to represent this strategy.
    '''

    return {'params': self.params, 'buy_weights': self.buy_weights, 'sell_weights': self.sell_weights}
  
  @classmethod
  def from_json(self, candles: pd.DataFrame, filename: str, n: int = 1) -> list['Strategy']:
    '''
    Return a list of n Strategy objects from json file data.

    Parameters
    ----------
      candles : pandas.DataFrame
        A DataFrame containing ohlcv data.

      filename : str
        Name of json file.

      n : int
        Number of Strategies to read from file.
    '''

    with open(filename, 'r') as f:
      data = json.load(f)
      return [Strategy(candles, *d.values()) for d in data]

if __name__ == '__main__':
  '''
  Testing
  '''
  
  from candles import get_candles

  candles = get_candles()

  # --- imitate simple strategy ---
  params = [
    {'window': 20}, # SMA
    {'window': 10}, # EMA
  ]
  buy_weights = [-1, 1]
  sell_weights = [1, -1]
  # -------------------------------

  strat1 = Strategy(candles, params, buy_weights, sell_weights)
  print(f'Strategy 1 fitness {strat1.evaluate():.2f}')

  filename = 'results/best.json'
  strat2 = Strategy.from_json(candles, filename)[0]

  fitness = strat2.evaluate(graph=True)