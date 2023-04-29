'''
A Strategy that triggers a:
- Buy when the buy_weighted sum of indicators turns from negative to positive, and
- Sell when the sell_weighted sum of indicators turns from negative to positive.

Ideas:
- Could have separate indicators with separate params for buy and sell triggers.
'''

import pandas as pd
from matplotlib import pyplot as plt
import random
import json
import indicator
import uuid

random.seed()

class Strategy():

  def __init__(
    self,
    candles: pd.DataFrame,
    buy_weights: list[float] = None,
    sell_weights: list[float] = None,
    params: list[dict] = None,
    market='BTC/AUD'
  ) -> None:
    '''
    Parameters
    ----------
      candles : pandas.DataFrame
        A DataFrame containing ohlcv data.

      params : list[dict]
        A list of dicts, where each dict contains the keyword arguments to pass to the corresponding indicator. The list of indicators are in indicator.py.

      buy_weights : list[float]
        A list of weights to be applied to each indicator in the buy sum.

      sell_weights : list[float]
        A list of weights to be applied to each indicator in the sell sum.
    '''

    self.id = uuid.uuid4()
    self.close_prices = []

    self.candles = candles
    self.close = self.candles.iloc[:, 4] # 5th column is close price

    self.base, self.quote = market.split('/') # currencies

    self.params = params or indicator.random_params()

    # if not given, weights are initialised randomly between -1 and 1
    self.buy_weights = buy_weights or [random.randrange(-10, 11)/10 for _ in range(indicator.NUM_INDICATORS)]
    self.sell_weights = sell_weights or [random.randrange(-10, 11)/10 for _ in range(indicator.NUM_INDICATORS)]

    self.indicators = indicator.get_indicators(self.candles, self.params)

    # could use these instead
    self.normalised_indicators = [(ind - (m:=ind.min())) / (ind.max() - m) for ind in self.indicators]

    self.fitness = self.evaluate() # evaluate fitness once on init

  def buy_sum(self, t: int) -> float:
    '''
    Return buy_weighted sum of indicators at time period t.

    Parameters
    ----------
      t : int
        The time period to assess. Assumed to be within [1, len(self.close)].
    '''

    # could use self.normalised_indicators here
    return sum([self.buy_weights[i] * self.indicators[i][t] for i in range(indicator.NUM_INDICATORS)])

  def sell_sum(self, t: int) -> float:
    '''
    Return sell_weighted sum of indicators at time period t.
    '''

    return sum([self.sell_weights[i] * self.indicators[i][t] for i in range(indicator.NUM_INDICATORS)])

  def buy_trigger(self, t: int) -> bool:
    '''
    Return True if should buy at time period t, else False.
    Returns True if buy-weighted sum of indicators is positive, else False.
    '''

    return self.buy_sum(t) > 0 and self.buy_sum(t-1) <= 0
  
  def sell_trigger(self, t: int) -> bool:
    '''
    Return True if should sell at time period t, else False.
    Returns True if sell-weighted sum of indicators is positive, else False.
    '''

    return self.sell_sum(t) > 0 and self.sell_sum(t-1) <= 0

  def evaluate(self, graph: bool = False) -> float:
    '''
    Return the fitness of the Strategy, which is defined as the quote currency remaining after:
    - starting with 1 unit of quote currency,
    - buying and selling at each trigger in the timeframe, and
    - selling in the last time period.

    Parameters
    ----------
      graph : bool
        Also plot the close price, indicators, and buy and sell points and block execution
    '''

    if graph:
      plt.plot(self.close, label='Close price')
      for i in range(indicator.NUM_INDICATORS): plt.plot(self.indicators[i], label=indicator.INDICATORS[i].name)

    quote = 100
    total_sum = 100
    base = 0
    bought, sold = 0, 0

    self.close_prices.append(total_sum)

    for t in range(1, len(self.close)):

      if bought == sold and self.buy_trigger(t):
        base += quote / self.close[t]
        total_sum = quote
        self.close_prices.append(total_sum)

        if graph:
          print(f'Bought {base:.2E} {self.base} for {quote:.2f} {self.quote} at time {t:3d}, price {self.close[t]:.2f}')
          plt.plot((t), (self.close[t]), 'o', color='red', label='Buy' if not bought else '')
        quote = 0
        bought += 1

      elif bought > sold and self.sell_trigger(t): # must buy before selling
        quote += base * self.close[t]
        total_sum = quote
        self.close_prices.append(total_sum)

        if graph:
          print(f'Sold   {base:.2E} {self.base} for {quote:.2f} {self.quote} at time {t:3d}, price {self.close[t]:.2f}')
          plt.plot((t), (self.close[t]), 'o', color='green', label='Sell' if not sold else '')
        base = 0
        sold += 1

      else:
        temp = quote + base * self.close[t]
        total_sum = temp
        self.close_prices.append(total_sum)



    # if haven't sold, sell in last time period
    if base:
      quote += base * self.close.iloc[-1]
      total_sum = quote
      self.close_prices.append(total_sum)

      if graph:
        print(f'Sold   {base:.2E} {self.base} for {quote:.2f} {self.quote} at time {t:3d}, price {self.close.iloc[-1]:.2f}')
        plt.plot((len(self.close)-1), (self.close.iloc[-1]), 'o', color='green', label='Sell' if not sold else '')

    if graph:
      plt.legend()
      plt.show(block=True)



    return quote
  
  def mutate(self, weight_prob=0.5, param_prob=0.5) -> 'Strategy':
    '''
    Return a new Strategy with randomly mutated weights and params.
    '''

    # mutate weights
    buy_weights, sell_weights = self.buy_weights.copy(), self.sell_weights.copy()
    ds = [-0.1, 0.1]
    for i in range(indicator.NUM_INDICATORS):
      if random.random() > weight_prob: buy_weights[i] += random.choice(ds)
      if random.random() > weight_prob: sell_weights[i] += random.choice(ds)

    # mutate params
    params = indicator.mutate_params(self.params, param_prob)

    return Strategy(self.candles, buy_weights, sell_weights, params)
  
  def to_json(self) -> dict:
    '''
    Return a dict of the minimum data needed to represent this strategy, as well as the fitness.
    '''

    return { 'buy_weights': self.buy_weights, 'sell_weights': self.sell_weights, 'params': self.params, 'fitness': self.fitness }
  
  @classmethod
  def from_json(self, candles: pd.DataFrame, filename: str, n: int = 1) -> list['Strategy']:
    '''
    Return a list of n Strategy objects from json file data.
    '''

    with open(filename, 'r') as f:
      data = json.load(f)
      return [Strategy(candles, d['buy_weights'], d['sell_weights'], d['params']) for d in data][:n]

  def __repr__(self) -> str:
    return f'<{self.__class__.__name__} {self.to_json()}>'

if __name__ == '__main__':
  '''
  Testing
  '''
  
  from candle import get_candles

  candles = get_candles()

  # example params
  params = [
    {'window': 20}, # SMA
    {'window': 10}, # EMA
  ]
  buy_weights = [-1, 1]
  sell_weights = [1, -1]

  strat1 = Strategy(candles, buy_weights, sell_weights, params)
  print(f'Strategy 1 fitness {strat1.fitness:.2f}\n')

  # random params
  strat2 = Strategy(candles, buy_weights, sell_weights)
  print(f'Strategy 2 fitness {strat2.fitness:.2f}\n')

  # params from json file
  filename = 'results/best_strategies.json'
  strat3 = Strategy.from_json(candles, filename)[0]
  
  print('Strategy 3')
  strat3.evaluate(graph=True)