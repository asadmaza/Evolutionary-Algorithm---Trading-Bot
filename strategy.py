"""
A Strategy that triggers a:
- Buy when the buy_weighted sum of indicators turns from negative to positive, and
- Sell when the sell_weighted sum of indicators turns from negative to positive.

Ideas:
- Could have separate indicators with separate params for buy and sell triggers.
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
from ta.trend import sma_indicator, ema_indicator
from ta.volatility import bollinger_lband, bollinger_hband
import uuid

from globals import *


class Strategy:
  def __init__(
      self,
      candles: pd.DataFrame,
      chromosome: dict[str, np.ndarray[int | float]] | None = None,
      fitness: int = None,
      market="BTC/AUD",
  ) -> None:
    """
    Init a Strategy with randomised indicator params.

    Parameters
    ----------
      candles : pandas.DataFrame
        A DataFrame containing ohlcv data.
    """

    self.candles = candles
    self.close = self.candles.iloc[:, 4]  # 5th column is close price

    self.base, self.quote = market.split("/")  # currencies

    self.id = uuid.uuid4()
    self.close_prices = []

    # Chromosome = 5 window sizes + 2 window deviations + 6 constants
    self.n_indicators = 5
    self.chromosome = chromosome or Strategy.gen_random_chromosome(
        self.n_indicators, 6, 2
    )
    self.set_indicators(self.chromosome)

    self.portfolio = self.evaluate()  # evaluate portfolio once on init
    self.fitness = fitness

  def set_indicators(self, chromosome: dict[str, int | float]):
    """Given a chromosome, set all indicators."""
    self.sma1 = sma_indicator(
        self.close, window=chromosome["window_sizes"][0], fillna=True
    )
    self.sma2 = sma_indicator(
        self.close, window=chromosome["window_sizes"][1], fillna=True
    )
    self.ema = ema_indicator(
        self.close, window=chromosome["window_sizes"][2], fillna=True
    )
    self.bollinger_lband = bollinger_lband(
        self.close,
        window=chromosome["window_sizes"][3],
        window_dev=chromosome["window_devs"][0],
        fillna=True,
    )
    self.bollinger_hband = bollinger_hband(
        self.close,
        window=chromosome["window_sizes"][4],
        window_dev=chromosome["window_devs"][1],
        fillna=True,
    )

  def buy_trigger(self, t: int) -> bool:
    """
    Return True if should buy at time period t, else False.
    """
    return (
        (self.sma1[t] > self.chromosome["constants"][0] * self.sma2[t])
        or (self.close[t] > self.chromosome["constants"][1] * self.ema[t])
        or (
            self.bollinger_lband[t]
            > self.chromosome["constants"][2] * self.close[t]
        )
    )

  def sell_trigger(self, t: int) -> bool:
    """
    Return True if should sell at time period t, else False.
    """
    return (
        (self.sma2[t] > self.chromosome["constants"][3] * self.sma1[t])
        or (self.ema[t] > self.chromosome["constants"][4] * self.close[t])
        or (
            self.close[t]
            > self.chromosome["constants"][5] * self.bollinger_hband[t]
        )
    )

  def evaluate(self, graph: bool = False) -> float:
    """
    Return the fitness of the Strategy, which is defined as the quote currency remaining after:
    - starting with 1 unit of quote currency,
    - buying and selling at each trigger in the timeframe, and
    - selling in the last time period.

    Parameters
    ----------
      graph : bool
        Also plot the close price, indicators, and buy and sell points and block execution
    """

    if graph:
      plt.plot(self.close, label="Close price")
      for i in range(self.n_indicators):
        plt.plot([self.sma1,
                  self.sma2,
                  self.ema,
                  self.bollinger_lband,
                  self.bollinger_hband,
                  ][i],
                 label=["SMA1",
                        "SMA2",
                        "EMA",
                        "Bollinger Lband",
                        "Bollinger Hband"][i],
                 )

    quote = 100  # AUD
    base = 0  # BTC
    bought, sold = 0, 0
    self.close_prices = [quote]

    for t in range(1, len(self.close)):
      if bought == sold and self.buy_trigger(t):
        base = (quote * 0.98) / self.close[t]
        self.close_prices.append(quote)

        if graph:
          print(
              f"Bought {base:.2E} {self.base} for {quote:.2f} {self.quote} at time {t:3d}, price {self.close[t]:.2f}"
          )
          plt.plot(
              (t),
              (self.close[t]),
              "o",
              color="red",
              label="Buy" if not bought else "",
          )
        quote = 0
        bought += 1

      elif bought > sold and self.sell_trigger(t):  # must buy before selling
        # NOTE: 2% is applied TO the bitcoin!
        quote = (base * 0.98) * self.close[t]
        self.close_prices.append(quote)

        if graph:
          print(
              f"Sold   {base:.2E} {self.base} for {quote:.2f} {self.quote} at time {t:3d}, price {self.close[t]:.2f}"
          )
          plt.plot(
              (t),
              (self.close[t]),
              "o",
              color="chartreuse",
              label="Sell" if not sold else "",
          )
        base = 0
        sold += 1
        # What is the point of this else statement?
      else:
        temp = quote + base * self.close[t]
        self.close_prices.append(temp)

    # if haven't sold, sell in last time period
    if base:
      quote = (base * 0.98) * self.close.iloc[-1]
      self.close_prices.append(quote)
      if graph:
        print(
            f"Sold   {base:.2E} {self.base} for {quote:.2f} {self.quote} at time {t:3d}, price {self.close.iloc[-1]:.2f}"
        )
        plt.plot(
            (len(self.close) - 1),
            (self.close.iloc[-1]),
            "o",
            color="chartreuse",
            label="Sell" if not sold else "",
        )

    if graph:
      plt.legend()
      plt.show(block=True)

    self.portfolio = quote
    return quote

  def to_json(self) -> dict:
    """
    Return a dict of the minimum data needed to represent this strategy, as well as the fitness.
    """

    return {
        "window_sizes": self.chromosome["window_sizes"].tolist(),
        "window_devs": self.chromosome["window_devs"].tolist(),
        "constants": self.chromosome["constants"].tolist(),
        "fitness": self.fitness,
        "portfolio": self.portfolio,
    }

  @classmethod
  def from_json(
      self, candles: pd.DataFrame, filename: str, n: int = 1
  ) -> list["Strategy"]:
    """
    Return a list of n Strategy objects from json file data.
    """

    with open(filename, "r") as f:
      data = json.load(f)
      strategies = []
      for i in range(len(data)):
        data[i]["window_sizes"] = np.array(data[i]["window_sizes"])
        data[i]["window_devs"] = np.array(data[i]["window_devs"])
        data[i]["constants"] = np.array(data[i]["constants"])
        strategies.append(Strategy(candles, data[i], data[i]['fitness']))

      return strategies

  def __repr__(self, format=True) -> str:
    if format:
      window_sizes = [round(size, 3)
                      for size in self.chromosome['window_sizes']]
      window_devs = [round(dev, 3) for dev in self.chromosome['window_devs']]
      constants = [round(c, 3) for c in self.chromosome['constants']]
      portfolio = round(self.portfolio, 2)
      fitness = round(self.fitness, 3)

      return (f"\n--- Strategy ---\n"
              f"Window sizes: {window_sizes}\n"
              f"Window deviations: {window_devs}\n"
              f"Constants: {constants}\n"
              f"Portfolio: {portfolio}\n"
              f"Fitness: {fitness}\n"
              f"----------------\n")

    return f"<{self.__class__.__name__} {self.to_json()}>"

  @staticmethod
  def gen_random_chromosome(n_window: int, n_constant: int, n_window_dev: int):
    return {
        "window_sizes": np.random.randint(1, WIN_MAX, size=n_window),
        "window_devs": np.round(
            np.random.uniform(1, WIN_DEV_MAX, size=n_window_dev), DECIMAL_PLACE
        ),
        "constants": np.round(
            np.random.uniform(0, CONST_MAX, size=n_constant), DECIMAL_PLACE
        ),
    }


if __name__ == "__main__":
  """
  Testing
  """

  from candle import get_candles

  candles = get_candles()

  best_fitness = 0
  # Randomly generate strategies and write the best to a file
  while True:
    strat = Strategy(candles)
    print(f"Strategy fitness {strat.fitness:.2f}\n")
    print(strat)
    strat.evaluate(graph=True)
