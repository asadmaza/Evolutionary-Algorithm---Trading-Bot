"""
A Strategy that triggers a:
- Buy when the buy_weighted sum of indicators turns from negative to positive, and
- Sell when the sell_weighted sum of indicators turns from negative to positive.

Ideas:
- Could have separate indicators with separate params for buy and sell triggers.
"""

import pickle
from time import sleep
import types
from typing import Any
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
import uuid

import ta

from globals import *
from chromosome import Chromosome, ChromosomeHandler

class Strategy:
  def __init__(
      self,
      candles: pd.DataFrame,
      buy_chromosome: Chromosome = None,
      sell_chromosome: Chromosome = None,
      chromosome_handler: ChromosomeHandler = None,
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

    if (
        buy_chromosome is None or sell_chromosome is None
    ) and chromosome_handler is None:
      pass
      # raise ValueError(
      #     "If buy and or sell chromosome is not provided, a chromosome"
      #     " handler must be provided to generate the missing chromosome(s)"
      # )
    else:
      self.set_chromosome(
          buy_chromosome or chromosome_handler.generate_chromosome(
              is_buy=True), is_buy=True, )
      self.set_chromosome(
          sell_chromosome
          or chromosome_handler.generate_symmetric_chromosome(
              self.buy_chromosome, is_buy=False
          ),
          is_buy=False,
      )

    self.fitness = None
    self.portfolio = self.evaluate()  # evaluate fitness once on init

  def set_chromosome(self, c: Chromosome, is_buy: bool) -> None:
    """Given a chromosome, set all indicators and triggers"""
    indicators = []
    # Create reference to right attribtue name
    if is_buy:
      self.buy_chromosome = c
      self.buy_indicators = indicators
    else:
      self.sell_chromosome = c
      self.sell_indicators = indicators

    # For each indicator, provide respective params and generate DataFrame features
    # Buy and sell triggers call self.indicators, hence we need to generate
    # them
    for i in range(len(c.indicators)):
      params = {}
      # Candle params is a list of olhcv names, replace them with actual Series
      # data
      for candle_name in c.candle_params[i]:
        params[candle_name] = self.candles[candle_name]
      # Convert (arg, value) tuples to dict pairs
      for param in [c.int_params[i], c.float_params[i]]:
        params.update({entry["arg"]: entry["value"] for entry in param})

      # Call this indicator and append its features to buy_indicators or
      # sell_indicators
      indicators.append(c.indicators[i][1](**params))

    # Bind the function of chromosome to self - so it acts like a class method
    if is_buy:
      self.buy_trigger = types.MethodType(c.to_function(), self)
    else:
      self.sell_trigger = types.MethodType(c.to_function(), self)

  def evaluate(
      self, graph: bool = False, fname: str = "strategy.png", title: str = "Best strategy"
  ) -> float:
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
      # for i in range(len(self.buy_chromosome.indicators)):
      #   plt.plot(
      #       self.buy_indicators[i],
      #       label=self.buy_chromosome.indicators[i][0],
      #   )
      # for i in range(len(self.sell_chromosome.indicators)):
      #   plt.plot(
      #       self.sell_indicators[i],
      #       label=self.sell_chromosome.indicators[i][0],
      #   )

    quote = 100  # AUD
    base = 0  # BTC
    bought, sold = 0, 0  # number of times bought and sold
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
      plt.legend(prop={"size": 6})
      plt.title(f"{title}, portfolio = {quote:.2f}")
      plt.savefig(fname, dpi=200)
      plt.clf()
      plt.cla()

    self.portfolio = quote
    return quote

  def get_pickle_data(self) -> dict:
    """
    Return a dict of chromosomes, fitness and portfolio for pickle dumping
    """
    return {
        "buy_chromosome": self.buy_chromosome,
        "sell_chromosome": self.sell_chromosome,
        "fitness": self.fitness,
        "portfolio": self.portfolio,
    }

  @classmethod
  def load_pickle_data(cls, candles, data: dict) -> "Strategy":
    """
    Load the chromosomes, fitness, and portfolio from a pickle data dump (dict)
    """
    s = cls(candles, data["buy_chromosome"], data["sell_chromosome"])
    # s.fitness = Fitness.get_fitness(s)
    s.portfolio = s.evaluate()
    return s

  def __repr__(self) -> str:
    return f"<{self.__class__.__name__} {self.get_pickle_data()}>"


if __name__ == "__main__":
  """
  Testing
  """

  from candle import get_candles

  candles = get_candles()

  # sortinos = []
  # portfolios = []
  # for n in ["sortino50", "sortino100", "portfolio50"]:
  #   for i in range(10):
  #     with open(f'results/{n}/best_strategies_tournament{i}.pkl', "rb") as f:
  #       data = pickle.load(f)
  #     portfolios.append(max([Strategy.load_pickle_data(candles, d).portfolio for d in data]))
  #     sortinos.append(max([Strategy.load_pickle_data(candles, d).fitness for d in data]))
  #   print(n, sum(sortinos) / len(sortinos), sum(portfolios) / len(portfolios))

  best_portfolio = 0
  modules = [ta.momentum]
  handler = ChromosomeHandler(modules)
  strat = Strategy(candles, chromosome_handler=handler)

  while True:
    strat = Strategy(candles, chromosome_handler=handler)

    if strat.portfolio > best_portfolio:
      print(strat)
      strat.evaluate(True)
      best_portfolio = strat.portfolio
      print(f"New best portfolio: {best_portfolio:.2f}\n")
      with open("best_strategy.pck", "wb") as f:
        pickle.dump(strat.get_pickle_data(), f)
