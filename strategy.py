"""
A Strategy that triggers a:
- Buy when the buy_weighted sum of indicators turns from negative to positive, and
- Sell when the sell_weighted sum of indicators turns from negative to positive.

Ideas:
- Could have separate indicators with separate params for buy and sell triggers.
"""

import types
from typing import Any
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
import uuid
import inspect

import ta

from globals import *
from dnf import ChromosomeHandler


class Strategy:
    def __init__(
        self,
        candles: pd.DataFrame,
        chromosome: dict[str, Any] = None,
        chromosome_handler: ChromosomeHandler = None,
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

        if chromosome is None and chromosome_handler is None:
            raise ValueError("Must provide either chromosome or chromosome_handler")

        self.set_chromosome(chromosome or chromosome_handler.generate_chromosome())

        self.portfolio = self.evaluate()  # evaluate fitness once on init
        self.fitness = None

    def set_chromosome(self, chromosome: dict[str, int | float]):
        """Given a chromosome, set all indicators and triggers"""
        self.chromosome = chromosome
        if (len(chromosome["indicators"]) != len(chromosome["candle_params"]) or 
        len(chromosome["indicators"]) != len(chromosome["int_params"]) or 
        len(chromosome["indicators"]) != len(chromosome["float_params"])):
            raise ValueError(
                "Chromosome must have same number of indicators and params"
            )

        if len(chromosome["functions"]) != 2 or len(chromosome["expressions"]) != 2:
            raise ValueError("Chromosome must have 2 functions for buy and sell")

        self.n_indicators = len(chromosome["indicators"])
        self.indicators = []
        # For each indicator, provide respective params and generate DataFrame features
        # Buy and sell triggers call self.indicators, hence we need to generate them
        for i in range(self.n_indicators):
            params = {}
            # Candle params is a list of olhcv names, replace them with actual Series data
            for candle_name in chromosome["candle_params"][i]:
                params[candle_name] = self.candles[candle_name]
            # Convert (arg, value) tuples to dict pairs
            for param in [chromosome["int_params"][i], chromosome["float_params"][i]]:
                params.update({entry['arg']: entry['value'] for entry in param})

            self.indicators.append(
                chromosome["indicators"][i][1](**params)
            )

        self.buy_trigger = types.MethodType(chromosome["functions"][0], self)
        self.sell_trigger = types.MethodType(chromosome["functions"][1], self)

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
                plt.plot(
                    self.indicators[i],
                    label=self.chromosome["indicators"][i][0],
                )

        quote = 100  # AUD
        base = 0  # BTC
        bought, sold = 0, 0    # number of times bought and sold
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
            plt.legend(prop={'size': 6})
            plt.savefig("graph.png", dpi=300)
            plt.clf()
            plt.cla()

        self.portfolio = quote
        return quote

    def to_json(self) -> dict:
        """
        Return a dict of the minimum data needed to represent this strategy, as well as the fitness.
        """

        return {
            "indicators": [i[0] for i in self.chromosome["indicators"]],
            "candle_names": self.chromosome["candle_names"],
            "candle_params": [i.tolist() for i in self.chromosome["candle_params"]],
            "int_params": [i.tolist() for i in self.chromosome["int_params"]],
            "float_params": [i.tolist() for i in self.chromosome["float_params"]],
            "constants": self.chromosome["constants"].tolist(),
            "expressions": self.chromosome["expressions"],
            "fitness": self.fitness,
            "portfolio": self.portfolio,
        }

    @classmethod
    def from_json(
        self, candles: pd.DataFrame, filename: str, modules: list, n: int = 1
    ) -> list["Strategy"]:
        """
        Return a list of n Strategy objects from json file data.
        """

        with open(filename, "r", encoding="UTF-8") as f:
            data = json.load(f)
            strategies = []
            for i in range(len(data)):
                indicators = []
                for name in data[i]["indicators"]:
                    for m in modules:
                        try:
                            indocators += [name, getattr(m, name)]
                        except e:
                            continue
                            
                chromosome = {
                    "indicators": indicators,
                    "candle_names": data[i]["candle_names"],
                    "candle_params": np.array(data[i]["candle_params"]),
                    "int_params": np.array(data[i]["int_params"]),
                    "float_params": np.array(data[i]["float_params"]),
                    "constants": np.array(data[i]["constants"]),
                    "expressions": data[i]["expressions"],
                }
                for e in chromosome["expressions"]:
                    e["function"] = ChromosomeHandler.dnf_list_to_function(e)

                self.set_chromosome(candles, data[i]["chromosome"])
                strategies.append(Strategy(candles, data[i]))

            return strategies

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.to_json()}>"


if __name__ == "__main__":
    """
    Testing
    """

    from candle import get_candles

    candles = get_candles()

    best_portfolio = 0
    #modules = [ta.trend, ta.momentum, ta.volatility, ta.volume, ta.others]
    handler = ChromosomeHandler()
    #strat = Strategy.from_json(candles, "best_strategy.json", modules)[0]
    while True:
        c = handler.generate_chromosome()
        strat = Strategy(candles, c)
        if strat.portfolio > best_portfolio:
            best_portfolio = strat.portfolio
            print(f"New best portfolio: {best_portfolio:.2f}\n")
            print(strat)
            strat.evaluate(True)
            with open("best_strategy.json", "w", encoding="UTF-8") as f:
                json.dump(strat.to_json(), f)