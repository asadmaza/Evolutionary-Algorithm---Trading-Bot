'''
Simple Strategy:
- Buy when exponential moving average (EMA) overtakes smooth moving average (SMA) 
- Sell when SMA overtakes EMA

TODO
- Convert this SimpleStrategy class into a general Strategy class that can change its strategy based on its parameters

Questions for lab facilitator:
- Is 720, 1 day data points all that is required to evolve strategy?
- Why is price between 0.26-1.24 as opposed to around $30,000 USD?
'''

import ta
import pandas as pd
from matplotlib import pyplot as plt

class SimpleStrategy():

  def __init__(self, ohlcv: pd.DataFrame, sma_window: int = 20, ema_window: int = 10) -> None:
    '''
    Parameters
    ----------
      ohlcv : pandas.DataFrame
        A DataFrame containing ohlcv candle data.
    '''

    self.close_prices = ohlcv.iloc[:, 4]

    # indicators
    self.sma = ta.trend.sma_indicator(self.close_prices, sma_window)
    self.ema = ta.trend.ema_indicator(self.close_prices, ema_window)

  def buy_trigger(self, t: int) -> bool:
    '''
    Return True if should buy at time period t, else False.

    Parameters
    ----------
    t : int
      The time period to assess. Assumed to be within [1, len(self.close_prices)]
    '''

    return self.ema[t] > self.sma[t] and self.ema[t-1] <= self.sma[t-1]
  
  def sell_trigger(self, t: int) -> bool:
    '''
    Return True if should sell at time period t, else False.

    Parameters
    ----------
    t : int
      The time period to assess. Assumed to be within [1, len(self.close_prices)]
    '''

    return self.sma[t] > self.ema[t] and self.sma[t-1] <= self.ema[t-1]

  def evaluate(self, verbose: bool = False) -> float:
    '''
    Return the fitness of the Strategy, which is defined as the USD remaining after starting with $1 USD, buying and selling at each trigger in the timeframe, and selling in the last time period.
    '''

    usd = 1
    bitcoin = 0

    for t in range(1, len(self.close_prices)):

      if self.buy_trigger(t):
        bitcoin += usd / self.close_prices[t]
        if verbose: print(f'Bought {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at time {t:3d}, price {self.close_prices[t]:4.2f}')
        usd = 0

      elif bitcoin and self.sell_trigger(t): # must buy before selling
        usd += bitcoin * self.close_prices[t]
        if verbose: print(f'Sold   {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at time {t:3d}, price {self.close_prices[t]:4.2f}')
        bitcoin = 0

    # if haven't sold, sell in last time period
    if bitcoin:
      usd += bitcoin * self.close_prices.iloc[-1]
      if verbose: print(f'Sold   {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at time {t:3d}, price {self.close_prices.iloc[-1]:4.2f}')

    return usd
  
  def graph(self) -> None:
    '''
    Graph the close prices, the Strategy's indicators, and the buy and sell points. Block until figure is closed.
    '''
    
    plt.plot(self.close_prices, color='grey', label='Close price')
    plt.plot(self.sma, color='blue', label='SMA')
    plt.plot(self.ema, color ='orange', label='EMA')

    bought, sold = 0, 0

    for t in range(1, len(self.close_prices)):

        if self.buy_trigger(t):
            plt.plot((t), (self.close_prices[t]), 'o', color='red', label='Buy' if not bought else '')
            bought += 1

        elif bought and self.sell_trigger(t): # must buy before selling
            plt.plot((t), (self.close_prices[t]), 'o', color='green', label='Sell' if not sold else '')
            sold += 1

    # if haven't sold, sell in last time period
    if bought > sold:
      plt.plot((len(self.close_prices)-1), (self.close_prices.iloc[-1]), 'o', color='green', label='Sell' if not sold else '')

    plt.legend()
    plt.show(block=True)


if __name__ == '__main__':
  import ccxt

  MARKET = 'BIT/USD'
  TIMEFRAME = '30m'

  kraken = ccxt.kraken()
  ohlcv = pd.DataFrame(kraken.fetch_ohlcv(MARKET, TIMEFRAME))

  strat = SimpleStrategy(ohlcv)
  
  strat.evaluate(verbose=True)
  strat.graph()

