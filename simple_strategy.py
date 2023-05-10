from strategy import Strategy
import ta
from matplotlib import pyplot as plt

class SimpleStrategy(Strategy):
  def __init__(self, candles):  # buy or sell roughly once every 20 days
    
    self.close = candles.iloc[:, 4]  # 5th column is close price

    self.sma = ta.trend.sma_indicator(self.close, 20)
    self.ema = ta.trend.ema_indicator(self.close, 20)

    super(SimpleStrategy, self).__init__(candles)

  def buy_trigger(self, t):
    return t > 0 and self.ema[t] > self.sma[t] and self.ema[t -
                                                            1] <= self.sma[t - 1]

  def sell_trigger(self, t):
    return t > 0 and self.sma[t] > self.ema[t] and self.sma[t -
                                                            1] <= self.ema[t - 1]

  def graph(self):
    plt.plot(self.close, label='close')
    plt.plot(self.sma, label='sma')
    plt.plot(self.ema, label='ema')
    plt.legend()
    plt.show(block=True)

if __name__ == '__main__':
  from candle import get_candles, get_candles_split
  from fitness import Fitness

  _, candles = get_candles_split()
  #candles = get_candles()
  s = SimpleStrategy(candles)

  print(Fitness().get_fitness(s))
  print(s.portfolio)
  s.evaluate(graph=True, fname='graphs/simple.png', title='Simple strategy, test data')
