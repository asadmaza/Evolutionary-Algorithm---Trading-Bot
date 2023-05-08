from strategy import Strategy
import random


class RandomStrategy(Strategy):
  def __init__(self, candles, prob=0.05):  # buy or sell roughly once every 20 days
    self.prob = prob
    super(RandomStrategy, self).__init__(candles)

  def buy_trigger(self, t):
    return random.random() < self.prob

  def sell_trigger(self, t):
    return self.buy_trigger(t)


if __name__ == '__main__':
  from candle import get_candles
  candles = get_candles()
  r = RandomStrategy(candles)
  r.evaluate(graph=True)
