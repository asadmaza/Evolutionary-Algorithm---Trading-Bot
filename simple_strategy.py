from strategy import Strategy
import ta

class SimpleStrategy(Strategy):
  def __init__(self, candles, window=20): # buy or sell roughly once every 20 days
    close = candles.iloc[:, 4]  # 5th column is close price

    self.sma = ta.trend.sma_indicator(close, window)
    self.ema = ta.trend.sma_indicator(close, window)

    super(SimpleStrategy, self).__init__(candles)
  
  def buy_trigger(self, t):
    return t>0 and self.ema[t] > self.sma[t] and self.ema[t-1] <= self.sma[t-1]
  
  def sell_trigger(self, t):
    return t>0 and self.sma[t] > self.ema[t] and self.sma[t-1] <= self.ema[t-1]
  
if __name__ == '__main__':
  from candle import get_candles
  candles = get_candles()
  s = SimpleStrategy(candles)
  s.evaluate(graph=True)