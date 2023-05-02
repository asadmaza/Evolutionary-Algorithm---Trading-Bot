"""
An optimal bot that trades at the perfect times and is able to gain the maximum profit
"""

from candle import get_candles_split
from strategy import Strategy
from fitness import Fitness


class Optimal(Strategy):
  def __init__(self, candles) -> None:
    super().__init__(candles)

    # Duplicate the first value - fixes the looping
    self.close.loc[-1] = self.close[0]
    self.close.index = self.close.index + 1
    self.close = self.close.sort_index()


<< << << < HEAD
<< << << < HEAD
    def buy_trigger(self, t: int) -> bool:
        if t + 1 >= len(self.close):
            return False
        return self.close[t + 1] > self.close[t]
== == == =
  def buy_trigger(self, t: int) -> bool:
    if t + 1 >= len(self.close):
      return False
    return self.close[t + 1] > self.close[t]
>>>>>> > d3c6011069549fa876415bdcd5f849c88c46ff16

  def sell_trigger(self, t: int) -> bool:
    if t + 1 >= len(self.close):
      return False
    return self.close[t + 1] < self.close[t]

== == == =
    # Removing indicators from graph
    indicator.NUM_INDICATORS = 0

  def buy_trigger(self, t: int) -> bool:
    if (t + 1 >= len(self.close)):
      return False
    return self.close[t + 1] > self.close[t]

  def sell_trigger(self, t: int) -> bool:
    if (t + 1 >= len(self.close)):
      return False
    return self.close[t + 1] < self.close[t]


if __name__ == '__main__':
  '''
  Testing
  '''

  from candle import get_candles
>>>>>>> c8fd3c663235f2ec7342209d49aeeeb86d49ac1c

if __name__ == "__main__":
  """
  Testing
  """

<<<<<<< HEAD
<<<<<<< HEAD
    from candle import get_candles
=======
  from candle import get_candles
>>>>>>> d3c6011069549fa876415bdcd5f849c88c46ff16

  train_candles, test_candles = get_candles_split(0.2)

<<<<<<< HEAD
    o = Optimal(train_candles)
    portfolio = o.evaluate(False)
    f = Fitness(batches=4)
    o.update_fitness(f.get_fitness(o))
    print(portfolio)
=======
=======
>>>>>>> d3c6011069549fa876415bdcd5f849c88c46ff16
  o = Optimal(train_candles)
  portfolio = o.evaluate(False)
  f = Fitness(batches=4)
  o.update_fitness(f.get_fitness(o))
  print(portfolio)
<<<<<<< HEAD
>>>>>>> c8fd3c663235f2ec7342209d49aeeeb86d49ac1c
=======
>>>>>>> d3c6011069549fa876415bdcd5f849c88c46ff16
