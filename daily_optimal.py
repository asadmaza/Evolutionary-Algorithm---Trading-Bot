'''
An optimal bot that trades at the perfect times and is able to gain the maximum profit
'''

from strategy import Strategy
from fitness import Fitness
import indicator

class Optimal(Strategy):
    def __init__(self, candles) -> None:
        super().__init__(candles)

        # Duplicate the first value - fixes the looping
        self.close.loc[-1] = self.close[0]
        self.close.index = self.close.index + 1
        self.close = self.close.sort_index()

        # Removing indicators from graph
        indicator.NUM_INDICATORS = 0

    def buy_trigger(self, t: int) -> bool:
        if (t+1 >= len(self.close)): return False
        return self.close[t+1] > self.close[t]
    
    def sell_trigger(self, t: int) -> bool:
        if (t+1 >= len(self.close)): return False
        return self.close[t+1] < self.close[t]

        
if __name__ == '__main__':
  '''
  Testing
  '''
  
  from candle import get_candles

  candles = get_candles()

  o = Optimal(candles)
  portfolio = o.evaluate(False)
  f = Fitness()
  o.update_fitness(f.get_sharpe_raw(o))
  print(portfolio)