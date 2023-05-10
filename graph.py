
from candle import get_candles_split, get_candles
from random_strategy import RandomStrategy
from simple_strategy import SimpleStrategy
import pickle
from strategy import Strategy

BEST = 'results/sortino100/best_strategies_tournament{}.pkl'


if __name__ == '__main__':
  _, candles = get_candles_split()

  r = RandomStrategy(candles)
  r.evaluate(graph=True, fname='graphs/random.png', title='Random strategy, test data')

  s = SimpleStrategy(candles)
  s.evaluate(graph=True, fname='graphs/simple.png', title='Simple strategy, test data')

  best = None
  for i in range(10):
    with open(BEST.format(i), "rb") as f: data = pickle.load(f)
    strats = [Strategy.load_pickle_data(candles, d) for d in data]
    local_best = sorted(strats, key=lambda s: s.portfolio)[-1]
    if not best or local_best.portfolio > best.portfolio: best = local_best
  
  best.evaluate(graph=True, fname='graphs/best.png', title='Best strategy, test data')