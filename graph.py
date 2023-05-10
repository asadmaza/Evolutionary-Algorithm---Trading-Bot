
from candle import get_candles_split, get_candles
from random_strategy import RandomStrategy
from simple_strategy import SimpleStrategy
import pickle
from strategy import Strategy
from fitness import Fitness

BEST_PORTFOLIO = 'results/portfolio100/best_strategies_tournament{}.pkl'
BEST_SORTINO = 'results/sortino200/best_strategies_tournament{}.pkl'

fit = Fitness()

if __name__ == '__main__':
  _, candles = get_candles_split()

  r = RandomStrategy(candles)
  r.evaluate(graph=True, fname='graphs/random.png',
             title='Random strategy, test data')
  print(f'random {r.portfolio:.4f} {fit.get_fitness(r):.4f}')

  s = SimpleStrategy(candles)
  s.evaluate(graph=True, fname='graphs/simple.png',
             title='Simple strategy, test data')
  print(f'simple {s.portfolio:.4f} {fit.get_fitness(s):.4f}')

  # portfolio
  best = None
  for i in range(10):
    with open(BEST_PORTFOLIO.format(i), "rb") as f:
      data = pickle.load(f)
    strats = [Strategy.load_pickle_data(candles, d) for d in data]
    local_best = sorted(strats, key=lambda s: s.portfolio)[-1]
    if not best or local_best.portfolio > best.portfolio:
      best = local_best

  best.evaluate(
      graph=True,
      fname='graphs/best_portfolio.png',
      title='Best strategy portfolio')

  print(f'{best.portfolio:.4f}')

  # sortino
  best = None
  for i in range(10):
    with open(BEST_SORTINO.format(i), "rb") as f:
      data = pickle.load(f)
    strats = [Strategy.load_pickle_data(candles, d) for d in data]
    local_best = sorted(strats, key=lambda s: fit.get_fitness(s))[-1]
    if not best or fit.get_fitness(local_best) > fit.get_fitness(best):
      best = local_best

  best.evaluate(
      graph=True,
      fname='graphs/best_sortino.png',
      title=f'Best strategy sortino = {fit.get_fitness(best):.4f}')

  print(f'{fit.get_fitness(best):.4f}')
