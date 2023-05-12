
from candle import get_candles_split, get_candles
from random_strategy import RandomStrategy
from simple_strategy import SimpleStrategy
import pickle
from strategy import Strategy
from fitness import Fitness
import sys

BEST_PORTFOLIO = 'results/portfolio100/best_strategies_tournament{}.pkl'
BEST_SORTINO = 'results/sortino200/best_strategies_tournament{}.pkl'

fit = Fitness()

if __name__ == '__main__':
  if len(sys.argv) > 1 and sys.argv[1] != 'test':
    dataset = 'train'
  else:
    dataset = 'test'

  train, test = get_candles_split()
  candles = train if dataset == 'train' else test

  r = RandomStrategy(candles)
  r.evaluate(graph=True, fname=f'graphs/random_{dataset}.png',
             title=f'Random strategy, {dataset} data')
  print(f'random {r.portfolio:.4f} {fit.get_fitness(r):.4f}')

  s = SimpleStrategy(candles)
  s.evaluate(graph=True, fname=f'graphs/simple_{dataset}.png',
             title=f'Simple strategy, {dataset} data')
  print(f'simple {s.portfolio:.4f} {fit.get_fitness(s):.4f}')

  # portfolio
  with open('results/best_portfolio.pkl', 'rb') as f:
    best_portfolio = Strategy.load_pickle_data(candles, pickle.load(f))

  best_portfolio.evaluate(
      graph=True,
      fname=f'graphs/best_portfolio_{dataset}.png',
      title=f'Best strategy portfolio, {dataset} data')

  # print(f'{best.portfolio:.4f}, {best.buy_chromosome} {best.sell_chromosome}')

  # sortino
  with open('results/best_sortino.pkl', 'rb') as f:
    best_sortino = Strategy.load_pickle_data(candles, pickle.load(f))

  best_sortino.evaluate(
      graph=True,
      fname=f'graphs/best_sortino_{dataset}.png',
      title=f'Best strategy sortino, {dataset} data, sortino = {fit.get_fitness(best_sortino):.4f}')

  print(f'{fit.get_fitness(best_sortino):.4f} {best_sortino.portfolio:.4f}, {best_sortino.buy_chromosome} {best_sortino.sell_chromosome}')
