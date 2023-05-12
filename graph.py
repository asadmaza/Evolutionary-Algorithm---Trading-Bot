
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
  if len(sys.argv) > 1 and sys.argv[1] != 'test': dataset = 'train'
  else: dataset = 'test'

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
  for n in [
        "sortino50",
        "sortino100",
        "sortino200",
        "portfolio50",
        "portfolio100",
        "portfolio200"]:
    with open(f'results/best_{n}.pkl', 'rb') as f:
      best = Strategy.load_pickle_data(candles, pickle.load(f))

      if n == 'portfolio100':
        with open('results/best_chromosome_portfolio_test.txt', 'w') as fp:
          fp.write(f'{best.buy_chromosome}\n{best.sell_chromosome}')
      if n=='sortino100':
        with open('results/best_chromosome_sortino_test.txt', 'w') as fp:
          fp.write(f'{best.buy_chromosome}\n{best.sell_chromosome}')

      best.evaluate(
          graph=True,
          fname=f'graphs/best_{n}_{dataset}.png',
          title=f'Best strategy {n} {("sortino =" + str(round(fit.get_fitness(best),4))) if "sortino" in n else ""}, {dataset} data')


  # print(f'{fit.get_fitness(best_sortino):.4f} {best_sortino.portfolio:.4f}, {best_sortino.buy_chromosome} {best_sortino.sell_chromosome}')
