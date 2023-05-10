import pickle
from strategy import Strategy
from fitness import Fitness
from matplotlib import pyplot as plt

if __name__ == "__main__":
  """
  Testing
  """

  from candle import get_candles_split

  train_candles, test_candles = get_candles_split()

  candles = test_candles

  fit = Fitness()

  for avg in [True, False]:
    for train in [True, False]:
      candles = train_candles if train else test_candles

      for n in [
          "sortino50",
          "sortino100",
          "sortino200",
          "portfolio50",
          "portfolio100",
          "portfolio200"]:
        sortinos = []
        portfolios = []

        for i in range(10):
          with open(f'results/{n}/best_strategies_tournament{i}.pkl', "rb") as f:
            data = pickle.load(f)

          best_fitness = sorted([Strategy.load_pickle_data(
              candles, d) for d in data], key=lambda s: fit.get_fitness(s))[-1]
          best_portfolio = sorted([Strategy.load_pickle_data(
              candles, d) for d in data], key=lambda s: s.portfolio)[-1]

          portfolios.append(best_portfolio.portfolio)
          sortinos.append(fit.get_fitness(best_fitness))

        with open(f'results/{"train" if train else "test"}_data_{"avg" if avg else "best"}.csv', 'a') as f:
          if avg:
            f.write(
                f'{n}, {sum(sortinos)/len(sortinos):.4f} {sum(portfolios)/len(sortinos):.4f}\n')
          else:
            f.write(f'{n}, {max(sortinos):.4f} {max(portfolios):.4f}\n')
