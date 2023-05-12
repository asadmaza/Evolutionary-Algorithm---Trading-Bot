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

  for train in [True, False]:
    candles = train_candles if train else test_candles

    for n in [
        "sortino50",
        "sortino100",
        "sortino200",
        "portfolio50",
        "portfolio100",
            "portfolio200"]:

      best_overall = None  # best bots from training data, save to file

      sortinos = []
      portfolios = []

      for i in range(10):
        with open(f'results/{n}/best_strategies_tournament{i}.pkl', "rb") as f:
          data = pickle.load(f)

        if 'portfolio' in n:
          best = sorted([Strategy.load_pickle_data(candles, d)
                        for d in data], key=lambda s: s.portfolio)[-1]
          if (best_overall is None or best.portfolio > best_overall.portfolio):
            best_overall = best

        else:
          best = sorted([Strategy.load_pickle_data(candles, d)
                        for d in data], key=lambda s: fit.get_fitness(s))[-1]
          if (best_overall is None or fit.get_fitness(
                  best) > fit.get_fitness(best_overall)):
            best_overall = best

        portfolios.append(best.portfolio)
        sortinos.append(fit.get_fitness(best))

      for avg in [True, False]:
        with open(f'results/{"train" if train else "test"}_data_{"avg" if avg else "best"}.csv', 'a') as f:
          if avg:
            f.write(
                f'{n}, {sum(sortinos)/len(sortinos):.4f} {sum(portfolios)/len(sortinos):.4f}\n')
          else:
            f.write(f'{n}, {max(sortinos):.4f} {max(portfolios):.4f}\n')

      if train:
        with open(f'results/best_{n}.pkl', "wb") as f:
          pickle.dump(p := best_overall.get_pickle_data(), f)
