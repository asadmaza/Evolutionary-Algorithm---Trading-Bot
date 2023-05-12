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

  best_portfolio = None # best bots from training data, save to file
  best_sortino = None

  
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

        if 'portfolio' in n:
          best = sorted([Strategy.load_pickle_data(candles, d) for d in data], key=lambda s: s.portfolio)[-1]
          if train and (best_portfolio is None or best.portfolio > best_portfolio.portfolio): best_portfolio = best
          
        else:
          best = sorted([Strategy.load_pickle_data(candles, d) for d in data], key=lambda s: fit.get_fitness(s))[-1]
          if train and (best_sortino is None or fit.get_fitness(best) > fit.get_fitness(best_sortino)): best_sortino = best
        
        portfolios.append(best.portfolio)
        sortinos.append(fit.get_fitness(best))

      for avg in [True, False]:
        with open(f'results/{"train" if train else "test"}_data_{"avg" if avg else "best"}.csv', 'a') as f:
          if avg:
            f.write(
                f'{n}, {sum(sortinos)/len(sortinos):.4f} {sum(portfolios)/len(sortinos):.4f}\n')
          else:
            f.write(f'{n}, {max(sortinos):.4f} {max(portfolios):.4f}\n')
  
  with open('results/best_portfolio.pkl', "wb") as f:
    pickle.dump(p:=best_portfolio.get_pickle_data(), f)
  with open('results/best_sortino.pkl', "wb") as f:
    pickle.dump(s:=best_sortino.get_pickle_data(), f)

  with open('results/best_chromosome.txt', 'w') as f:
    f.write(f"{p['buy_chromosome']}\n{p['sell_chromosome']}")
