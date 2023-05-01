'''
Calculate the normalised fitness of a strategy
'''

import statistics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from strategy import Strategy
import matplotlib.pyplot as plt
import copy

class Fitness():
    def __init__(self, strats) -> None:
        self.strats = copy.deepcopy(strats)
        self.generation = 0

        self.fitness = {}

    def update_generation(self, strats):
        self.generation += 1
        self.strats = copy.deepcopy(strats)
        self.fitness[self.generation] = []

    def ROI(self):
        scaler = MinMaxScaler()
        quotes = [s.quote for s in self.strats]
        scaled_quotes = scaler.fit_transform(np.array(quotes).reshape(-1, 1))
        self.roi_quotes = {s.id: q[0] for s, q in zip(self.strats, scaled_quotes)}
    
    def get_ROI_fitness_normalised(self, strat):
        try:
            return self.roi_quotes[strat.id]
        except:
            self.ROI()
            return self.roi_quotes[strat.id]
        
    def get_ROI_raw(self, strat):
        return strat.quote
    
    def get_sharpe_raw(self, strat):
        daily_returns = []
        self.rf = 0.012/365

        for i in range(1, len(strat.close_prices)):
            daily_return = (strat.close_prices[i] - strat.close_prices[i-1]) / (strat.close_prices[i-1])
            daily_returns.append(daily_return)

        avg_daily_return = sum(daily_returns) / len(daily_returns)
        
        std_dev_daily_return = statistics.stdev(daily_returns)
        
        sharpe_ratio = 0
        
        if not std_dev_daily_return == 0:
            sharpe_ratio = (avg_daily_return - self.rf) / std_dev_daily_return

        self.fitness[self.generation].append(sharpe_ratio)

        return sharpe_ratio
    
    
    def generate_generation_graph(self, generation=-1):
        if generation == -1: generation = self.generation
        self.fitness[generation] = sorted(self.fitness[generation])
        plt.plot(self.fitness[generation])
        plt.show()

    def generate_average_graph(self):

        averages = []
        generations = []
        for g in range(1, self.generation+1):
            averages.append(np.average(self.fitness[g]))
            generations.append(g)
        plt.plot(generations, averages, marker="o")
        plt.xticks(range(1, len(generations)+1), map(str, generations))

        plt.show()
        


from candle import get_candles

candles = get_candles()

if __name__ == '__main__':
    # params from json file
    filename = 'results/best_strategies.json'
    s = Strategy.from_json(candles, filename)[0]

    # s.evaluate(graph=True)
    f = Fitness([s])
    f.get_sharpe_raw(s)





