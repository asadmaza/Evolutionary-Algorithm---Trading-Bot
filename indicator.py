import pandas as pd
import ta
import random
import copy
import random

def mutate_moving_average(params: dict) -> dict:
  '''
  Return a randomly mutated copy of keyword arguments for a moving average indicator.

  Parameters
  ----------
    params : dict
      Keyword arguments assumed to contain a 'window' key.
  '''
  
  mults = [0.9, 1.1]
  return { 'window': max(1, int(params['window'] * random.choice(mults))) } # cannot drop below 1

def moving_average_params() -> dict:
  '''
  Return random, sensible params for the moving average indicator.
  '''

  return { 'window': random.randrange(100) }

'''
Keys of INDICATORS dictionary
----
  name
    Name of the indicator.
  
  indicator
    function that takes close Series and keyword arguments as parameters and returns a pandas Series.

  mutator
    function that takes a dictionary of keyword arguments for that indicator and returns a mutated copy of the dictionary.

  params
    function that returns a random, sensible param dictionary for the indicator.
'''
INDICATORS = [
  { 'name': 'SMA', 'indicator': ta.trend.sma_indicator, 'mutator': mutate_moving_average, 'params': moving_average_params },
  { 'name': 'EMA', 'indicator': ta.trend.ema_indicator, 'mutator': mutate_moving_average, 'params': moving_average_params }
]

NUM_INDICATORS = len(INDICATORS)

def get_indicators(close: pd.Series, params: list[dict]) -> list[pd.Series]:
  '''
  Return a list of indicator Series from above indicators.

  Parameters
  ----------
  close : pd.Series
    Pandas Series of close prices.

  params : list[dict]
    List of dictionaries containing keyword arguments for each indicator.
  '''
  
  return [INDICATORS[i]['indicator'](close, **params[i]) for i in range(NUM_INDICATORS)]

def random_params() -> list[dict]:
  '''
  Return a list of random, sensible params for each indicator.
  '''

  return [INDICATORS[i]['params']() for i in range(NUM_INDICATORS)]

def mutate_params(params: list[dict], prob: float = 1.0) -> list[dict]:
  '''
  Return a copy of params where each param dict was mutated with probability prob, else unchanged.
  Each param dict is mutated by the corresponding mutator function above.

  Parameters:
    params : list[dict]
      List of dictionaries containing keyword arguments for each indicator.

    prob : float
      Probability between [0, 1] of mutation for each dictionary of keyword arguments.
  '''
  
  params = copy.deepcopy(params)
  for i in range(NUM_INDICATORS):
    if random.random() < prob: params[i] = INDICATORS[i]['mutator'](params[i])

  return params