'''
Helper functions to initialise, get and mutate the indicators that are used within a strategy.
'''

import pandas as pd
import ta
import random
import copy
import random

random.seed()


class Indicator():
  def __init__(
      self,
      name: str,
      ind_fn,  # function
      columns: list[str],
      mutator_fn,  # function,
      random_params_fn,  # function
  ) -> None:
    '''
    Store all the data that represents an Indicator.

    Parameters
    ----------
      name : str
        Name of the indicator.

      ind_fn : function
        Function that returns an indicator Series.

      columns : list[str]
        Ordered list of ohlcv column names to pass as the first constant parameters to ind_fn. Column names are found in candle.py.

      mutator_fn : function
        Function that takes a dictionary of parameters to ind_fn and returns a mutated copy of them.

      random_params_fn : function
        Function that returns a dictionary of random, sensible parameters to ind_fn.
    '''

    self.name = name
    self.ind_fn = ind_fn
    self.columns = columns
    self.mutator_fn = mutator_fn
    self.random_params_fn = random_params_fn

# ----- Indicator-specific mutator functions ----------


def mutate_ma(params: dict) -> dict:
  '''
  Return a randomly mutated copy of keyword arguments for a moving average indicator.

  Parameters
  ----------
    params : dict
      Keyword arguments, assumed to contain only a 'window' key.
  '''

  mults = [0.9, 1.1]
  return {
      'window': max(
          1,
          int(
              params['window'] *
              random.choice(mults)))}  # cannot drop below 1

# ----- Indicator-specific random params functions -----


def random_ma_params() -> dict:
  '''
  Return random, sensible params for the moving average indicator.
  '''

  return {'window': random.randrange(1, 100)}

# ------------------------------------------------------


INDICATORS = [
    Indicator(
        'SMA',
        ta.trend.sma_indicator,
        ['close'],
        mutate_ma,
        random_ma_params),
    Indicator(
        'EMA',
        ta.trend.ema_indicator,
        ['close'],
        mutate_ma,
        random_ma_params),

    # Indicator('ADX', ta.trend.adx, ['high', 'low', 'close'], )
]

NUM_INDICATORS = len(INDICATORS)


def get_indicators(candles: pd.DataFrame,
                   params: list[dict]) -> list[pd.Series]:
  '''
  Return a list of indicator Series from the above indicators in INDICATORS.

  Parameters
  ----------
  candles : pd.DataFrame
    ohlcv data.

  params : list[dict]
    List of dictionaries containing keyword arguments for each indicator.
  '''

  return [INDICATORS[i].ind_fn(
      *[candles[c] for c in INDICATORS[i].columns], **params[i]) for i in range(NUM_INDICATORS)]


def random_params() -> list[dict]:
  '''
  Return a list of random, sensible params for each indicator by calling its random_params_fn.
  '''

  return [ind.random_params_fn() for ind in INDICATORS]


def mutate_params(params: list[dict], prob: float = 1.0) -> list[dict]:
  '''
  Return a copy of params where each param dict was mutated with probability prob by the indicator's mutator_fn, else unchanged.

  Parameters:
    params : list[dict]
      List of dictionaries containing keyword arguments for each indicator.

    prob : float
      Probability between [0, 1] of mutation for each dictionary of keyword arguments.
  '''

  params = copy.deepcopy(params)
  for i in range(NUM_INDICATORS):
    if random.random() < prob:
      params[i] = INDICATORS[i].mutator_fn(params[i])

  return params
