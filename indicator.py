"""
Helper functions to initialise, get and mutate the indicators that are used within a strategy.
"""

from typing import Literal
import pandas as pd
import ta
import random
import copy
import random

# import ta.volatility for Bollinger Bands
import ta.volatility as volatility

random.seed()


# Parameters for Gaussian mutation
# Mean determines most likely value, sigma determines spread from mean
WINDOW_GAUSSIAN_MEAN = 9
WINDOW_GAUSSIAN_SIGMA = 15
CONSTANT_GAUSSIAN_MEAN = 1
CONSTANT_GAUSSIAN_SIGMA = 0.5
WINDOW_DEV_GAUSSIAN_MEAN = 2
WINDOW_DEV_GAUSSIAN_SIGMA = 0.5


def mutate(
    params: list[int | float],
    name: Literal["window", "constant", "window_dev"],
    lower_bound: int | float,
    upper_bound: int | float,
) -> list[int | float]:
    """
    Given list of param values, mutate them and return a new list of mutated params.

    Gaussian mutation is used, where a random value is generated from the Gaussian
    distribution, and added to the current value. `name` of the parameter determines
    Gaussian parameters (mean and sigma).

    Parameters
    ----------
      lower_bound : int | float
        Lower bound for the mutated value.

      upper_bound : int | float
        Upper bound for the mutated value.
    """
    params = copy.deepcopy(params)
    match name:
        case "window":
            mean = WINDOW_GAUSSIAN_MEAN
            sigma = WINDOW_GAUSSIAN_SIGMA
        case "constant":
            mean = CONSTANT_GAUSSIAN_MEAN
            sigma = CONSTANT_GAUSSIAN_SIGMA
        case "window_dev":
            mean = WINDOW_DEV_GAUSSIAN_MEAN
            sigma = WINDOW_DEV_GAUSSIAN_SIGMA
        case _:
            raise ValueError(f"Unknown parameter name: {name}")

    change = random.gauss(mean, sigma)
    change = min(upper_bound, max(lower_bound, change))

    return [x + change for x in params]
