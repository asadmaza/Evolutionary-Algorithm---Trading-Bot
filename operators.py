"""
Genetic operators: selection, mutation, crossover.
"""

import random

import numpy as np

from strategy import Strategy
from globals import *


def mutation(
    strategy: "Strategy",
    prob: float = 0.09,
) -> list[int | float]:
    """
    Given a strategy, mutate each value in each param list with probability `prob`.

    Gaussian mutation is used, where a random value is generated from the Gaussian
    distribution, and added to the current value
    """
    # Go through each param list, and mutate each value with probability `prob`
    for name in strategy.chromosome.keys():
        # Convert to float to allow adding Gaussian noise
        gene = strategy.chromosome[name].astype(np.float64)

        # Elements that should be mutated
        mutation_mask = np.random.random(size=len(gene)) < prob

        sigma = __match_gaussian_sigma(name)
        changes = np.array([random.gauss(0, sigma) for _ in range(len(gene))])
        gene[mutation_mask] += changes[mutation_mask]

        strategy.chromosome[name] = __round_and_clip(gene, name)

    # Apply the new chromosome to the strategy
    strategy.set_indicators(strategy.chromosome)
    strategy.fitness = strategy.evaluate()


def __match_range_bounds(name: str):
    """Helper function for rerieving param range bounds"""
    match name:
        case "window_sizes":
            low = 1
            high = WIN_MAX
            decimal_place = 0
        case "constants":
            low = 0
            high = CONST_MAX
            decimal_place = DECIMAL_PLACE
        case "window_devs":
            low = 1
            high = WIN_DEV_MAX
            decimal_place = DECIMAL_PLACE
        case _:
            raise ValueError(f"Unknown parameter name: {name}")
    return low, high, decimal_place


def __match_gaussian_sigma(name: str):
    """Helper function for mutation, match name to Gaussian sigma"""
    match name:
        case "window_sizes":
            sigma = WINDOW_GAUSSIAN_SIGMA
        case "constants":
            sigma = CONSTANT_GAUSSIAN_SIGMA
        case "window_devs":
            sigma = WINDOW_DEV_GAUSSIAN_SIGMA
        case _:
            raise ValueError(f"Unknown parameter name: {name}")
    return sigma


def __round_and_clip(lst: np.ndarray, name: str):
    """Helper function that rounds and set each value in lst to be within the range of the parameter"""
    lbound, ubound, decimal_place = __match_range_bounds(name)
    np.round(lst, decimal_place, out=lst)
    np.clip(lst, lbound, ubound, out=lst)
    if decimal_place == 0:
        lst = lst.astype(int)
    return lst


def crossover(parents: list["Strategy"]) -> list["Strategy"]:
    """Given a list of Strategies to mate, return a list of offspring of same size

    Crossover is done by taking the average of the parameters of the parents.
    """
    offspring = []
    while len(offspring) < len(parents):
        p1, p2 = random.choices(parents, k=2)
        child = {}
        for name in p1.chromosome.keys():
            child[name] = (p1.chromosome[name] + p2.chromosome[name]) / 2
            child[name] = __round_and_clip(child[name], name)
        offspring.append(Strategy(p1.candles, chromosome=child))
    return offspring


def selection(population: list["Strategy"], n_selections) -> list["Strategy"]:
    """
    Roulette wheel selection, select n_selections individuals from population.
    """
    all_fitness = np.array([s.fitness for s in population])
    all_fitness = all_fitness - np.min(all_fitness)  # Make all fitness positive
    total_fitness = np.sum(all_fitness)

    # Probability of selection is proportional to fitness
    probs = all_fitness / total_fitness

    return np.random.choice(population, size=n_selections, p=probs)


def gen_random_chromosome(n_window: int, n_constant: int, n_window_dev: int):
    """Generate random chromosome using settings from globals.py"""
    return {
        "window_sizes": np.random.randint(1, WIN_MAX, size=n_window),
        "window_devs": np.round(
            np.random.uniform(1, WIN_DEV_MAX, size=n_window_dev), DECIMAL_PLACE
        ),
        "constants": np.round(
            np.random.uniform(0, CONST_MAX, size=n_constant), DECIMAL_PLACE
        ),
    }
