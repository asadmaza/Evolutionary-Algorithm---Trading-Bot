"""
Genetic operators: selection, mutation, crossover.
"""

import copy
import random
from typing import Tuple

import numpy as np

from strategy import Strategy
from globals import *

from chromosome import Chromosome, ChromosomeHandler


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
    for int_param in strategy.chromosome.keys():
        # Convert to float to allow adding Gaussian noise
        gene = strategy.buy_chromosome.int_params.astype(np.float64)

        # Elements that should be mutated
        mutation_mask = np.random.random(size=len(gene)) < prob

        sigma = __match_gaussian_sigma()
        changes = np.array([random.gauss(0, sigma) for _ in range(len(gene))])
        gene[mutation_mask] += changes[mutation_mask]

        # strategy.chromosome[name] = __round_and_clip(gene, name)

    # Apply the new chromosome to the strategy
    strategy.set_indicators(strategy.chromosome)
    strategy.fitness = strategy.evaluate()


def __match_range_bounds(name: str):
    """Helper function for rerieving param range bounds"""
    match name:
        case "window_sizes":
            low = 1
            high = INT_OFFSET
            decimal_place = 0
        case "constants":
            low = 0
            high = CONST_MAX
            decimal_place = DECIMAL_PLACES
        case "window_devs":
            low = 1
            high = FLOAT_OFFSET
            decimal_place = DECIMAL_PLACES
        case _:
            raise ValueError(f"Unknown parameter name: {name}")
    return low, high, decimal_place


def __match_gaussian_sigma(name: str):
    """Helper function for mutation, match name to Gaussian sigma"""
    match name:
        case "window_sizes":
            sigma = INT_GAUSSIAN_SIGMA
        case "constants":
            sigma = CONSTANT_GAUSSIAN_SIGMA
        case "window_devs":
            sigma = FLOAT_GAUSSIAN_SIGMA
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


def __expression_crossover(
    c1: Chromosome, c2: Chromosome
) -> Tuple[Chromosome, Chromosome]:
    """Perform 1-point crossover on two chromosomes and return the two children

    Assumes DNF of both chromosomes have more than 1 literals, otherwise
    crossover would simply swap both chromosomes.
    """
    conj_point1 = random.randint(
        1 if len(c1.expression_list) != 1 else 0, len(c1.expression_list) - 1
    )
    conj_point2 = random.randint(0, len(c2.expression_list) - 1)

    child1 = copy.deepcopy(c1)
    child2 = copy.deepcopy(c2)

    child1.expression_list[conj_point1:], child2.expression_list[conj_point2:] = (
        child2.expression_list[conj_point2:],
        child1.expression_list[conj_point1:],
    )

    __match_symbol_lengths(c1, c2, child1, conj_point1, conj_point2)
    __match_symbol_lengths(c2, c1, child2, conj_point2, conj_point1)

    return child1, child2


def __match_symbol_lengths(
    parent1: Chromosome,
    parent2: Chromosome,
    child: Chromosome,
    point1: int,
    point2: int,
):
    """
    Ensure the number of indicators, constants, and params match the number of symbols in the new DNF expression

    `parent1` must have the beginning portion of the child's DNF expression!
    """

    # Get the swapped segments and count A, B, and C symbols in them
    swapped1 = parent1.expression_list[:point1]
    swapped2 = parent2.expression_list[point2:]
    n_A1 = sum(
        [literal.count(child.A) for conjunction in swapped1 for literal in conjunction]
    )
    n_B1 = sum(
        [literal.count(child.B) for conjunction in swapped1 for literal in conjunction]
    )
    n_C1 = sum(
        [literal.count(child.C) for conjunction in swapped1 for literal in conjunction]
    )
    n_A2 = sum(
        [literal.count(child.A) for conjunction in swapped2 for literal in conjunction]
    )
    n_B2 = sum(
        [literal.count(child.B) for conjunction in swapped2 for literal in conjunction]
    )
    n_C2 = sum(
        [literal.count(child.C) for conjunction in swapped2 for literal in conjunction]
    )

    child.indicators = parent1.indicators[:n_A1] + parent2.indicators[-n_A2:]
    child.int_params = parent1.int_params[:n_A1] + parent2.int_params[-n_A2:]
    child.float_params = parent1.float_params[:n_A1] + parent2.float_params[-n_A2:]
    child.candle_params = parent1.candle_params[:n_A1] + parent2.candle_params[-n_A2:]
    child.candle_names = parent1.candle_names[:n_B1] + parent2.candle_names[-n_B2:]
    child.constants = np.append(parent1.constants[:n_C1], parent2.constants[-n_C2:], 0)

    return child


def selection(population: list["Strategy"], n_selections) -> list["Strategy"]:
    """
    Roulette wheel selection, select n_selections individuals from population.
    """
    all_fitness = np.array([s.fitness for s in population])
    all_fitness = all_fitness - np.min(all_fitness)  # Make all fitness positive
    total_fitness = np.sum(all_fitness)

    # Probability of selection is proportional to fitness
    probs = all_fitness / total_fitness
    probs[np.isnan(probs)] = 0  # Set NaN to 0

    return np.random.choice(population, size=n_selections, p=probs)


if __name__ == "__main__":
    ch = ChromosomeHandler()
    c1 = ch.generate_chromosome()
    c2 = ch.generate_chromosome()
    c3, c4 = __expression_crossover(c1, c2)
    print(c3.expression_str.count(c3.A) == len(c3.indicators))
    print(c3.expression_str.count(c3.A) == len(c3.int_params))
    print(c3.expression_str.count(c3.A) == len(c3.float_params))
    print(c3.expression_str.count(c3.A) == len(c3.candle_params))
    print(c3.expression_str.count(c3.B) == len(c3.candle_names))
    print(c3.expression_str.count(c3.C) == len(c3.constants))
    print()
    print(c4.expression_str.count(c4.A) == len(c4.indicators))
    print(c4.expression_str.count(c4.A) == len(c4.int_params))
    print(c4.expression_str.count(c4.A) == len(c4.float_params))
    print(c4.expression_str.count(c4.A) == len(c4.candle_params))
    print(c4.expression_str.count(c4.B) == len(c4.candle_names))
    print(c4.expression_str.count(c4.C) == len(c4.constants))
