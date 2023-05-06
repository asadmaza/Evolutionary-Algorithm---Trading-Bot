"""
Genetic operators: selection, mutation, crossover.
"""

import copy
import random
from typing import Literal, Tuple

import numpy as np

from strategy import Strategy
from globals import *

from chromosome import Chromosome, ChromosomeHandler


def mutation(chromosome: Chromosome) -> None:
    """Randomly select a mutation to apply to the given chromosome.

    Mutations include: shuffle constants, mutate numeric indicator params, and
    mutate constants
    """
    p = random.uniform(0, 1)

    if p < 1 / 4 and len(chromosome.constants) > 0:
        __shuffle_constants(chromosome)
    elif p < 2 / 4 and len(chromosome.int_params) > 0:
        __mutate_numeric_indicator_params(chromosome, "int")
    elif p < 3 / 4 and len(chromosome.float_params) > 0:
        __mutate_numeric_indicator_params(chromosome, "float")
    elif len(chromosome.constants) > 0:
        __mutate_constants(chromosome)


def __shuffle_constants(c: Chromosome):
    """Shuffle the constants in the chromosome"""
    random.shuffle(c.constants)


def __mutate_numeric_indicator_params(
    chromosome: Chromosome, dtype: Literal["int", "float"]
) -> None:
    """Mutate a single numeric param list for an indicator

    Each number in the list will be mutated with probability 'prob'. Mutation
    involves adding a number from Gaussian distribution to that number.
    """

    if dtype == "float":
        all_params = chromosome.float_params
        sigma = FLOAT_GAUSSIAN_SIGMA
    elif dtype == "int":
        all_params = chromosome.int_params
        sigma = INT_GAUSSIAN_SIGMA
    else:
        raise ValueError("dtype must be 'int' or 'float'")

    non_empty_params = [lst for lst in all_params if len(lst) > 0]
    if len(non_empty_params) == 0:
        return

    params = random.choice(non_empty_params)

    if len(params) == 0:
        return

    # Bitmask of which elements to mutate
    mutation_mask = np.random.random(size=len(params)) < ELEMENT_WISE_MUTATION_PROB
    changes = np.array([random.gauss(0, sigma) for _ in range(len(params))])

    # Round to nearest integer if necessary
    if dtype == "int":
        changes = np.rint(changes).astype(int)

    params["value"][mutation_mask] += changes[mutation_mask]

    # Clip values to be above 0 for floats, and above 1 for ints
    if dtype == "float":
        params["value"] = np.round(params["value"], DECIMAL_PLACES)
        params["value"] = np.clip(params["value"], 0, None)
    else:
        params["value"] = np.clip(params["value"], 1, None)


def __mutate_constants(chromosome: Chromosome):
    """Mutate all constants with probability `prob` using Gaussian distribution"""
    mutation_mask = (
        np.random.random(size=len(chromosome.constants)) < ELEMENT_WISE_MUTATION_PROB
    )
    changes = np.array(
        [
            random.gauss(0, CONSTANT_GAUSSIAN_SIGMA)
            for _ in range(len(chromosome.constants))
        ]
    )
    chromosome.constants[mutation_mask] += changes[mutation_mask]


def crossover(parents: list["Strategy"]) -> list["Strategy"]:
    """Given a list of Strategies to mate, return a list of offspring of same size

    One-point crossover is used to generate the offspring. Crosspoint is chosen
    for each pair of parents randomly, and the DNF expressions are swapped
    at that point to produce two children.

    NOTE: Only the buy chromosome is crossed over. The sell chromosome is
          generated as the symmetric of the buy chromosome.
    """
    offspring = []
    while len(offspring) < len(parents):
        p1, p2 = np.random.choice(parents, size=2)

        child_buy1, child_buy2 = __expression_crossover(
            p1.buy_chromosome, p2.buy_chromosome
        )

        ch = ChromosomeHandler()
        child_sell1 = ch.generate_symmetric_chromosome(child_buy1, is_buy=False)
        child_sell2 = ch.generate_symmetric_chromosome(child_buy2, is_buy=False)

        offspring.append(Strategy(p1.candles, child_buy1, child_sell1))
        offspring.append(Strategy(p1.candles, child_buy2, child_sell2))
    return offspring


def __expression_crossover(
    c1: Chromosome, c2: Chromosome
) -> Tuple[Chromosome, Chromosome]:
    """Perform 1-point crossover on two chromosomes and return the two children"""
    point1 = random.randint(
        1 if len(c1.expression_list) != 1 else 0, len(c1.expression_list) - 1
    )
    point2 = random.randint(
        1 if len(c1.expression_list) != 1 else 0, len(c1.expression_list) - 1
    )

    child1 = copy.deepcopy(c1)
    child2 = copy.deepcopy(c2)

    child1.expression_list[point1:], child2.expression_list[point2:] = (
        child2.expression_list[point2:],
        child1.expression_list[point1:],
    )

    __match_symbol_lengths(child1, c1, c2, point1, point2)
    __match_symbol_lengths(child2, c2, c1, point2, point1)

    return child1, child2


def __match_symbol_lengths(
    child: Chromosome,
    parent1: Chromosome,
    parent2: Chromosome,
    point1: int,
    point2: int,
):
    """
    Ensure the number of indicators, constants, and params match the number of symbols in the new DNF expression

    `parent1` must have the beginning portion of the child's DNF expression!
    """

    def count_symbols(parent, point, symbol):
        return str(parent.expression_list[:point]).count(symbol)

    # Number oF A, B, C for parent1 and parent2
    n_A1, n_B1, n_C1 = (
        count_symbols(parent1, point1, s) for s in [parent1.A, parent1.B, parent1.C]
    )
    n_A2, n_B2, n_C2 = (
        count_symbols(parent2, point2, s) for s in [parent2.A, parent2.B, parent2.C]
    )

    child.indicators = parent1.indicators[:n_A1] + parent2.indicators[n_A2:]
    child.candle_params = parent1.candle_params[:n_A1] + parent2.candle_params[n_A2:]
    child.int_params = parent1.int_params[:n_A1] + parent2.int_params[n_A2:]
    child.float_params = parent1.float_params[:n_A1] + parent2.float_params[n_A2:]
    child.candle_names = parent1.candle_names[:n_B1] + parent2.candle_names[n_B2:]
    child.constants = np.append(parent1.constants[:n_C1], parent2.constants[n_C2:], 0)


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
    while True:
        ch = ChromosomeHandler()
        c1 = ch.generate_chromosome()
        c2 = ch.generate_chromosome()
        c3, c4 = __expression_crossover(c1, c2)
