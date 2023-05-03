import inspect
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import ta.others
import pandas as pd

import warnings
import random

from globals import *


__indicator_lst = []
for module in [ta.momentum, ta.trend, ta.volatility, ta.volume]:
    # Ignore 'adx' cause of "invalid value encountered in scalar divide" warning
    __indicator_lst.extend([func for func in inspect.getmembers(module, inspect.isfunction) if not func[0].startswith('_') and func[0] != 'adx'])


def generate_indicator_param(indicator) -> dict:
    """Generate random parameters for indicator
    
    Returns:
        Keyword arguments for indicator function and their values
        NOTE: candle parameters only have the name of the candle, not the actual candle
    """
    param_lst = {}
    # Automatically fill in parameters with default values
    for param in inspect.signature(indicator[1]).parameters.values():
        if param.default == inspect._empty:   # param for candles have no default value
            param_lst[param.name] = param.name
        elif isinstance(param.default, int) and not isinstance(param.default, bool):  # bools are ints????
            val = param.default + random.randint(-INT_OFFSET, INT_OFFSET)
            param_lst[param.name] = max(val, 1)
        elif isinstance(param.default, float):
            val = param.default + random.uniform(-FLOAT_OFFSET, FLOAT_OFFSET)
            param_lst[param.name] = round(max(val, 0.01), DECIMAL_PLACES)
    
    return param_lst

def random_indicator():
    """Return random indicator features with valid random parameters"""
    indicator = random.choice(__indicator_lst)
    return indicator


def __test_param_errors():
    """Run infinite loop to test for parameter errors"""
    while True:
        indicator = random.choice(__indicator_lst)
        param_lst = generate_indicator_param(indicator)
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                indicator[1](**param_lst)
                if caught_warnings:
                    for warning in caught_warnings:
                        print(indicator)
                        print(warning.message)
        except Exception as e:
            print(e)
            print(param_lst)

"""
1. Generate random DNF expressions
2. If value = indicator, randomly get indicator and generate random param, write to chromosome
    - indicator.append
    - 
3. if const, randomly generate and write to chromosome
"""

class ChromosomeHandler:
    """Generates a random DNF expression and converts it to a function."""

    # DISPLAY ONLY - used instead of function names for brevity
    __VAL1 = "A"
    __VAL2 = "B"

    # Change below to match signature in Strategy class
    __VAL1_NAME = "self.chromosome.indicators"
    __VAL2_NAME = "self.candles"

    # Discourage long expressions, adjust as needed
    DNF_PROBABILITY = 0.4
    CONJ_PROBABILITY = 0.2
    N_CONJ_MAX = 3

    @staticmethod
    def generate_chromosome() -> dict:
        chromosome = {}
        

    @staticmethod
    def generate_dnf_list() -> list:
        """Generate a random DNF expression.

        Returns:
            2-level nested list of DNF conjunctions and literals. Lists in the
            first level are implicitly joined by OR. Lists in the second level
            (i.e. literals) are implicitly joined by AND.
            e.g.
                [[A, B], [¬A]] -> (A ∧ B) ∨ ¬A
        """
        expression = [ChromosomeHandler.__generate_conjunctive()]
        if random.uniform(0, 1) < ChromosomeHandler.DNF_PROBABILITY:
            expression += ChromosomeHandler.generate_dnf_list()
        return expression

    def __generate_conjunctive() -> list:
        """Generate a random DNF conjunction"""
        expression = [ChromosomeHandler.__generate_literal()]
        if random.uniform(0, 1) < ChromosomeHandler.CONJ_PROBABILITY:
            expression += ChromosomeHandler.__generate_conjunctive()
        return expression

    def __generate_literal() -> str:
        """Generate a random DNF literal"""
        expression = (
            f"({ChromosomeHandler.__generate_value()} > "
            f"{ChromosomeHandler.__generate_constant()} * {ChromosomeHandler.__generate_value()})"
        )
        if random.uniform(0, 1) < 0.5:
            expression = f"(not {expression})"
        return expression

    def __generate_value() -> str | float:
        """Generate a random terminal DNF value"""
        if random.uniform(0, 1) < 1 / 3:
            return ChromosomeHandler.__VAL1_NAME
        if random.uniform(0, 1) < 2 / 3:
            return ChromosomeHandler.__VAL2_NAME
        return ChromosomeHandler.__generate_constant()

    def __generate_constant() -> float:
        """Generate random constant value"""
        return round(random.uniform(0, CONST_MAX), DECIMAL_PLACES)

    # =============== HELPER FUNCTIONS ============================
    
    @staticmethod
    def dnf_list_to_str(dnf: list) -> str:
        """Convert a DNF expression list to a string, for display purposes"""
        expr = ""
        for i in range(len(dnf)):
            conj = dnf[i]
            for j in range(len(conj)):
                lit = conj[j]
                expr += (
                    lit.replace("not ", "¬")
                    .replace(ChromosomeHandler.__VAL1_NAME, ChromosomeHandler.__VAL1)
                    .replace(ChromosomeHandler.__VAL2_NAME, ChromosomeHandler.__VAL2)
                )
                if j != len(conj) - 1:
                    expr += " ∧ "
            if i != len(dnf) - 1:
                expr += " ∨ "
        return expr
    
    
if __name__ == "__main__":
    while True:
        c = ChromosomeHandler.generate_dnf_list()
        print(ChromosomeHandler.dnf_list_to_str(c))