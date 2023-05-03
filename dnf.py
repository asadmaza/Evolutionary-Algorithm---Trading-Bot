from typing import Tuple
import inspect
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import ta.others
import pandas as pd

import warnings
import random
import re     # regex my beloved <3

from globals import *



class ChromosomeHandler:
    """Handles chromosome and DNF expression generation
    
    Chromosome is defined as a dictionary, with keys:
    - "indicators": list in the form [("name", <function object>), ...]
    - "params": list of dictionary storing kargs for each indicator
        NOTE: arguments expecting oclhv data only have name of candle stored,
            not the actual pandas.Series!
        e.g. [{"high": "high", "window": 10, "offset" 1}, ...]
    - "constants": list of constants used in the expression, floats
    - "candle_names": list of candle names for each candle_values() in expression
    """

    # DISPLAY ONLY - used instead of function names for brevity
    __VAL1 = "A"
    __VAL2 = "B"
    __VAL3 = "C"

    # Change below to match signature in Strategy class
    __VAL1_NAME = "self.chromosome['indicators']"
    __VAL2_NAME = "self.chromosome['candle_names']"
    __VAL3_NAME = "self.chromosome['constants']"

    # Discourage long expressions, adjust as needed
    DNF_PROBABILITY = 0.1
    CONJ_PROBABILITY = 0.2
    N_CONJ_MAX = 3

    __indicator_lst = []
    for module in [ta.momentum, ta.trend, ta.volatility, ta.volume]:
        # Ignore 'adx' cause of "invalid value encountered in scalar divide" warning
        __indicator_lst.extend(
            [
                func
                for func in inspect.getmembers(module, inspect.isfunction)
                if not func[0].startswith("_") and func[0] != "adx"
            ]
        )

    @staticmethod
    def generate_chromosome() -> Tuple[dict, list]:
        chromosome = {"indicators": [], "params": [], "candle_names": [], "constants": []}
        expression = ChromosomeHandler.__gen_dnf_list(chromosome)
        return chromosome, expression

    # ==================== DNF EXPRESSION ====================

    @staticmethod
    def __gen_dnf_list(chromosome: dict) -> list:
        """Generate a random DNF expression.

        Returns:
            2-level nested list of DNF conjunctions and literals. Lists in the
            first level are implicitly joined by OR. Lists in the second level
            (i.e. literals) are implicitly joined by AND.
            e.g.
                [[A, B], [¬A]] -> (A ∧ B) ∨ ¬A
        """
        expression = [ChromosomeHandler.__gen_conjunctive(chromosome)]
        if random.uniform(0, 1) < ChromosomeHandler.DNF_PROBABILITY:
            expression += ChromosomeHandler.__gen_dnf_list(chromosome)
        return expression

    def __gen_conjunctive(chromosome: dict) -> list:
        """Generate a random DNF conjunction"""
        expression = [ChromosomeHandler.__gen_literal(chromosome)]
        if random.uniform(0, 1) < ChromosomeHandler.CONJ_PROBABILITY:
            expression += ChromosomeHandler.__gen_conjunctive(chromosome)
        return expression

    def __gen_literal(chromosome: dict) -> str:
        """Generate a random DNF literal"""
        expression = (
            f"({ChromosomeHandler.__gen_value(chromosome)} > "
            f"{ChromosomeHandler.__gen_constant(chromosome)} * {ChromosomeHandler.__gen_value(chromosome)})"
        )
        if random.uniform(0, 1) < 0.5:
            expression = f"(not {expression})"
        return expression

    def __gen_value(chromosome: dict) -> str | float:
        """Generate a random terminal DNF value"""
        rand_num = random.uniform(0, 1)

        # Add indicator to chromosome
        if rand_num < 1 / 3:
            chromosome["indicators"].append(
                random.choice(ChromosomeHandler.__indicator_lst)
            )
            chromosome["params"].append(
                ChromosomeHandler.__gen_indicator_param(chromosome["indicators"][-1])
            )
            return (
                f'{ChromosomeHandler.__VAL1_NAME}[{len(chromosome["indicators"]) - 1}]'
            )

        # Add candle name to chromosome
        elif rand_num < 2 / 3:
            chromosome["candle_names"].append(
                random.choice(["open", "high", "low", "close", "volume"])
            )
            return f'{ChromosomeHandler.__VAL2_NAME}[{len(chromosome["candle_names"]) - 1}]'

        return ChromosomeHandler.__gen_constant(chromosome)

    def __gen_constant(chromosome: dict) -> float:
        """Add constant to chromosome"""
        chromosome["constants"].append(round(random.uniform(0, CONST_MAX), DECIMAL_PLACES))
        return f'{ChromosomeHandler.__VAL3_NAME}[{len(chromosome["constants"]) - 1}]'
        

    # ==================== INDICATORS ====================

    def __gen_indicator_param(indicator) -> dict:
        """Generate random parameters for indicator

        Returns:
            Keyword arguments for indicator function and their values
            NOTE: candle parameters only have the name of the candle, not the actual candle
        """
        param_lst = {}
        # Automatically fill in parameters with default values
        for param in inspect.signature(indicator[1]).parameters.values():
            # param for candles have no default value, save candle name
            if param.default == inspect._empty:  
                param_lst[param.name] = param.name
            # bools are ints???? generate random int in range
            elif isinstance(param.default, int) and not isinstance(param.default, bool):
                val = param.default + random.randint(-INT_OFFSET, INT_OFFSET)
                param_lst[param.name] = max(val, 1)
            # Generate random float in range
            elif isinstance(param.default, float):
                val = param.default + random.uniform(-FLOAT_OFFSET, FLOAT_OFFSET)
                param_lst[param.name] = round(max(val, 0.01), DECIMAL_PLACES)

        return param_lst

    def __test_param_errors():
        """Run infinite loop to test for parameter errors"""
        while True:
            indicator = random.choice(ChromosomeHandler.__indicator_lst)
            param_lst = ChromosomeHandler.gen_indicator_param(indicator)
            # Catch warnings and print them instead of ignoring
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

    # =============== HELPER FUNCTIONS ============================

    @staticmethod
    def dnf_list_to_str(dnf: list) -> str:
        """Convert a DNF expression list to a string, for display purposes"""
        expr = ""
        for i in range(len(dnf)):
            conj = dnf[i]
            for j in range(len(conj)):
                lit = conj[j]
                # Replace names with symbols, forgive me for this mess
                lit = re.sub("not ", "¬", lit)
                lit = re.sub(re.escape(ChromosomeHandler.__VAL1_NAME) + r"\[\d+\]", ChromosomeHandler.__VAL1, lit)
                lit = re.sub(re.escape(ChromosomeHandler.__VAL2_NAME) + r"\[\d+\]", ChromosomeHandler.__VAL2, lit)
                lit = re.sub(re.escape(ChromosomeHandler.__VAL3_NAME) + r"\[\d+\]", ChromosomeHandler.__VAL3, lit)
                expr += lit

                # Add AND between literals unless it's the last one
                if j != len(conj) - 1:
                    expr += " ∧ "
            if i != len(dnf) - 1:
                expr += " ∨ "
        return expr


if __name__ == "__main__":
    c, expression = ChromosomeHandler.generate_chromosome()
    print(expression)
    print(ChromosomeHandler.dnf_list_to_str(expression))
    print(c)