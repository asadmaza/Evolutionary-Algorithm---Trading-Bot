import copy
import types
from typing import Callable, Tuple
import inspect
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import ta.others
import pandas as pd

import warnings
import random
import re  # regex my beloved <3

from globals import *


class ChromosomeHandler:
    """Handles chromosome and DNF expression generation

    Candle value DataFrame is required as it's used as input to indicators.

    Chromosome is defined as a dictionary, with keys:
        - "indicators": list in the form [("name", <function object>), ...]
        - "params": list of dictionary storing kargs for each indicator
            NOTE: arguments expecting oclhv data only have name of candle stored,
                not the actual pandas.Series!
            e.g. [{"high": "high", "window": 10, "offset" 1}, ...]
        - "constants": list of constants used in the expression, floats
        - "candle_names": list of candle names for each candle_values() in expression
        - "expression": list of lists, each sublist is a conjunction of indicators
        - "function": function object of the DNF expression
    """

    # DISPLAY ONLY - symbolic names used instead of function names for brevity
    __VAL1 = "A"
    __VAL2 = "B"
    __VAL3 = "C"

    # Change below to match signature in Strategy class
    __VAL1_NAME = "self.indicators"
    __VAL2_NAME = "self.chromosome['candle_names']"
    __VAL3_NAME = "self.chromosome['constants']"

    # Discourage long expressions, adjust as needed
    DNF_PROBABILITY = 0.1
    CONJ_PROBABILITY = 0.2
    N_CONJ_MAX = 3

    CHROMOSOME_FORMAT = {
        "indicators": [],
        "params": [],
        "candle_names": [],
        "constants": [],
    }

    def __init__(
        self,
        candles: pd.DataFrame,
        modules: list[types.ModuleType] = [
            ta.momentum,
            ta.volatility,
            ta.trend,
            ta.volume,
            ta.others,
        ],
    ):
        """Discover indicators in modules and bind self to specific candle values."""
        self.candles = candles  # Used as input to indicators

        self.__indicator_lst = []  # List of indicator functions to choose from
        for module in modules:
            # Ignore 'adx' cause of "invalid value encountered in scalar divide" warning
            self.__indicator_lst.extend(
                [
                    func
                    for func in inspect.getmembers(module, inspect.isfunction)
                    if not func[0].startswith("_") and func[0] != "adx"
                ]
            )

    def generate_chromosome(self) -> dict:
        """
        Return dictionary with key: 'indicators', 'params', 'constants', 'expression', 'function'
        """
        chromosome = copy.deepcopy(self.CHROMOSOME_FORMAT)
        chromosome["expression"] = self.__gen_dnf_list(chromosome)
        chromosome["function"] = self.dnf_list_to_function(chromosome["expression"])
        return chromosome

    # ==================== DNF EXPRESSION ====================

    def __gen_dnf_list(self, chromosome: dict) -> list:
        """Generate a random DNF expression.

        Returns:
            2-level nested list of DNF conjunctions and literals. Lists in the
            first level are implicitly joined by OR. Lists in the second level
            (i.e. literals) are implicitly joined by AND.
            e.g.
                [[A, B], [¬A]] -> (A ∧ B) ∨ ¬A
        """
        expression = [self.__gen_conjunctive(chromosome)]
        if random.uniform(0, 1) < self.DNF_PROBABILITY:
            expression += self.__gen_dnf_list(chromosome)
        return expression

    def __gen_conjunctive(self, chromosome: dict) -> list:
        """Generate a random DNF conjunction"""
        expression = [self.__gen_literal(chromosome)]
        if random.uniform(0, 1) < self.CONJ_PROBABILITY:
            expression += self.__gen_conjunctive(chromosome)
        return expression

    def __gen_literal(self, chromosome: dict) -> str:
        """Generate a random DNF literal"""
        expression = (
            f"({self.__gen_value(chromosome)} > "
            f"{self.__gen_constant(chromosome)} * {self.__gen_value(chromosome)})"
        )
        if random.uniform(0, 1) < 0.5:
            expression = f"(not {expression})"
        return expression

    def __gen_value(self, chromosome: dict) -> str | float:
        """Generate a random terminal DNF value"""
        rand_num = random.uniform(0, 1)

        # Add indicator to chromosome
        if rand_num < 1 / 3:
            chromosome["indicators"].append(random.choice(self.__indicator_lst))
            chromosome["params"].append(
                self.__gen_indicator_param(chromosome["indicators"][-1])
            )
            return f'{self.__VAL1_NAME}[{len(chromosome["indicators"]) - 1}]'

        # Add candle name to chromosome
        elif rand_num < 2 / 3:
            chromosome["candle_names"].append(
                random.choice(["open", "high", "low", "close", "volume"])
            )
            return f'{self.__VAL2_NAME}[{len(chromosome["candle_names"]) - 1}]'

        return self.__gen_constant(chromosome)

    def __gen_constant(self, chromosome: dict) -> float:
        """Add constant to chromosome"""
        chromosome["constants"].append(
            round(random.uniform(0, CONST_MAX), DECIMAL_PLACES)
        )
        return f'{self.__VAL3_NAME}[{len(chromosome["constants"]) - 1}]'

    # ==================== INDICATORS ====================

    def __gen_indicator_param(self, indicator) -> dict:
        """Generate random parameters for indicator

        Returns:
            Keyword arguments for indicator function and their values
        """
        param_lst = {}
        # Automatically fill in parameters with default values
        for param in inspect.signature(indicator[1]).parameters.values():
            # param for candles have no default value, save candle name
            if param.default == inspect._empty:
                param_lst[param.name] = self.candles[param.name]
            # bools are ints???? generate random int in range
            elif isinstance(param.default, int) and not isinstance(param.default, bool):
                val = param.default + random.randint(-INT_OFFSET, INT_OFFSET)
                param_lst[param.name] = max(val, 1)
            # Generate random float in range
            elif isinstance(param.default, float):
                val = param.default + random.uniform(-FLOAT_OFFSET, FLOAT_OFFSET)
                param_lst[param.name] = round(max(val, 0.01), DECIMAL_PLACES)

        return param_lst

    def __test_param_errors(self):
        """Run infinite loop to test for parameter errors"""
        while True:
            indicator = random.choice(self.__indicator_lst)
            param_lst = self.gen_indicator_param(indicator)
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

    # =============== CONVERSION FUNCTIONS ============================

    @staticmethod
    def dnf_list_to_str(dnf: list, symbolic: bool = True) -> str:
        """Convert a DNF expression list to a string, replace with symbols if symbolic is True"""
        expr = ""
        if not symbolic:
            # Convert list representation to a string
            for i in range(len(dnf)):
                conj = dnf[i]
                for j in range(len(conj)):
                    lit = conj[j]
                    expr += lit
                    if j != len(conj) - 1:
                        expr += " and "
                if i != len(dnf) - 1:
                    expr += " or "
            return expr

        # Replace names with symbols, forgive me for this mess
        for i in range(len(dnf)):
            conj = dnf[i]
            for j in range(len(conj)):
                lit = conj[j]
                lit = re.sub("not ", "¬", lit)
                lit = re.sub(
                    re.escape(ChromosomeHandler.__VAL1_NAME) + r"\[\d+\]",
                    ChromosomeHandler.__VAL1,
                    lit,
                )
                lit = re.sub(
                    re.escape(ChromosomeHandler.__VAL2_NAME) + r"\[\d+\]",
                    ChromosomeHandler.__VAL2,
                    lit,
                )
                lit = re.sub(
                    re.escape(ChromosomeHandler.__VAL3_NAME) + r"\[\d+\]",
                    ChromosomeHandler.__VAL3,
                    lit,
                )
                expr += lit
                # Add AND between literals unless it's the last one
                if j != len(conj) - 1:
                    expr += " ∧ "
            if i != len(dnf) - 1:
                expr += " ∨ "
        return expr

    @staticmethod
    def dnf_list_to_function(dnf: list) -> Callable:
        """Convert a DNF expression list to a function"""
        expr = ChromosomeHandler.dnf_list_to_str(dnf, symbolic=False)

        func_signature = f"""def dnf_func(self, time):
            return {expr}
        """

        # Create temporary module to store function
        temp = types.ModuleType("temp_module")
        exec(func_signature, temp.__dict__)
        return temp.dnf_func

    # ==================== DNF MANIPULATION ====================

    @staticmethod
    def to_symmetric_literals(expr: str) -> str:
        """For dnf string with literals in form 'X > C * Y', replace with 'Y > C * X'"""
        return re.sub(r"\(([^()]+)\)", ChromosomeHandler.__swap_literals, expr)

    def __swap_literals(m: re.Match):
        """Swap literals in form 'X > C * Y' to 'Y > C * X'"""
        lit = m.group(1)
        val_1, rest = lit.split(" > ", 1)
        constant, val_2 = rest.split(" * ", 1)
        return f"({val_2} > {constant} * {val_1})"


if __name__ == "__main__":
    """Example usage"""
    from candle import get_candles

    candles = get_candles()
    handler = ChromosomeHandler(
        candles, [ta.momentum, ta.volatility, ta.volume, ta.trend]
    )

    c = handler.generate_chromosome()

    expression = c["expression"]
    print(f"EXPRESSION SYMBOLIC:\n{ChromosomeHandler.dnf_list_to_str(expression)}")
    print(
        f"EXPRESSION SYMBOLIC SYMMETRIC:\n{ChromosomeHandler.to_symmetric_literals(ChromosomeHandler.dnf_list_to_str(expression))}"
    )
    print(f"EXPRESSION LIST:\n{expression}")
    print(
        f"EXPRESSION:\n{ChromosomeHandler.dnf_list_to_str(expression, symbolic=False)}"
    )
    print(f"DICTOINARY:\n{c}")
