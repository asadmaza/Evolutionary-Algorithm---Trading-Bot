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
import numpy as np

import warnings
import random
import re  # regex my beloved <3

from globals import *


class ChromosomeHandler:
    """Handles chromosome and DNF expression generation
    """

    # DISPLAY ONLY - symbolic names used instead of function names for brevity
    __VAL1 = "A"
    __VAL2 = "B"
    __VAL3 = "C"

    # Change below to match signature in Strategy class
    __VAL1_NAME = "self.indicators"
    __VAL2_NAME = "self.candles[self.chromosome['candle_names']"
    __VAL3_NAME = "self.chromosome['constants']"

    # Discourage long expressions, adjust as needed
    DNF_PROBABILITY = 0.05
    CONJ_PROBABILITY = 0.2
    N_CONJ_MAX = 3

    CHROMOSOME_FORMAT = {
        # List of indicator (name:function) tuples; e.g. [("sma", <ta.trend.sma>), ...]
        "indicators": [],      
        # List of candle names for each candle_values() in expression
        "candle_names": [],    
        # List of arrays, i-th array is name of candle params for i-th indicator
        # e.g. [["close"], ["close", "open"]] for indicator0(close, ...), indicator1(close, open, ...)
        "candle_params": [],   
        # List of np arrays, i-th array is int (argname:value) for i-th indicator
        # e.g. [np.array([('window_size', 1), ('b', 2), ('c', 3)], dtype=int_dtypes), ...]
        #      for indicator0(window_size=1, b=2, c=3), indicator1(...)
        "int_params": [],      
        # Ditto, but for float args
        "float_params": [],
        # List of constants used in the expression
        "constants": [],
        # List of python boolean expressions
        "expressions": [],
        # List of boolean function objects for each trigger
        "functions": [],
    }
    INT_DTYPES = [('arg', 'U20'), ('value', np.int8)]
    FLOAT_DTYPES = [('arg', 'U20'), ('value', np.float16)]

    def __init__(
        self,
        modules: list[types.ModuleType] = [
            ta.momentum,
            ta.volatility,
            ta.trend,
            ta.volume,
            ta.others,
        ],
        candle_names: list[str] = ["close"],
    ):
        """Discover indicators in modules and bind self to specific candle values."""
        self.candle_names = candle_names
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
    
    def generate_chromosome(self, n_expressions: int = 2) -> dict:
        """
        Return dictionary with key: 'indicators', 'params', 'constants', 'expression', 'function'

        Parameters:
        ----------
            n_expression: int
                number of DNF expressions to generate
        """
        chromosome = copy.deepcopy(self.CHROMOSOME_FORMAT)
        for _ in range(n_expressions):
            chromosome["expressions"].append(self.__gen_dnf_list(chromosome))
            chromosome["functions"].append(
                self.dnf_list_to_function(chromosome["expressions"][-1])
            )
        chromosome["constants"] = np.array(chromosome["constants"], dtype=np.float16)
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
        if rand_num < 3 / 6:
            chromosome["indicators"].append(random.choice(self.__indicator_lst))
            c_params, i_params, f_params = self.__gen_indicator_params(chromosome["indicators"][-1])
            chromosome["candle_params"] += [c_params]
            chromosome["int_params"] += [i_params]
            chromosome["float_params"] += [f_params]
            return f'{self.__VAL1_NAME}[{len(chromosome["indicators"]) - 1}][t]'

        # Add candle name to chromosome
        elif rand_num < 2 / 6:
            chromosome["candle_names"].append(
                random.choice(self.candle_names)
            )
            return f'{self.__VAL2_NAME}[{len(chromosome["candle_names"]) - 1}]][t]'

        return self.__gen_constant(chromosome)

    def __gen_constant(self, chromosome: dict) -> float:
        """Add constant to chromosome"""
        chromosome['constants'] += [round(random.uniform(0, CONST_MAX), DECIMAL_PLACES)]
        return f'{self.__VAL3_NAME}[{len(chromosome["constants"]) - 1}]'

    # ==================== INDICATORS ====================

    def __gen_indicator_params(self, indicator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random parameters for indicator

        Returns:
            Three parameter numpy arrays for candle names, int args, and float args
        """
        candle_params = []
        int_params = []
        float_params = []

        # Automatically fill in parameters with default values
        for param in inspect.signature(indicator[1]).parameters.values():
            # param for candles have no default value, save candle name
            if param.default == inspect._empty:
                candle_params += [param.name]
            elif isinstance(param.default, int) and not isinstance(param.default, bool):
                val = param.default + random.randint(-INT_OFFSET, INT_OFFSET)
                int_params += [(param.name, max(val, 1))]
            elif isinstance(param.default, float):
                val = param.default + random.uniform(-FLOAT_OFFSET, FLOAT_OFFSET)
                float_params += [(param.name, round(max(val, 0.01), DECIMAL_PLACES))]

        candle_params = np.array(candle_params, dtype='U6')
        int_params = np.array(int_params, dtype=self.INT_DTYPES)
        float_params = np.array(float_params, dtype=self.FLOAT_DTYPES)

        return candle_params, int_params, float_params

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
                    re.escape(ChromosomeHandler.__VAL1_NAME) + r"\[\d+\]\[t\]",
                    ChromosomeHandler.__VAL1,
                    lit,
                )
                lit = re.sub(
                    re.escape(ChromosomeHandler.__VAL2_NAME) + r"\[\d+\]]\[t\]",
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

        func_signature = f"""def dnf_func(self, t):
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
        [ta.momentum, ta.volatility, ta.volume, ta.trend]
    )

    c = handler.generate_chromosome()

    for e in c["expressions"]:
        print(f"EXPRESSION SYMBOLIC:\n\t\t{ChromosomeHandler.dnf_list_to_str(e)}\n")
        print(
            f"EXPRESSION SYMBOLIC SYMMETRIC:\n\t\t{ChromosomeHandler.to_symmetric_literals(ChromosomeHandler.dnf_list_to_str(e))}\n"
        )
        print(
            f"EXPRESSION:\n\t\t{ChromosomeHandler.dnf_list_to_str(e, symbolic=False)}\n"
        )
        print(f"EXPRESSION LIST:\n\t\t{e}\n")
        print("-" * 50)
    print(f"DICTOINARY:\n\t\t{c}\n")
