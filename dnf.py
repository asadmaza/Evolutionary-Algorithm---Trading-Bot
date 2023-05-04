import copy
from dataclasses import dataclass, field
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


@dataclass
class Chromosome:
    """Represents a chromosome storing a single DNF expression

    Attributes:
    ----------
    `indicators`: `list[str, Callable]`
        List of indicator (name:function) tuples; e.g. [("sma", <ta.trend.sma>), ...]
    `candle_names`: `list[str]`
        List of candle names for each candle_values() in expression
    `candle_params`: `list[list[str]]`
        List of arrays, i-th array is name of candle params for i-th indicator
        e.g. [["close"], ["close", "open"]] for indicator0(close, ...), indicator1(close, open, ...)
    `int_params`: `list[np.array]`
        List of np arrays, i-th array is int (argname:value) for i-th indicator
        e.g. [np.array([('window_size', 1), ('b', 2), ('c', 3)], dtype=int_dtypes), ...]
                for indicator0(window_size=1, b=2, c=3), indicator1(...)
    `float_params`: `list[np.array]`
        List of np arrays, i-th array is int (argname:value) for i-th indicator
    `constants`: `list[float]`
        List of constants used in the expression
    `expression_list`: `list`
        3D list, outer is a conjunction, middle is a disjunction, inner is a literal
        e.g. "(¬(A > C * B)) ∧ (A > C * C) ∨ (B > A * B)" is
            [[["not", "A", "C", "B"]
              ["A", "C", "C"]],
             [["B", "A", "B"]]
    """

    indicators: list[str, Callable] = field(default_factory=list)
    candle_names: list[str] = field(default_factory=list)
    candle_params: list[list[str]] = field(default_factory=list)
    int_params: list[list[str]] = field(default_factory=list)
    float_params: list[list[str]] = field(default_factory=list)
    constants: list[float] = field(default_factory=list)
    _expression_str: str = field(default_factory=str)
    _expression_list: list = field(default_factory=list)

    # DISPLAY ONLY - symbolic names used instead of function names for brevity
    A = "A"
    B = "B"
    C = "C"

    # Change below to match signature in Strategy class
    A_NAME = "self.indicators"
    B_NAME = "self.candles[self.chromosome['candle_names']"
    C_NAME = "self.chromosome['constants']"

    @property
    def expression_str(self) -> str:
        if self._expression_str == "":
            self._expression_str = self.to_str()
        return self._expression_str

    @property
    def expression_list(self) -> list:
        return self._expression_list

    @expression_list.setter
    def expression_list(self, value: list):
        if (
            len(value) != len(self.int_params)
            or len(value) != len(self.float_params)
            or len(value) != len(self.candle_params)
        ):
            raise ValueError(
                "Length of expression_list must match length of int_params, float_params, and candle_params"
            )
        self._expression_list = value
        self._expression_str = self.to_str()

    @expression_str.setter
    def expression_str(self, value: str):
        raise AttributeError(
            "Attribute expression_str cannot be set, create a new Chromosome object instead"
        )

    def to_str(self) -> str:
        """Convert internal list representation to string symbols"""

        def __literal_to_str(lit):
            if lit[0] == "not":
                return f"¬({lit[1]} > {lit[2]} * {lit[3]})"
            else:
                return f"({lit[0]} > {lit[1]} * {lit[2]})"

        def __disjunction_to_str(disj):
            return " ∨ ".join(__literal_to_str(lit) for lit in disj)

        dnf = " ∧ ".join(__disjunction_to_str(disj) for disj in self.expression_list)

        return dnf

    def to_function(self) -> Callable:
        expr = self.to_str_non_symbolic()

        func_signature = f"""def dnf_func(self, t):
            return {expr}
        """

        # Create temporary module to store function
        temp = types.ModuleType("temp_module")
        exec(func_signature, temp.__dict__)
        return temp.dnf_func

    def to_str_non_symbolic(self) -> str:
        """Expand symbolic representation to non-symbolic"""
        counters = {
            self.A: 0,
            self.B: 0,
            self.C: 0,
        }

        replacement_fn = lambda match: self.__replace_value_symbols(match, counters)
        expr = re.sub("¬", "not ", self.expression_str)
        expr = re.sub("∧", " and ", expr)
        expr = re.sub("∨", " or ", expr)
        expr = re.sub(
            f"{self.A}|{self.B}|{self.C}",
            replacement_fn,
            expr,
        )

        return expr

    def __repr__(self) -> str:
        pass

    def __replace_value_symbols(self, m: re.Match, counters: dict) -> str:
        """Replace value symbols with actual values"""
        value = m.group(0)
        index = counters[value]
        counters[value] += 1
        match value:
            case self.A:
                return self.A_NAME + f"[{index}][t]"
            case self.B:
                return self.B_NAME + f"[{index}][t]"
            case self.C:
                return self.C_NAME + f"[{index}]"
            case _:
                raise ValueError(f"Invalid value symbol {value}")


class ChromosomeHandler:
    """Handles chromosome and DNF expression generation"""

    # Discourage long expressions, adjust as needed
    DNF_PROBABILITY = 0.05
    CONJ_PROBABILITY = 0.2
    # Probability that indicator or candle value is chosen as value in DNF literal
    INDICATOR_PROBABILITY = 0.6
    CANDLE_VALUE_PROBABILITY = (
        INDICATOR_PROBABILITY + 0.3
    )  # 30%, ignore the INDICATOR_PROBABILITY part

    # Data types for numpy structured arrays
    INT_DTYPES = [("arg", "U20"), ("value", np.int8)]
    FLOAT_DTYPES = [("arg", "U20"), ("value", np.float16)]

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

    def generate_chromosome(self) -> Chromosome:
        """Generate a random chromosome object representing a DNF expression."""
        c = Chromosome()

        c.expression_list = self.__gen_dnf(c)
        c.constants = np.array(c.constants, dtype=np.float16)
        return c

    # ==================== DNF EXPRESSION ====================

    def __gen_dnf(self, chromosome: Chromosome) -> list:
        """Generate a random DNF expression."""
        expression = [self.__gen_conjunctive(chromosome)]

        if random.uniform(0, 1) < self.DNF_PROBABILITY:
            expression += self.__gen_dnf(chromosome)

        return expression

    def __gen_conjunctive(self, chromosome: dict) -> list:
        """Generate a random DNF conjunction"""
        expression = [self.__gen_literal(chromosome)]

        if random.uniform(0, 1) < self.CONJ_PROBABILITY:
            expression += self.__gen_conjunctive(chromosome)

        return expression

    def __gen_literal(self, chromosome: dict) -> list:
        """Generate a random DNF literal"""
        expression = [
            self.__gen_value(chromosome),
            self.__gen_constant(chromosome),
            self.__gen_value(chromosome),
        ]
        if random.uniform(0, 1) <= 0.5:
            expression = ["not"] + expression
        return expression

    def __gen_value(self, chromosome: dict) -> str:
        """Generate a random terminal DNF value"""
        rand_num = random.uniform(0, 1)

        # Add indicator to chromosome
        if rand_num < self.INDICATOR_PROBABILITY:
            chromosome.indicators += [random.choice(self.__indicator_lst)]
            c_params, i_params, f_params = self.__gen_indicator_params(
                chromosome.indicators[-1]
            )
            chromosome.candle_params += [c_params]
            chromosome.int_params += [i_params]
            chromosome.float_params += [f_params]
            return chromosome.A

        # Add candle name to chromosome
        elif rand_num < self.CANDLE_VALUE_PROBABILITY:
            chromosome.candle_names += random.choice(self.candle_names)
            return chromosome.B

        return self.__gen_constant(chromosome)

    def __gen_constant(self, chromosome: dict) -> float:
        """Add constant to chromosome"""
        chromosome.constants += [round(random.uniform(0, CONST_MAX), DECIMAL_PLACES)]
        return chromosome.C

    # ==================== INDICATORS ====================

    def __gen_indicator_params(
        self, indicator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        candle_params = np.array(candle_params, dtype="U6")
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


if __name__ == "__main__":
    """Example usage"""
    from candle import get_candles

    candles = get_candles()
    handler = ChromosomeHandler([ta.momentum, ta.volatility, ta.volume, ta.trend])

    c = handler.generate_chromosome()
    print()
    print(c.expression_str)
    print(c.expression_list)
    print(c.to_str_non_symbolic())
