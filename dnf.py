import inspect
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume
import ta.others
import pandas as pd

import warnings
import random

from candle import get_candles_split
from globals import *


__indicator_lst = []
for module in [ta.momentum, ta.trend, ta.volatility, ta.volume]:
    # Ignore 'adx' cause of "invalid value encountered in scalar divide" warning
    __indicator_lst.extend([func for func in inspect.getmembers(module, inspect.isfunction) if not func[0].startswith('_') and func[0] != 'adx'])


def gen_param(indicator, candles: pd.DataFrame) -> dict:
    param_lst = {}
    # Automatically fill in parameters with default values
    for param in inspect.signature(indicator[1]).parameters.values():
        if param.default == inspect._empty:   # param for candles have no default value
            param_lst[param.name] = candles[param.name]
        elif isinstance(param.default, int) and not isinstance(param.default, bool):  # bools are ints????
            val = param.default + random.randint(-INT_OFFSET, INT_OFFSET)
            param_lst[param.name] = max(val, 1)
        elif isinstance(param.default, float):
            val = param.default + random.uniform(-FLOAT_OFFSET, FLOAT_OFFSET)
            param_lst[param.name] = max(val, 0.01)
    
    return param_lst

def random_indicator():
    """Return random indicator features with valid random parameters"""
    indicator = random.choice(__indicator_lst)
    return indicator


def __test_param_errors():
    """Run infinite loop to test for parameter errors"""
    while True:
        indicator = random.choice(__indicator_lst)
        param_lst = gen_param(indicator)
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


        



