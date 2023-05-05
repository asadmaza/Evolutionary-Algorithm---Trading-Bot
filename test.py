import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ta.trend import *
from ta.momentum import *
from ta.volatility import *
from ta.volume import *
from ta.others import *

from dnf import ChromosomeHandler
from candle import get_candles

candles = get_candles()

i = rsi(candles["close"], 9)

plt.plot(candles["close"], label="Close price")
plt.plot(
    i,
)
plt.show()
