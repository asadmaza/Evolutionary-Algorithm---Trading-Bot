import ccxt
import ta
import pandas as pd
import math
from matplotlib import pyplot as plt

MARKET = 'BIT/USD'
TIMEFRAME = '1d'

SMA_WINDOW = 20
EMA_WINDOW = 10

kraken = ccxt.kraken()
res = kraken.fetch_ohlcv(MARKET, TIMEFRAME)

candles = pd.DataFrame(res)
close = candles.iloc[:, 4] # 5th column is close price

sma = ta.trend.sma_indicator(close, SMA_WINDOW)
ema = ta.trend.ema_indicator(close, EMA_WINDOW)

plt.plot(close, color='grey', label='Close price')
plt.plot(sma, color='blue', label='SMA')
plt.plot(ema, color ='orange', label='EMA')

usd = 1
bitcoin = 0

bought, sold = False, False

# locate the buy and sell trigger points
for i in range(1, len(close)):

    # indicators are nan until first window has elapsed
    if math.isnan(ema[i]) or math.isnan(sma[i]): continue

    # buy trigger
    if ema[i] > sma[i] and ema[i-1] <= sma[i-1]:
        bitcoin += usd / close[i]
        print(f'Bought {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at {close[i]:4.2f}')
        usd = 0
        
        plt.plot((i), (close[i]), 'o', color='red', label='Buy' if not bought else '')
        bought = True

    # sell trigger
    elif sma[i] > ema[i] and sma[i-1] <= ema[i-1]:
        usd += bitcoin * close[i]
        print(f'Sold   {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at {close[i]:4.2f}')
        bitcoin = 0

        plt.plot((i), (close[i]), 'o', color='green', label='Sell' if not sold else '')
        sold = True

# sell at last close price
if bitcoin:
    usd += bitcoin * close.iloc[-1]
    print(f'Sold   {bitcoin:4.2f} bitcoin for {usd:4.2f} USD at {close.iloc[-1]:4.2f}')

plt.legend()
plt.show(block=True)
