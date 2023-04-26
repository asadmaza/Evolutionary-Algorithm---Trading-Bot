'''
Get ohlcv candle data from the Kraken exchange.

TODO:
- extend this to fetch batches of past data and save in file.
- split data into batches of training and test data.
'''

import ccxt
import pandas as pd

COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def get_candles(time: str = '1d', market: str = 'BTC/AUD') -> pd.DataFrame:
  '''
  Return a pandas DataFrame of ohlcv candle data.

  Parameters
  ----------
    time : str
      Time period of each candle eg. '1d' or '15m'.

    market : str
      The market to fetch eg. 'BTC/AUD'.
  '''
  
  kraken = ccxt.kraken()
  return pd.DataFrame(kraken.fetch_ohlcv(market, time), columns=COLUMNS)

if __name__ == '__main__':
  candles = get_candles()

  print(candles)
  print(min(c:=candles.iloc[:, 4]), max(c)) # print min and max of close price column
