'''
Get ohlcv candle data from the Kraken exchange.

TODO:
- extend this to fetch batches of past data and save in file.
- split data into batches of training and test data.
'''

import ccxt
import pandas as pd

COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
SINCE = 1620345600000

def get_candles(since: int = SINCE, fetch: bool = False, time: str = '1d', market: str = 'BTC/AUD') -> pd.DataFrame:
  '''
  Return a pandas DataFrame of ohlcv candle data.

  Parameters
  ----------
    fetch : bool
      Fetch new data from the Kraken exchange as opposed to from file.

    time : str
      Time period of each candle eg. '1d' or '15m'.

    market : str
      The market to fetch eg. 'BTC/AUD'.
  '''

  try:
    return pd.read_csv('candles.csv')
  except:  
    kraken = ccxt.kraken()
    candles = pd.DataFrame(kraken.fetch_ohlcv(market, time, since), columns=COLUMNS)
    candles.to_csv('candles.csv', index=False)
    return candles

if __name__ == '__main__':
  candles = get_candles()

  print(min(c:=candles.iloc[:, 4]), max(c)) # print min and max of close price column