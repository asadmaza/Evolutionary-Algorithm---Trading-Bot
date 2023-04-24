import ccxt
import pandas as pd

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
  return pd.DataFrame(kraken.fetch_ohlcv(market, time))

if __name__ == '__main__':
  candles = get_candles()

  # print min and max of close price column
  print(min(c:=candles.iloc[:, 4]), max(c))
