"""
Get ohlcv candle data from the Kraken exchange.

TODO:
- extend this to fetch batches of past data and save in file.
- split data into batches of training and test data.
"""

import ccxt
import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
SINCE = 1620345600000


def get_candles(
        since: int = SINCE,
        fetch: bool = False,
        time: str = "1d",
        market: str = "BTC/AUD") -> pd.DataFrame:
  """
  Return a pandas DataFrame of ohlcv candle data.

  Parameters
  ----------
    fetch : bool
      Fetch new data from the Kraken exchange as opposed to from file.

    time : str
      Time period of each candle eg. '1d' or '15m'.

    market : str
      The market to fetch eg. 'BTC/AUD'.
  """

  read = True

  if not fetch:
    try:
      return pd.read_csv("candles.csv")
    except BaseException:
      read = False
  if fetch or not read:
    kraken = ccxt.kraken()
    candles = pd.DataFrame(
        kraken.fetch_ohlcv(
            market,
            time,
            since),
        columns=COLUMNS)
    candles.to_csv("candles.csv", index=False)
    return candles


def get_candles_split(training=0.8):
  candles = get_candles()
  n = len(candles)
  train_end = int(n * training)  # index of the last row in the training set
  train_df = candles.iloc[:train_end].reset_index(
      drop=True
  )  # select the first 80% of the rows
  test_df = candles.iloc[train_end:].reset_index(
      drop=True
  )  # select the last 20% of the rows

  return train_df, test_df


if __name__ == "__main__":
  candles = get_candles()
  candles2 = get_candles(fetch=True)

  print(all(candles == candles2))

  # print min and max of close price column
  print(min(c := candles.iloc[:, 4]), max(c))

  tr, te = get_candles_split()
  print(te.head())
