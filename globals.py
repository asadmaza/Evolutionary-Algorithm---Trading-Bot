# Sigma determines spread of Gaussian, used in mutation
import time
from functools import wraps


# Sigma determines spread of Gaussian, used in mutation
INT_GAUSSIAN_SIGMA = 3
CONSTANT_GAUSSIAN_SIGMA = 0.5
FLOAT_GAUSSIAN_SIGMA = 0.5

# Offset from default int or float value, range is +/- this value
INT_OFFSET = 30
FLOAT_OFFSET = 5
FLOAT_DECIMAL = 5
CONST_MAX = 5


def timer_decorator(func):
  """Decorator to time function execution"""

  @wraps(func)
  def wrapper(*args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Execution time of {func.__name__}: {elapsed_time} seconds")
    return result

  return wrapper
