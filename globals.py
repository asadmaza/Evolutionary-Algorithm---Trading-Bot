# Sigma determines spread of Gaussian, used in mutation
import time
from functools import wraps
from typing import Final


# Sigma determines spread of Gaussian, used in mutation
INT_GAUSSIAN_SIGMA: Final[int] = 3
CONSTANT_GAUSSIAN_SIGMA: Final[float] = 0.5
FLOAT_GAUSSIAN_SIGMA: Final[float] = 0.5

# Offset from default int or float value, range is +/- this value
INT_OFFSET: Final[int] = 10
FLOAT_OFFSET: Final[int] = 3
DECIMAL_PLACES: Final[int] = 3
CONST_MAX: Final[int] = 5
# WHen a mutation DOES occur, the probability that each element is altered
ELEMENT_WISE_MUTATION_PROB: Final[float] = 0.1


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
