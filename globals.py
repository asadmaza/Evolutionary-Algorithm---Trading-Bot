# Sigma determines spread of Gaussian, used in mutation
WINDOW_GAUSSIAN_SIGMA = 3
CONSTANT_GAUSSIAN_SIGMA = 0.5
WINDOW_DEV_GAUSSIAN_SIGMA = 0.5

# Upperbound for mutation for window, window dev, and constant
WIN_MAX = 50
WIN_DEV_MAX = 10
CONST_MAX = 5
DECIMAL_PLACE = 5


from functools import wraps
import time


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
