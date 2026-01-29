import time
import logging

from functools import wraps, lru_cache, cache

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler('app.log')],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_calls(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()

        try:
            logger.info(
                f'{func.__name__} called | '
                f'args={args} kwargs={kwargs}'
            )

            result = func(*args, **kwargs)

            duration = time.time() - start
            logger.info(
                f'{func.__name__} finished in {duration:.3f}s | '
                f'result={result}'
            )

            return result

        except Exception as e:
            duration = time.time() - start
            logger.exception(
                f'{func.__name__} failed after {duration:.3f}s: {e}'
            )
            raise

    return wrapper


@log_calls
def add(a: int, b: int) -> int:
    return a + b


@log_calls
def subtract(a: int, b: int) -> int:
    return a - b


if __name__ == '__main__':
    print(add(1, 2))
    print(subtract(1, 2))