import asyncio
import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from psycopg.errors import ConnectionTimeout, InternalError, OperationalError
from psycopg_pool.errors import PoolTimeout, TooManyRequests

P = ParamSpec("P")
T = TypeVar("T")


class RetryableException(Exception):
    pass


RETRIABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OperationalError,
    InternalError,
    RetryableException,
)

OVERLOADED_EXCEPTIONS: tuple[type[BaseException], ...] = (
    PoolTimeout,
    ConnectionTimeout,
    TooManyRequests,
)


def retry_db(func: Callable[P, T]) -> Callable[P, T]:
    attempts = 3

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        for i in range(attempts):
            if i == attempts - 1:
                return await func(*args, **kwargs)
            try:
                return await func(*args, **kwargs)
            except RETRIABLE_EXCEPTIONS:
                await asyncio.sleep(0.01)

    return wrapper
