import asyncio
import threading
from types import TracebackType
from typing import Self

import coredis
import coredis.commands
import coredis.pool
import coredis.retry
import structlog
from langgraph_api.config import (
    REDIS_CLUSTER,
    REDIS_CONNECT_TIMEOUT,
    REDIS_MAX_CONNECTIONS,
    REDIS_URI,
    STATS_INTERVAL_SECS,
)

logger = structlog.stdlib.get_logger(__name__)

# main thread redis clients
_aredis: coredis.Redis[bytes] | coredis.RedisCluster[bytes]
_aredis_noretry: coredis.Redis[bytes] | coredis.RedisCluster[bytes]
_stats_task: asyncio.Task

# Thread-local storage for per-thread Redis clients
_thread_local = threading.local()

# class for connection pool and client
_cls_cp = (
    coredis.pool.ClusterConnectionPool if REDIS_CLUSTER else coredis.pool.ConnectionPool
)
_cls_cl = coredis.RedisCluster if REDIS_CLUSTER else coredis.Redis


async def start_redis() -> None:
    global _aredis, _aredis_noretry, _stats_task

    # create a redis connection pool
    _aredis_pool = _cls_cp.from_url(
        REDIS_URI,
        max_connections=REDIS_MAX_CONNECTIONS,
        connect_timeout=REDIS_CONNECT_TIMEOUT,
    )
    # create a redis client with retries
    _aredis = _cls_cl(connection_pool=_aredis_pool, decode_responses=False)
    # test the connection
    try:
        await asyncio.wait_for(_aredis.ping(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error("Redis ping timed out")
    except Exception as e:
        logger.error("Redis ping failed", error=str(e))
    # create a redis client with no retries
    _aredis_noretry = _cls_cl(
        connection_pool=_aredis_pool,
        decode_responses=False,
        retry_policy=coredis.retry.NoRetryPolicy(),
    )
    # start stats loop
    _stats_task = asyncio.create_task(stats_loop())


async def stop_redis() -> None:
    _stats_task.cancel()
    _aredis.connection_pool.disconnect()


async def stats_loop() -> None:
    while True:
        pool_stats = redis_stats()
        await logger.ainfo("Redis pool stats", **pool_stats)
        await asyncio.sleep(STATS_INTERVAL_SECS)


def redis_stats() -> dict[str, int]:
    """Get stats for the main Redis client"""
    global _aredis

    return {
        "idle_connections": len(_aredis.connection_pool._available_connections),
        "in_use_connections": len(_aredis.connection_pool._in_use_connections),
        "max_connections": _aredis.connection_pool.max_connections,
    }


def get_redis() -> coredis.Redis[bytes] | coredis.RedisCluster[bytes]:
    if threading.current_thread() is threading.main_thread():
        return _aredis
    else:
        # Create a new Redis client for this thread if it doesn't exist
        if not hasattr(_thread_local, "redis_client"):
            if not hasattr(_thread_local, "redis_pool"):
                _thread_local.redis_pool = _cls_cp.from_url(
                    REDIS_URI,
                    max_connections=REDIS_MAX_CONNECTIONS,
                    connect_timeout=REDIS_CONNECT_TIMEOUT,
                )
            _thread_local.redis_client = _cls_cl(
                connection_pool=_thread_local.redis_pool,
                decode_responses=False,
            )
            logger.info(
                "Created new thread-local Redis client",
                thread_name=threading.current_thread().name,
            )

        return _thread_local.redis_client


def get_redis_noretry() -> coredis.Redis[bytes] | coredis.RedisCluster[bytes]:
    if threading.current_thread() is threading.main_thread():
        return _aredis_noretry
    else:
        # Create a new Redis client for this thread if it doesn't exist
        if not hasattr(_thread_local, "redis_noretry_client"):
            if not hasattr(_thread_local, "redis_pool"):
                _thread_local.redis_pool = _cls_cp.from_url(
                    REDIS_URI,
                    max_connections=REDIS_MAX_CONNECTIONS,
                    connect_timeout=REDIS_CONNECT_TIMEOUT,
                )
            _thread_local.redis_noretry_client = _cls_cl(
                connection_pool=_thread_local.redis_pool,
                decode_responses=False,
                retry_policy=coredis.retry.NoRetryPolicy(),
            )
            logger.info(
                "Created new thread-local Redis client (no retry)",
                thread_name=threading.current_thread().name,
            )

        return _thread_local.redis_noretry_client


def get_pubsub() -> (
    coredis.commands.pubsub.BasePubSub[bytes, coredis.pool.ConnectionPool]
):
    if threading.current_thread() is threading.main_thread():
        pool = _aredis.connection_pool
    else:
        if not hasattr(_thread_local, "redis_pool"):
            _thread_local.redis_pool = _cls_cp.from_url(
                REDIS_URI,
                max_connections=REDIS_MAX_CONNECTIONS,
                connect_timeout=REDIS_CONNECT_TIMEOUT,
            )
        pool = _thread_local.redis_pool
    if REDIS_CLUSTER:
        return ClusterPubSub(pool)
    else:
        return PubSub(pool)


class PubSub(coredis.commands.pubsub.PubSub[bytes]):
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()


class ClusterPubSub(coredis.commands.pubsub.ClusterPubSub[bytes]):
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()


# keys

CHANNEL_RUN_STREAM = "run:{}:stream:{}"
CHANNEL_RUN_CONTROL = "run:{}:control"
STRING_RUN_CONTROL = "run:{}:control"
STRING_RUN_ATTEMPT = "run:{}:attempt"
STRING_RUN_RUNNING = "run:{}:running"
LIST_RUN_QUEUE = "run:queue"
LOCK_RUN_SWEEP = "run:sweep"
