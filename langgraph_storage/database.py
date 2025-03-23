import asyncio
import os
import re
import threading
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypeAlias

import langgraph_api.config as config
import structlog
from langgraph_api.serde import Fragment, json_dumpb
from psycopg import AsyncConnection
from psycopg.conninfo import conninfo_to_dict
from psycopg.errors import UndefinedTable
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import set_json_dumps, set_json_loads
from psycopg_pool import AsyncConnectionPool

from langgraph_storage.redis import get_redis, redis_stats, start_redis, stop_redis

Row: TypeAlias = dict[str, Any]


logger = structlog.stdlib.get_logger(__name__)
_pg_pool: AsyncConnectionPool[AsyncConnection[DictRow]]
_stats_task: asyncio.Task

# Thread-local storage for per-thread connection pools
_thread_local = threading.local()


async def healthcheck() -> None:
    # check postgres
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SELECT 1")
    # check redis
    await get_redis().ping()


@asynccontextmanager
async def connect(*, __test__: bool = False) -> AsyncIterator[AsyncConnection[DictRow]]:
    if __test__:
        async with await create_conn(__test__) as conn:
            yield conn
    elif threading.current_thread() is not threading.main_thread():
        # Use thread-local connection pool
        if not hasattr(_thread_local, "pg_pool"):
            # Create a new pool for this thread on first use
            _thread_local.pg_pool = create_pool(__test__=__test__, thread_local=True)
            await _thread_local.pg_pool.open(wait=True)
            logger.info(
                "Created new thread-local Postgres connection pool",
                thread_name=threading.current_thread().name,
            )

        # Use the thread-local pool
        async with _thread_local.pg_pool.connection() as conn:
            yield conn
    else:
        async with _pg_pool.connection() as conn:
            yield conn


# Define an configure function that sets JSON adapters for each new connection
async def _configure_connection(conn: AsyncConnection[DictRow]):
    # Register custom JSON dumps/loads on this connection
    set_json_dumps(json_dumpb, conn)
    set_json_loads(Fragment, conn)


def create_pool(
    *, __test__: bool = False, thread_local: bool = False
) -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    # parse connection string
    params = conninfo_to_dict(config.DATABASE_URI)
    params.setdefault("options", "")
    #if not __test__:
        #params["options"] += " -c lock_timeout=1000"  # ms

    # For thread-local pools, use smaller pool sizes
    if thread_local:
        pool_min_size = 1
        pool_max_size = 10
        pool_max_idle = 30
    else:
        pool_min_size = 1
        pool_max_size = 150
        pool_max_idle = 60

    # create connection pool
    return AsyncConnectionPool(
        connection_class=AsyncConnection[DictRow],
        min_size=pool_min_size,
        max_size=pool_max_size,
        max_idle=pool_max_idle,  # seconds
        timeout=15,
        kwargs={
            **params,
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        configure=_configure_connection,
        open=False,
    )


async def create_conn(__test__: bool = False) -> AsyncConnection[DictRow]:
    params = conninfo_to_dict(config.DATABASE_URI)
    params.setdefault("options", "")
    #if not __test__:
        #params["options"] += " -c lock_timeout=1000"  # ms

    conn = await AsyncConnection.connect(
        config.DATABASE_URI,
        options=params["options"],
        row_factory=dict_row,
        autocommit=True,
        prepare_threshold=0,
    )
    await _configure_connection(conn)
    return conn


async def migrate() -> None:
    async with connect() as conn, conn.cursor() as cur:
        try:
            results = await cur.execute(
                "select version from schema_migrations order by version desc limit 1",
                prepare=False,
            )
            if row := await results.fetchone():
                current_version = row["version"]
            else:
                current_version = -1
        except UndefinedTable:
            await cur.execute(
                """
                CREATE TABLE schema_migrations (
                    version bigint primary key,
                    dirty boolean not null
                )
                """,
                prepare=False,
            )
            current_version = -1
        migration_paths = defaultdict(dict)
        for migration_path in sorted(os.listdir(config.MIGRATIONS_PATH)):
            version = int(migration_path.split("_")[0])
            which = migration_path.split(".")[-2]
            if which == "up":
                migration_paths[version]["standard"] = migration_path
            elif which == "lite":
                migration_paths[version]["lite"] = migration_path
            else:
                raise ValueError(f"Unknown migration file: {migration_path}")

        # A couple of the migrations have a "lite" fallback for those
        # whose deployments don't support certain extensions.
        postgres_extensions = config.LANGGRAPH_POSTGRES_EXTENSIONS
        for version, step_options in migration_paths.items():
            if postgres_extensions not in step_options:
                migration = step_options["standard"]
            else:
                migration = step_options[postgres_extensions]
            if version <= current_version:
                continue
            with open(os.path.join(config.MIGRATIONS_PATH, migration)) as f:
                sql = f.read().strip()
            # Split by create index concurrently statements to ensure they are executed in separate transactions
            statements = re.split(r"(?i)create\s+index\s+concurrently", sql)
            for i, stmt in enumerate(statements):
                if i > 0:
                    stmt = "CREATE INDEX CONCURRENTLY" + stmt
                try:
                    await cur.execute(stmt.strip(), prepare=False)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to apply database migration {version}\n\nStatement: {stmt}"
                    ) from e
            await cur.execute(
                "INSERT INTO schema_migrations (version, dirty) VALUES (%s, %s)",
                (version, False),
            )
            logger.info("Applied database migration", version=version)


async def migrate_vector_index():
    from langgraph_storage import store as lg_store

    if not config.STORE_CONFIG:
        logger.info(
            "No LANGGRAPH_STORE configuration found, using default configuration"
        )
        return
    config_ = config.STORE_CONFIG
    lg_store.set_store_config(config_)
    logger.info(
        "Setting up vector index",
        store_config=config_,
    )
    await lg_store.setup_vector_index(lg_store.Store())


async def start_pool() -> None:
    global _pg_pool, _stats_task

    _pg_pool = create_pool()
    # confirm connectivity
    await _pg_pool.open(wait=True)
    # migrate database
    await migrate()
    await migrate_vector_index()

    # start stats loop
    _stats_task = asyncio.create_task(stats_loop())
    # start redis
    await start_redis()


async def stats_loop() -> None:
    while True:
        logger.info("Postgres pool stats", **_pg_pool.pop_stats())
        await asyncio.sleep(config.STATS_INTERVAL_SECS)


async def stop_pool() -> None:
    global _pg_pool, _stats_task

    if threading.current_thread() is not threading.main_thread():
        # Close thread-local connection pools
        if hasattr(_thread_local, "pg_pool"):
            await _thread_local.pg_pool.close()
            del _thread_local.pg_pool
            logger.info(
                "Closed thread-local Postgres connection pool",
                thread_name=threading.current_thread().name,
            )
        return

    # stop stats loop
    _stats_task.cancel()
    try:
        await _stats_task
    except asyncio.CancelledError:
        pass
    finally:
        _stats_task = None
    # close main pool (thread-local pools are closed when the thread exits)
    await _pg_pool.close()
    _pg_pool = None
    # stop redis
    await stop_redis()


def pool_stats() -> dict[str, dict[str, int]]:
    """Get stats for the main Postgres pool"""
    return {
        "postgres": _pg_pool.get_stats(),
        "redis": redis_stats(),
    }


def get_pool() -> AsyncConnectionPool[AsyncConnection[DictRow]]:
    if threading.current_thread() is not threading.main_thread():
        return _thread_local.pg_pool
    else:
        return _pg_pool


__all__ = [
    "start_pool",
    "stop_pool",
    "connect",
    "pool_stats",
    "get_pool",
]
