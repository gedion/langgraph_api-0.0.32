import threading
from collections.abc import Callable
from typing import Any, cast

import orjson
import structlog
from langgraph.store.postgres.aio import AsyncPostgresStore, PostgresIndexConfig
from langgraph_api.config import StoreConfig, TTLConfig
from langgraph_api.graph import resolve_embeddings
from langgraph_api.serde import json_loads
from psycopg import AsyncConnection, AsyncCursor, AsyncPipeline
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool

logger = structlog.stdlib.get_logger(__name__)

_STORE_CONFIG: StoreConfig = {}


class PGSTore(AsyncPostgresStore):
    """The async store."""

    def __init__(
        self,
        conn: AsyncConnectionPool[AsyncConnection[DictRow]],
        *,
        pipe: AsyncPipeline | None = None,
        deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
        index: PostgresIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        if index is None:
            index = _STORE_CONFIG.get("index")
        if ttl is None:
            ttl = _STORE_CONFIG.get("ttl")
        super().__init__(conn, deserializer=json_loads, index=index, ttl=ttl)

    async def setup(self) -> None:
        raise NotImplementedError("Do not use the OSS's setup method.")


def set_store_config(config: StoreConfig) -> None:
    global _STORE_CONFIG
    _STORE_CONFIG = config.copy()
    if "index" not in _STORE_CONFIG or not _STORE_CONFIG["index"]:
        return
    _STORE_CONFIG["index"]["embed"] = resolve_embeddings(_STORE_CONFIG["index"])


async def setup_vector_index(store: PGSTore) -> None:
    """Set up the store database asynchronously.

    This method creates the necessary tables in the Postgres database if they don't
    already exist and runs database migrations. It MUST be called directly by the user
    the first time the store is used.
    """

    async def _get_version(cur: AsyncCursor[DictRow], table: str) -> int:
        await cur.execute(
            f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    v INTEGER PRIMARY KEY
                )
            """
        )
        await cur.execute(f"SELECT v FROM {table} ORDER BY v DESC LIMIT 1")
        row = cast(dict, await cur.fetchone())
        if row is None:
            version = -1
        else:
            version = row["v"]
        return version

    if store.index_config:
        async with store._cursor() as cur:
            version = await _get_version(cur, table="vector_migrations")
            for v, migration in enumerate(
                store.VECTOR_MIGRATIONS[version + 1 :], start=version + 1
            ):
                sql = migration.sql
                if migration.params:
                    params = {
                        k: v(store) if v is not None and callable(v) else v
                        for k, v in migration.params.items()
                    }
                    sql = sql % params
                await cur.execute(sql)
                await cur.execute("INSERT INTO vector_migrations (v) VALUES (%s)", (v,))
                await logger.ainfo("Applied vector migration", v=v)
            await logger.ainfo(
                "Done applying vector migrations",
                version=version,
            )
    else:
        await logger.awarning("No vector migrations to apply")


_STORE = threading.local()


def start_store(pool: AsyncConnectionPool[AsyncConnection[DictRow]]) -> PGSTore:
    _STORE.store = PGSTore(
        conn=pool, index=_STORE_CONFIG.get("index"), ttl=_STORE_CONFIG.get("ttl")
    )


def Store(*args: Any, **kwargs: Any) -> PGSTore:
    if not hasattr(_STORE, "store"):
        from langgraph_storage.database import get_pool

        start_store(get_pool())
    return _STORE.store
