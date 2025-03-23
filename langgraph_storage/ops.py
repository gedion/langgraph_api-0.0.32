import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from typing import Any, AsyncContextManager, Literal, cast  # noqa: UP035
from uuid import UUID, uuid4

import coredis
import coredis.commands
import coredis.exceptions
import coredis.pool
import orjson
import psycopg.errors
import structlog
from coredis.recipes.locks import LuaLock
from croniter import croniter
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base.id import uuid6
from langgraph.pregel.debug import CheckpointPayload
from langgraph.pregel.types import StateSnapshot
from langgraph_api.asyncio import SimpleTaskGroup, ValueEvent, create_task
from langgraph_api.auth.custom import handle_event
from langgraph_api.command import map_cmd
from langgraph_api.config import BG_JOB_HEARTBEAT, BG_JOB_INTERVAL
from langgraph_api.errors import UserInterrupt, UserRollback
from langgraph_api.graph import (
    GRAPHS,
    assert_graph_exists,
    get_assistant_id,
    get_graph,
    graph_exists,
)
from langgraph_api.schema import (
    Assistant,
    Checkpoint,
    Config,
    Cron,
    IfNotExists,
    MetadataInput,
    MetadataValue,
    MultitaskStrategy,
    OnConflictBehavior,
    QueueStats,
    Run,
    RunStatus,
    StreamMode,
    Thread,
    ThreadStatus,
    ThreadUpdateResponse,
)
from langgraph_api.serde import Fragment, ajson_loads
from langgraph_api.state import state_snapshot_to_thread_state
from langgraph_api.utils import fetchone, get_auth_ctx, next_cron_date
from langgraph_sdk import Auth
from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg.types.json import Jsonb
from starlette.exceptions import HTTPException

from langgraph_storage.checkpoint import Checkpointer
from langgraph_storage.database import connect
from langgraph_storage.redis import (
    CHANNEL_RUN_CONTROL,
    CHANNEL_RUN_STREAM,
    LIST_RUN_QUEUE,
    LOCK_RUN_SWEEP,
    STRING_RUN_ATTEMPT,
    STRING_RUN_CONTROL,
    STRING_RUN_RUNNING,
    get_pubsub,
    get_redis,
    get_redis_noretry,
)
from langgraph_storage.retry import RetryableException

logger = structlog.stdlib.get_logger(__name__)

StreamHandler = coredis.commands.pubsub.BasePubSub[bytes, coredis.pool.ConnectionPool]

WAIT_TIMEOUT = 5  # seconds, set to DRAIN_TIMEOUT when switching to "drain" state
DRAIN_TIMEOUT = 0.01  # drain queue, but don't wait for more

connect = cast(Callable[[], AsyncContextManager[AsyncConnection[DictRow]]], connect)


class Authenticated:
    resource: Literal["threads", "crons", "assistants"]

    @classmethod
    def _context(
        cls,
        ctx: Auth.types.BaseAuthContext | None,
        action: Literal["create", "read", "update", "delete", "search", "create_run"],
    ) -> Auth.types.AuthContext | None:
        if not ctx:
            return None
        return Auth.types.AuthContext(
            user=ctx.user,
            permissions=ctx.permissions,
            resource=cls.resource,
            action=action,
        )

    @classmethod
    async def handle_event(
        cls,
        ctx: Auth.types.BaseAuthContext | None,
        action: Literal["create", "read", "update", "delete", "search", "create_run"],
        value: Any,
    ) -> Auth.types.FilterType | None:
        ctx = ctx or get_auth_ctx()
        if not ctx:
            return None
        return await handle_event(cls._context(ctx, action), value)


class Assistants(Authenticated):
    resource = "assistants"

    @staticmethod
    async def search(
        conn: AsyncConnection[DictRow],
        *,
        graph_id: str | None,
        metadata: MetadataInput,
        limit: int,
        offset: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        metadata = metadata if metadata is not None else {}
        filters = await Assistants.handle_event(
            ctx,
            "search",
            Auth.types.AssistantsSearch(
                graph_id=graph_id, metadata=metadata, limit=limit, offset=offset
            ),
        )

        query = """SELECT * FROM assistant
        WHERE graph_id = ANY(%(graph_ids)s) AND metadata @> %(metadata)s"""
        params = {"graph_ids": list(GRAPHS.keys()), "metadata": Jsonb(metadata)}
        if graph_id:
            assert_graph_exists(graph_id)

            query += " AND graph_id = %(graph_id)s"
            params["graph_id"] = graph_id

        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="assistant",
        )
        if filter_params:
            query += filter_clause
            params.update(filter_params)

        query += " ORDER BY created_at DESC LIMIT %(limit)s OFFSET %(offset)s"
        params["limit"] = limit
        params["offset"] = offset

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur)

    @staticmethod
    async def get(
        conn: AsyncConnection[DictRow],
        assistant_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        filters = await Assistants.handle_event(
            ctx,
            "read",
            Auth.types.AssistantsRead(assistant_id=assistant_id),
        )

        query = "SELECT * FROM assistant WHERE graph_id = ANY(%(graph_id)s) AND assistant_id = %(assistant_id)s"
        params = {"graph_id": list(GRAPHS.keys()), "assistant_id": assistant_id}

        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="assistant",
        )
        if filter_params:
            query += filter_clause
            params.update(filter_params)

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur if graph_exists(row["graph_id"]))

    @staticmethod
    async def put(
        conn: AsyncConnection[DictRow],
        assistant_id: UUID,
        *,
        graph_id: str,
        config: Config,
        metadata: MetadataInput,
        if_exists: OnConflictBehavior,
        name: str,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Insert an assistant.

        Args:
            assistant_id: The assistant ID.
            graph_id: The graph ID.
            config: The assistant config.
            metadata: The assistant metadata.
            if_exists: "do_nothing" or "raise"
            name: The name of the assistant.

        Returns:
            return the assistant model if inserted.
        """
        metadata = metadata if metadata is not None else {}
        config = config if config is not None else {}
        assert_graph_exists(graph_id)
        filters = await Assistants.handle_event(
            ctx,
            "create",
            Auth.types.AssistantsCreate(
                assistant_id=assistant_id,
                graph_id=graph_id,
                config=config,
                metadata=metadata,
                name=name,
            ),
        )

        query = """WITH inserted_assistant as (
            INSERT INTO assistant (assistant_id, graph_id, config, metadata, name)
            VALUES (%(assistant_id)s, %(graph_id)s, %(config)s, %(metadata)s, %(name)s)
            ON CONFLICT (assistant_id) DO NOTHING
            RETURNING *
        ),
        inserted_version as (
            INSERT INTO assistant_versions (assistant_id, graph_id, config, metadata, version, name)
            SELECT assistant_id, graph_id, config, metadata, 1 as version, name
            FROM inserted_assistant
            ON CONFLICT (assistant_id, version) DO NOTHING
        )
        SELECT * FROM inserted_assistant
        """  # If Alice makes assistant abcd, and Bob tries to do the same, Alice will always have a version 1 existing, so this query will
        # do nothing
        params = {
            "assistant_id": assistant_id,
            "graph_id": graph_id,
            "config": Jsonb(config),
            "metadata": Jsonb(metadata),
            "name": name,
        }
        if if_exists == "do_nothing":
            filter_clause, filter_params = _build_filter_query(
                filters=filters,
            )
            # return the row if it already exists
            where_clause = "WHERE assistant_id = (%(assistant_id)s)"
            if filter_params:
                params.update(filter_params)
                where_clause += filter_clause

            query += f"""
            UNION ALL
            SELECT * FROM assistant
            {where_clause}
            LIMIT 1;
            """
        elif if_exists == "raise":
            # we'll raise downstream if there is a conflict
            pass

        cur = await conn.execute(
            query,
            params,
            binary=True,
        )
        return (row async for row in cur)

    @staticmethod
    async def patch(
        conn: AsyncConnection[DictRow],
        assistant_id: UUID,
        *,
        config: dict | None = None,
        graph_id: str | None = None,
        metadata: MetadataInput | None = None,
        name: str | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Update an assistant.

        Args:
            assistant_id: The assistant ID.
            graph_id: The graph ID.
            config: The assistant config.
            metadata: The assistant metadata.
            name: The assistant name.

        Returns:
            return the updated assistant model.
        """
        metadata = metadata if metadata is not None else {}
        config = config if config is not None else {}
        filters = await Assistants.handle_event(
            ctx,
            "update",
            Auth.types.AssistantsUpdate(
                assistant_id=assistant_id,
                graph_id=graph_id,
                config=config,
                metadata=metadata,
                name=name,
            ),
        )
        args = {
            "assistant_id": assistant_id,
            "graph_id": graph_id,
            "config": Jsonb(config) if config is not None else None,
            "metadata": Jsonb(metadata) if metadata is not None else None,
            "name": name,
        }

        update_fields = []
        if filters:
            where_clause, filter_params = _build_filter_query(
                filters=filters,
                table_alias="assistant",
            )
            args.update(filter_params)
        else:
            where_clause = ""
        if graph_id is not None:
            assert_graph_exists(graph_id)
            update_fields.append("graph_id = %(graph_id)s")
        if config:
            update_fields.append("config = %(config)s")

        if metadata:
            update_fields.append("metadata = assistant.metadata || %(metadata)s")

        if name is not None:
            update_fields.append("name = %(name)s")

        update_sql = ""
        if update_fields:
            update_sql = ", " + ", ".join(update_fields)
        query = f"""
            WITH current_assistant AS (
                SELECT * FROM assistant WHERE assistant_id = %(assistant_id)s{where_clause}
            ),
            inserted_version AS (
                INSERT INTO assistant_versions (assistant_id, graph_id, config, metadata, version, name)
                SELECT 
                    current_assistant.assistant_id,
                    COALESCE(%(graph_id)s, current_assistant.graph_id),
                    COALESCE(%(config)s, current_assistant.config),
                    CASE 
                        WHEN %(metadata)s IS NULL THEN current_assistant.metadata 
                        ELSE current_assistant.metadata || %(metadata)s::jsonb 
                    END,
                    COALESCE((SELECT MAX(version) FROM assistant_versions WHERE assistant_id = %(assistant_id)s) + 1, 1),
                    COALESCE(%(name)s, current_assistant.name)
                FROM current_assistant
                RETURNING *
            )
            UPDATE assistant
            SET version = inserted_version.version,
                updated_at = inserted_version.created_at
                {update_sql}
            FROM inserted_version
            WHERE assistant.assistant_id = %(assistant_id)s
            RETURNING *
        """

        cur = await conn.execute(query, args, binary=True)
        return (row async for row in cur)

    @staticmethod
    async def delete(
        conn: AsyncConnection[DictRow],
        assistant_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        """Delete an assistant by ID."""
        filters = await Assistants.handle_event(
            ctx,
            "delete",
            Auth.types.AssistantsDelete(
                assistant_id=assistant_id,
            ),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="assistant",
        )
        params = {"assistant_id": assistant_id, **filter_params}

        cur = await conn.execute(
            f"DELETE FROM assistant WHERE assistant_id = %(assistant_id)s{filter_clause} RETURNING assistant_id",
            params,
            binary=True,
        )
        return (row["assistant_id"] async for row in cur)

    @staticmethod
    async def set_latest(
        conn: AsyncConnection,
        assistant_id: UUID,
        version: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        filters = await Assistants.handle_event(
            ctx,
            "update",
            Auth.types.AssistantsUpdate(
                assistant_id=assistant_id,
                version=version,
            ),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="assistant",
        )
        params = {"assistant_id": assistant_id, "version": version, **filter_params}
        assistants_join = ""
        if filter_clause:
            assistants_join = "JOIN assistant ON assistant.assistant_id = assistant_versions.assistant_id"

        query = f"""
            WITH versioned_assistant AS (
                SELECT assistant_versions.* FROM assistant_versions 
                {assistants_join}
                WHERE assistant_versions.assistant_id = %(assistant_id)s AND assistant_versions.version = %(version)s{filter_clause}
            )

            UPDATE assistant
            SET 
                config = versioned_assistant.config,
                metadata = versioned_assistant.metadata,
                version = versioned_assistant.version
            FROM versioned_assistant
            WHERE assistant.assistant_id = versioned_assistant.assistant_id
            RETURNING assistant.*;
        """
        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur if graph_exists(row["graph_id"]))

    @staticmethod
    async def get_versions(
        conn: AsyncConnection[DictRow],
        assistant_id: UUID,
        metadata: MetadataInput,
        limit: int,
        offset: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Assistant]:
        """Get all versions of an assistant."""
        metadata = metadata if metadata is not None else {}
        filters = await Assistants.handle_event(
            ctx,
            "search",
            Auth.types.AssistantsRead(assistant_id=assistant_id, metadata=metadata),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="assistant",
        )
        params = {
            "assistant_id": assistant_id,
            "metadata": Jsonb(metadata),
            "limit": limit,
            "offset": offset,
            **filter_params,
        }

        join_clause = "" if not filter_params else "JOIN assistant USING (assistant_id)"

        query = f"""SELECT * FROM assistant_versions {join_clause}
        WHERE assistant_id = %(assistant_id)s AND assistant_versions.metadata @> %(metadata)s{filter_clause}
        ORDER BY assistant_versions.version DESC LIMIT %(limit)s OFFSET %(offset)s;"""

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur)


class Threads(Authenticated):
    resource = "threads"

    @staticmethod
    async def search(
        conn: AsyncConnection[DictRow],
        *,
        metadata: MetadataInput,
        values: MetadataInput,
        status: ThreadStatus | None,
        limit: int,
        offset: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Thread]:
        metadata = metadata if metadata is not None else {}
        values = values if values is not None else {}
        filters = await Threads.handle_event(
            ctx,
            "search",
            Auth.types.ThreadsSearch(
                metadata=metadata,
                values=values,
                status=status,
                limit=limit,
                offset=offset,
            ),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
            prefix="",
        )

        query = "SELECT * FROM thread"
        params = {"limit": limit, "offset": offset}
        where_clauses = []
        if metadata:
            where_clauses.append("metadata @> %(metadata)s")
            params["metadata"] = Jsonb(metadata)
        if values:
            where_clauses.append("values @> %(values)s")
            params["values"] = Jsonb(values)
        if status:
            where_clauses.append("status = %(status)s")
            params["status"] = status
        if filter_params:
            where_clauses.append(filter_clause)
            params.update(filter_params)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY created_at DESC LIMIT %(limit)s OFFSET %(offset)s"

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur)

    @staticmethod
    async def get(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
        filters: Auth.types.FilterType | None = None,
    ) -> AsyncIterator[Thread]:
        get_filters = await Threads.handle_event(
            ctx,
            "read",
            Auth.types.ThreadsRead(thread_id=thread_id),
        )
        # The parent filters, if provided, take precedence
        # since this is called from e.g., update
        # and presumably you may want to have more restrictive
        # filters on writes than on reads
        filters = {**(get_filters or {}), **(filters or {})}
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
        )
        query = "SELECT * FROM thread WHERE thread_id = %(thread_id)s"
        params = {"thread_id": thread_id}
        if filter_clause:
            query += " " + filter_clause
            params.update(filter_params)
        if filter_params:
            query += " " + filter_clause
            params.update(filter_params)

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur)

    @staticmethod
    async def put(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        *,
        metadata: MetadataInput,
        if_exists: OnConflictBehavior,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Thread]:
        """Insert or update a thread."""
        metadata = metadata if metadata is not None else {}
        filters = await Threads.handle_event(
            ctx,
            "create",
            Auth.types.ThreadsCreate(
                thread_id=thread_id, metadata=metadata, if_exists=if_exists
            ),
        )

        query = """WITH inserted_thread as (
            INSERT INTO thread (thread_id, metadata)
            values (%(thread_id)s, %(metadata)s)
            ON CONFLICT (thread_id) DO NOTHING
            RETURNING *
        )
        SELECT * FROM inserted_thread
        """
        params = {
            "thread_id": thread_id,
            "metadata": Jsonb(metadata),
        }
        if if_exists == "do_nothing":
            # return the row if it already exists
            filter_clause, filter_params = _build_filter_query(
                filters=filters, metadata_field="metadata"
            )
            where_clause = "WHERE thread_id = %(thread_id)s"
            if filter_params:
                params.update(filter_params)
                where_clause += filter_clause

            query += f"""
            UNION ALL
            SELECT * FROM thread
            {where_clause}
            LIMIT 1;
            """
        elif if_exists == "raise":
            # we'll raise downstream if there is a conflict
            pass

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur)

    @staticmethod
    async def patch(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        *,
        metadata: MetadataValue,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Thread]:
        metadata = metadata if metadata is not None else {}
        filters = await Threads.handle_event(
            ctx,
            "update",
            Auth.types.ThreadsUpdate(thread_id=thread_id, metadata=metadata),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
        )
        params = {"metadata": Jsonb(metadata), "thread_id": thread_id}
        where_clause = "WHERE thread_id = %(thread_id)s"
        if filter_params:
            params.update(filter_params)
            where_clause += filter_clause

        cur = await conn.execute(
            f"""update thread
            set metadata = metadata || %(metadata)s
            {where_clause}
            returning *;""",
            params,
            binary=True,
        )
        return (row async for row in cur)

    @staticmethod
    async def set_status(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        checkpoint: CheckpointPayload | None,
        exception: BaseException | None,
    ) -> None:
        """Set the status of a thread."""
        # No auth since it's internal
        if checkpoint is None:
            has_next = False
        else:
            has_next = bool(checkpoint["next"])
        if exception and not isinstance(exception, UserInterrupt | UserRollback):
            status = "error"
        elif has_next:
            status = "interrupted"
        else:
            status = "idle"
        interrupts = (
            {
                t["id"]: t["interrupts"]
                for t in checkpoint["tasks"]
                if t.get("interrupts")
            }
            if checkpoint
            else {}
        )
        async with await conn.execute(
            """update thread set
            updated_at = now(),
            values = %(values)s,
            interrupts = %(interrupts)s,
            status = case
                when exists(
                    select 1 from run
                    where thread_id = %(thread_id)s
                    and status in ('pending', 'running')
                ) then 'busy'
                else %(status)s
            end
            where thread_id = %(thread_id)s
            returning status;
            """,
            {
                "thread_id": thread_id,
                "status": status,
                "values": Jsonb(checkpoint["values"]) if checkpoint else None,
                "interrupts": Jsonb(interrupts),
            },
            binary=True,
        ) as cur:
            async for row in cur:
                if row["status"] == "busy":
                    # there's more runs for this thread, wake up the worker
                    # this happens when multitask_strategy != "reject"
                    await wake_up_worker()

    @staticmethod
    async def delete(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        """Delete a thread by ID."""
        filters = await Threads.handle_event(
            ctx,
            "delete",
            Auth.types.ThreadsDelete(thread_id=thread_id),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
        )
        params = {"thread_id": thread_id, **filter_params}
        cur = await conn.execute(
            f"DELETE FROM thread WHERE thread_id = %(thread_id)s{filter_clause} RETURNING thread_id",
            params,
            binary=True,
        )
        return (row["thread_id"] async for row in cur)

    @staticmethod
    async def copy(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Thread]:
        """Create a copy of an existing thread."""
        filters = await Threads.handle_event(
            ctx,
            "read",
            Auth.types.ThreadsRead(
                thread_id=thread_id,
            ),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
        )
        where_clause = f"WHERE thread_id = %(thread_id)s{filter_clause}"
        thread_join = "JOIN thread USING (thread_id) " if filter_params else ""
        new_thread_id = uuid4()
        query_thread_params = {
            "new_thread_id": new_thread_id,
            "thread_id": thread_id,
            **filter_params,
        }

        async with conn.pipeline():
            cur = await conn.execute(
                f"""INSERT INTO thread (thread_id, metadata)
                SELECT %(new_thread_id)s, metadata
                FROM thread
                {where_clause}
                ON CONFLICT (thread_id) DO NOTHING
                RETURNING *""",
                query_thread_params,
            )
            # then, copy all of the checkpoint data in parallel
            await asyncio.gather(
                conn.execute(
                    f"""
                    INSERT INTO checkpoints (run_id, thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
                    SELECT run_id, %(new_thread_id)s, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, jsonb_set(
                        checkpoints.metadata,
                        '{{thread_id}}',
                        to_jsonb(%(new_thread_id)s)
                    )
                    FROM checkpoints
                    {thread_join}
                    {where_clause}
                    ON CONFLICT DO NOTHING
                    """,
                    query_thread_params,
                ),
                conn.execute(
                    f"""
                    INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                    SELECT %(new_thread_id)s, checkpoint_ns, channel, version, type, blob
                    FROM checkpoint_blobs
                    {thread_join}
                    {where_clause}
                    ON CONFLICT DO NOTHING
                    """,
                    query_thread_params,
                ),
                conn.execute(
                    f"""
                    INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                    SELECT %(new_thread_id)s, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob
                    FROM checkpoint_writes
                    {thread_join}
                    {where_clause}
                    ON CONFLICT DO NOTHING
                    """,
                    query_thread_params,
                ),
            )
        return (row async for row in cur)

    class State(Authenticated):
        # treat this like threads resource
        resource = "threads"

        @staticmethod
        async def get(
            conn: AsyncConnection[DictRow],
            config: Config,
            subgraphs: bool,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> StateSnapshot:
            checkpointer = Checkpointer(conn)
            # fetch both in parallel
            async with conn.pipeline():
                thread, checkpoint = await asyncio.gather(
                    Threads.get(conn, config["configurable"]["thread_id"], ctx=ctx),
                    checkpointer.aget_iter(cast(RunnableConfig, config)),
                )
            thread = await fetchone(thread)
            metadata = await ajson_loads(thread["metadata"])
            thread_config = await ajson_loads(thread["config"])

            # Filters already applied in Threads.get so no need to use them again here

            if graph_id := metadata.get("graph_id"):
                # format latest checkpoint for response
                checkpointer.latest_iter = checkpoint
                async with get_graph(
                    graph_id, thread_config, checkpointer=checkpointer
                ) as graph:
                    return await graph.aget_state(config, subgraphs=subgraphs)
            else:
                return StateSnapshot(
                    values={},
                    next=[],
                    config=None,
                    metadata=None,
                    created_at=None,
                    parent_config=None,
                    tasks=tuple(),
                )

        @staticmethod
        async def post(
            conn: AsyncConnection[DictRow],
            config: Config,
            values: Sequence[dict] | dict[str, Any] | None,
            as_node: str | None = None,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> ThreadUpdateResponse:
            thread_id = UUID(config["configurable"]["thread_id"])
            filters = await Threads.State.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id),
            )

            checkpointer = Checkpointer(conn)
            # fetch both in parallel
            async with conn.pipeline():
                thread, checkpoint = await asyncio.gather(
                    Threads.get(
                        conn,
                        config["configurable"]["thread_id"],
                        ctx=ctx,
                        # This lets us use update filters on the get
                        # operation if we want
                        filters=filters,
                    ),
                    checkpointer.aget_iter(cast(RunnableConfig, config)),
                )
            thread = await fetchone(thread)
            metadata = await ajson_loads(thread["metadata"])
            thread_config = await ajson_loads(thread["config"])
            if graph_id := metadata.get("graph_id"):
                # update state
                config["configurable"].setdefault("graph_id", graph_id)
                checkpointer.latest_iter = checkpoint
                async with AsyncExitStack() as stack:
                    graph = await stack.enter_async_context(
                        get_graph(graph_id, thread_config, checkpointer=checkpointer)
                    )
                    await stack.enter_async_context(conn.transaction())
                    next_config = await graph.aupdate_state(
                        cast(RunnableConfig, config), values, as_node=as_node
                    )
                    # update thread values
                    state = await Threads.State.get(
                        conn, config, subgraphs=False, ctx=ctx
                    )
                    await Threads.set_status(
                        conn,
                        thread_id,
                        state_snapshot_to_thread_state(state),
                        None,
                    )
                    return {
                        "checkpoint": next_config["configurable"],
                        # below are deprecated
                        **next_config,
                        "checkpoint_id": next_config["configurable"]["checkpoint_id"],
                    }
            else:
                raise HTTPException(status_code=400, detail="Thread has no graph ID.")

        @staticmethod
        async def bulk(
            conn: AsyncConnection[DictRow],
            config: Config,
            supersteps: Sequence[dict],
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> ThreadUpdateResponse:
            """Update a thread with a batch of state updates."""

            from langgraph.pregel.types import StateUpdate

            thread_id = UUID(config["configurable"]["thread_id"])
            filters = await Threads.State.handle_event(
                ctx,
                "update",
                Auth.types.ThreadsUpdate(thread_id=thread_id),
            )

            checkpointer = Checkpointer(conn)

            thread = await fetchone(
                await Threads.get(conn, thread_id, ctx=ctx, filters=filters)
            )
            thread_config = await ajson_loads(thread["config"])
            metadata = await ajson_loads(thread["metadata"])

            if graph_id := metadata.get("graph_id"):
                # update state
                config["configurable"].setdefault("graph_id", graph_id)
                config["configurable"].setdefault("checkpoint_ns", "")

                async with AsyncExitStack() as stack:
                    graph = await stack.enter_async_context(
                        get_graph(graph_id, thread_config, checkpointer=checkpointer)
                    )

                    await stack.enter_async_context(conn.transaction())
                    next_config = await graph.abulk_update_state(
                        config,
                        [
                            [
                                StateUpdate(
                                    map_cmd(update.get("command"))
                                    if update.get("command")
                                    else update.get("values"),
                                    update.get("as_node"),
                                )
                                for update in superstep.get("updates", [])
                            ]
                            for superstep in supersteps
                        ],
                    )

                    # update thread values
                    state = await Threads.State.get(
                        conn, config, subgraphs=False, ctx=ctx
                    )

                    await Threads.set_status(
                        conn,
                        thread_id,
                        state_snapshot_to_thread_state(state),
                        None,
                    )

                    return ThreadUpdateResponse(checkpoint=next_config["configurable"])
            else:
                raise HTTPException(status_code=400, detail="Thread has no graph ID")

        @staticmethod
        async def list(
            conn: AsyncConnection[DictRow],
            *,
            config: Config,
            limit: int = 10,
            before: str | Checkpoint | None = None,
            metadata: MetadataInput = None,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> list[StateSnapshot]:
            """Get the history of a thread."""
            thread = await fetchone(
                await Threads.get(conn, config["configurable"]["thread_id"], ctx=ctx)
            )
            thread_metadata = await ajson_loads(thread["metadata"])
            thread_config = await ajson_loads(thread["config"])
            if graph_id := thread_metadata.get("graph_id"):
                async with get_graph(
                    graph_id, thread_config, checkpointer=Checkpointer(conn)
                ) as graph:
                    return [
                        c
                        async for c in graph.aget_state_history(
                            config,
                            limit=limit,
                            filter=metadata,
                            before=(
                                {"configurable": {"checkpoint_id": before}}
                                if isinstance(before, str)
                                else before
                            ),
                        )
                    ]
            else:
                return []


class Runs(Authenticated):
    # Auth for runs is applied at the thread level.
    # We do have a special "create_run" handler, however, to let
    # users add checks for runs in particular
    resource = "threads"

    @staticmethod
    async def stats(conn: AsyncConnection[DictRow]) -> QueueStats:
        # We don't have auth on stats right now
        async with await conn.execute(
            """select
        count(*) filter (where status = 'pending') as n_pending,
        count(*) filter (where status = 'running') as n_running,
        extract(epoch from (min(now() - created_at))) as min_age_secs,
        extract(epoch from (percentile_cont(0.5) within group (order by now() - created_at))) as med_age_secs
    from run where status in ('pending', 'running')
    """
            # TODO: add running
        ) as cur:
            stats = await cur.fetchone()
            if stats["min_age_secs"]:
                stats["min_age_secs"] = float(stats["min_age_secs"])
            if stats["med_age_secs"]:
                stats["med_age_secs"] = float(stats["med_age_secs"])
            return stats

    @staticmethod
    async def next(wait: bool, limit: int = 1) -> AsyncIterator[tuple[Run, int]]:
        """Get the next run from the queue, and the attempt number.
        1 is the first attempt, 2 is the first retry, etc."""
        # Internal for workers, no auth here.

        # wait for a run to be available (or check every BG_JOB_INTERVAL anyway)
        # all scenarios that make a run available for running need to wake_up_worker()
        # - a new run is created - Runs.put()
        # - a run is marked for retry - Runs.set_status()
        # - a run finishes with other runs pending in same thread - Threads.set_status()
        if wait:
            try:
                await get_redis_noretry().blpop(
                    [LIST_RUN_QUEUE], timeout=BG_JOB_INTERVAL
                )
            except coredis.exceptions.ConnectionError:
                return
        else:
            await asyncio.sleep(0)

        # get the run
        async with (
            connect() as conn,
            conn.transaction(),
            await conn.execute(
                """
                with selected as (
                    select *
                    from run
                    where run.status = 'pending'
                        and run.created_at < now()
                        and not exists (
                            select 1 from run r2
                            where r2.thread_id = run.thread_id
                                and r2.status = 'running'
                        )
                    order by run.created_at
                    limit %(limit)s
                )
                update run set status = 'running'
                from selected
                where run.run_id = selected.run_id
                returning run.*;
                """,
                {"limit": limit},
                binary=True,
            ) as cur,
        ):
            runs = await cur.fetchall()
            # get attempt number, and set running marker
            async with await get_redis().pipeline() as pipe:
                for run in runs:
                    await pipe.set(
                        STRING_RUN_RUNNING.format(run["run_id"]),
                        "1",
                        ex=BG_JOB_HEARTBEAT,
                    )
                    await pipe.expire(STRING_RUN_ATTEMPT.format(run["run_id"]), 60)
                    await pipe.incrby(STRING_RUN_ATTEMPT.format(run["run_id"]), 1)
                results = await pipe.execute()
            # yield each run and its attempt number
            for idx, run in enumerate(runs):
                run["kwargs"] = await ajson_loads(run["kwargs"])
                run["metadata"] = await ajson_loads(run["metadata"])
                attempt = results[idx * 3 + 2]
                yield run, attempt

    @asynccontextmanager
    @staticmethod
    async def enter(
        run_id: UUID, loop: asyncio.AbstractEventLoop
    ) -> AsyncIterator[ValueEvent]:
        """Enter a run, listen for cancellation while running, signal when done."
        This method should be called as a context manager by a worker executing a run.
        """
        async with get_pubsub() as pubsub, SimpleTaskGroup(cancel=True) as tg:
            done = ValueEvent()
            # start listener, will be cancelled when exiting context
            tg.create_task(listen_for_cancellation(pubsub, run_id, done))
            # start heartbeat, will be cancelled when exiting context
            hb = asyncio.run_coroutine_threadsafe(heartbeat(run_id), loop)
            # give done event to caller
            try:
                yield done
                # signal done
                await get_redis().publish(CHANNEL_RUN_CONTROL.format(run_id), "done")
            finally:
                hb.cancel()

    @staticmethod
    async def sweep(conn: AsyncConnection[DictRow]) -> list[UUID]:
        """Sweep runs that have been in running state for too long."""
        async with LuaLock(get_redis_noretry(), LOCK_RUN_SWEEP, timeout=30.0):
            cur = await conn.execute(
                """
                    select run_id
                    from run
                    where status = 'running'
                """
            )
            run_ids = [row["run_id"] async for row in cur]
            if not run_ids:
                return []
            exists = await get_redis().mget(
                [STRING_RUN_RUNNING.format(run_id) for run_id in run_ids]
            )
            to_sweep = [
                run_id
                for run_id, exists in zip(run_ids, exists, strict=True)
                if exists is None
            ]
            if to_sweep:
                try:
                    await conn.execute(
                        """
                        update run
                        set status = 'pending'
                        where run_id = any(%(run_ids)s)
                            and status = 'running'
                        """,
                        {"run_ids": to_sweep},
                    )
                    await wake_up_worker()
                    return to_sweep
                except psycopg.errors.IntegrityError:
                    # catch concurrent update error
                    logger.warning(
                        "Tried to sweep runs that are no longer running",
                        run_ids=to_sweep,
                    )
            return []

    @staticmethod
    async def put(
        conn: AsyncConnection[DictRow],
        assistant_id: UUID,
        kwargs: dict,
        *,
        thread_id: UUID | None = None,
        user_id: str | None = None,
        run_id: UUID | None = None,
        status: RunStatus | None = "pending",
        metadata: MetadataInput,
        prevent_insert_if_inflight: bool,
        multitask_strategy: MultitaskStrategy = "reject",
        if_not_exists: IfNotExists = "reject",
        after_seconds: int = 0,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Run]:
        """Create a run."""
        metadata = metadata or {}
        metadata.setdefault("assistant_id", assistant_id)
        kwargs = kwargs or {}
        kwargs.setdefault("config", {})
        filters = await Runs.handle_event(
            ctx,
            "create_run",
            Auth.types.RunsCreate(
                thread_id=thread_id,
                assistant_id=assistant_id,
                run_id=run_id,
                status=status,
                metadata=metadata,
                prevent_insert_if_inflight=prevent_insert_if_inflight,
                multitask_strategy=multitask_strategy,
                if_not_exists=if_not_exists,
                after_seconds=after_seconds,
                kwargs=kwargs,
            ),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
        )
        thread_join = "JOIN thread USING (thread_id) " if filter_params else ""

        thread_query_cte = (
            f"""WITH inserted_thread AS (
                INSERT INTO thread (thread_id, status, metadata, config)
                SELECT
                    %(thread_id)s,
                    'busy',
                    jsonb_build_object(
                        'graph_id', assistant.graph_id,
                        'assistant_id', assistant.assistant_id
                    ) || %(metadata)s::jsonb,
                    assistant.config
                    || %(config)s::jsonb
                    || jsonb_build_object(
                        'configurable',
                            coalesce((assistant.config -> 'configurable'), '{{}}') ||
                            coalesce(%(config)s::jsonb -> 'configurable', '{{}}')
                       )
                FROM assistant
                WHERE assistant_id = %(assistant_id)s
                ON CONFLICT (thread_id) DO NOTHING
                RETURNING *
            ),
            
            run_thread AS (
                SELECT * FROM thread where thread_id = %(thread_id)s {filter_clause}
                UNION ALL
                SELECT * FROM inserted_thread
            ),"""
            if thread_id is None or if_not_exists == "create"
            else f"""WITH run_thread AS (
                        SELECT * FROM thread 
                        WHERE thread_id = %(thread_id)s
                             {filter_clause}),"""
        )

        params = {
            "multitask_strategy": multitask_strategy,
            "run_id": run_id or uuid6(),
            "thread_id": thread_id or uuid4(),
            "assistant_id": assistant_id,
            "metadata": Jsonb(metadata),
            "kwargs": Jsonb(kwargs),
            "config": Jsonb(kwargs.get("config")),
            "status": status,
            "user_id": user_id,
            "after_seconds": f"{after_seconds} second",
        }
        params.update(filter_params)

        query = (
            thread_query_cte
            + f"""
inflight_runs AS (
    SELECT run.*
    FROM run
    {thread_join}
    WHERE thread_id = %(thread_id)s AND run.status in ('pending', 'running') {filter_clause}
),

inserted_run AS (
    INSERT INTO run (run_id, thread_id, assistant_id, metadata, status, kwargs, multitask_strategy, created_at)
    SELECT
        %(run_id)s,
        thread_id,
        assistant_id,
        %(metadata)s,
        %(status)s,
        %(kwargs)s::jsonb || jsonb_build_object(
            'config', assistant.config || run_thread.config || %(config)s::jsonb || jsonb_build_object(
                'configurable',
                    coalesce((assistant.config -> 'configurable'), '{{}}') ||
                    coalesce((run_thread.config -> 'configurable'), '{{}}') ||
                    coalesce(%(config)s::jsonb -> 'configurable', '{{}}') ||
                    jsonb_build_object(
                        'run_id', %(run_id)s::text,
                        'thread_id', thread_id,
                        'graph_id', graph_id,
                        'assistant_id', assistant_id,
                        'user_id', coalesce(
                            %(config)s::jsonb -> 'configurable' ->> 'user_id',
                            run_thread.config -> 'configurable' ->> 'user_id',
                            assistant.config -> 'configurable' ->> 'user_id',
                            %(user_id)s::text
                        )
                    ),
                'metadata',
                    assistant.metadata || run_thread.metadata || %(metadata)s
            )
        ),
        %(multitask_strategy)s,
        now() + %(after_seconds)s::interval
    FROM run_thread
    CROSS JOIN assistant
    WHERE thread_id = %(thread_id)s
        AND assistant_id = %(assistant_id)s"""
            + (
                " AND NOT EXISTS (SELECT 1 FROM inflight_runs)"
                if prevent_insert_if_inflight
                else ""
            )
            + """ RETURNING run.*
),

updated_thread AS (
    UPDATE thread SET
        metadata = jsonb_set(
            jsonb_set(thread.metadata, '{graph_id}', to_jsonb(assistant.graph_id)),
            '{assistant_id}',
            to_jsonb(assistant.assistant_id)
        ),
        config = assistant.config
            || thread.config
            || %(config)s::jsonb
            || jsonb_build_object(
                'configurable',
                    coalesce((assistant.config -> 'configurable'), '{}') ||
                    coalesce(thread.config -> 'configurable', '{}') ||
                    coalesce(%(config)s::jsonb -> 'configurable', '{}')
                ),
        status = 'busy'
    FROM inserted_run
    INNER JOIN assistant
        ON assistant.assistant_id = inserted_run.assistant_id
    WHERE
        thread.thread_id = inserted_run.thread_id
        AND thread.status != 'busy'
)

SELECT * FROM inserted_run
UNION ALL
SELECT * FROM inflight_runs"""
        )

        cur = await conn.execute(query, params, binary=True)

        async def consume() -> AsyncIterator[Run]:
            async for row in cur:
                yield row
                if row["run_id"] == run_id:
                    # inserted run, notify queue
                    if not after_seconds:
                        await wake_up_worker()
                    else:
                        create_task(wake_up_worker(after_seconds))

        return consume()

    @staticmethod
    async def get(
        conn: AsyncConnection[DictRow],
        run_id: UUID,
        *,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Run]:
        """Get a run by ID."""
        filters = await Runs.handle_event(
            ctx,
            "read",
            Auth.types.ThreadsRead(run_id=run_id, thread_id=thread_id),
        )

        where_clause, where_params = _build_filter_query(
            filters=filters, table_alias="thread"
        )

        query = f"""SELECT run.*
        FROM run
        JOIN thread USING (thread_id)
        WHERE run_id = %(run_id)s AND run.thread_id = %(thread_id)s
        {where_clause}
        """
        cur = await conn.execute(
            query,
            {**where_params, "run_id": run_id, "thread_id": thread_id},
            binary=True,
        )
        return (row async for row in cur)

    @staticmethod
    async def delete(
        conn: AsyncConnection[DictRow],
        run_id: UUID,
        *,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        filters = await Runs.handle_event(
            ctx,
            "delete",
            Auth.types.ThreadsDelete(run_id=run_id, thread_id=thread_id),
        )

        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
        )
        thread_join = (
            "JOIN thread USING (thread_id) WHERE" if filter_params else "WHERE"
        )

        params = {**filter_params, "run_id": run_id, "thread_id": thread_id}
        async with conn.transaction():
            cur = await conn.execute(
                f"""
                WITH selected AS (
                    SELECT run_id
                    FROM run
                    {thread_join}
                    run_id = %(run_id)s 
                        AND run.thread_id = %(thread_id)s 
                        {filter_clause} 
                ),
                
                del_checkpoint_writes AS (
                    DELETE FROM checkpoint_writes
                    USING selected
                    INNER JOIN checkpoints
                        ON checkpoints.run_id = selected.run_id
                    WHERE checkpoint_writes.checkpoint_id = checkpoints.checkpoint_id
                        AND checkpoint_writes.thread_id = checkpoints.thread_id
                        AND checkpoint_writes.checkpoint_ns = checkpoints.checkpoint_ns
                )

                DELETE FROM run 
                USING selected
                WHERE run.run_id = selected.run_id
                RETURNING run.run_id""",
                params,
                binary=True,
            )
        return (row["run_id"] async for row in cur)

    @staticmethod
    async def join(
        run_id: UUID,
        *,
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> Fragment:
        """Wait for a run to complete. If already done, return immediately.

        Returns:
            the final state of the run.
        """
        last_chunk: bytes | None = None
        # wait for the run to complete
        async for mode, chunk in Runs.Stream.join(
            run_id, thread_id=thread_id, stream_mode="values", ctx=ctx, ignore_404=True
        ):
            if mode == b"values":
                last_chunk = chunk
            elif mode == b"error":
                last_chunk = orjson.dumps({"__error__": orjson.Fragment(chunk)})
        # if we received a final chunk, return it
        if last_chunk is not None:
            # ie. if the run completed while we were waiting for it
            return Fragment(last_chunk)
        else:
            # otherwise, the run had already finished, so fetch the state from thread
            async with connect() as conn:
                thread_iter = await Threads.get(conn, thread_id)
                thread = await fetchone(thread_iter)
                return thread["values"]

    @staticmethod
    async def cancel(
        conn: AsyncConnection[DictRow],
        run_ids: Sequence[UUID],
        *,
        action: Literal["interrupt", "rollback"] = "interrupt",
        thread_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> None:
        """Cancel a run."""
        # cancellation tries to take two actions, to cover runs in different states
        # - for any run, set the control key to interrupt or rollback (this is read by
        #   the worker after subscribing to the control channel)
        # - for queued run not yet picked up by a worker, remove it from queue by
        #   updating status
        # - for run currently being worked on (locked), we notify the worker
        #   to cancel the run
        # - for runs in any other state, we raise a 404
        filters = await Runs.handle_event(
            ctx,
            "update",
            Auth.types.ThreadsUpdate(
                thread_id=thread_id,
                action=action,
                metadata={"run_ids": run_ids},
            ),
        )
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="thread",
        )
        params = {"run_ids": run_ids, "thread_id": thread_id, "action": action}
        thread_join = ""
        if filter_params:
            thread_join = "JOIN thread USING (thread_id)"
            params.update(filter_params)

        async with await get_redis().pipeline() as pipe:
            for run_id in run_ids:
                await pipe.set(STRING_RUN_CONTROL.format(run_id), action, ex=60)
                await pipe.publish(CHANNEL_RUN_CONTROL.format(run_id), action)
            cur, _ = await asyncio.gather(
                conn.execute(
                    f"""
                    with
                    
                    running as (
                        select run_id
                        from run
                        {thread_join}
                        where run_id = any(%(run_ids)s)
                            and thread_id = %(thread_id)s
                            {filter_clause}
                            and run.status = 'running'
                    ),

                    -- Not currently being worked on
                    pending as (
                        select run_id
                        from run
                        {thread_join}
                        where run_id = any(%(run_ids)s)
                            and thread_id = %(thread_id)s
                            {filter_clause}
                            and run.status = 'pending'
                    ),

                    updated as (
                        update run
                        set status = 'interrupted'
                        from pending
                        where run.run_id = pending.run_id
                        and %(action)s = 'interrupt'
                        returning run.run_id
                    ),
                    
                    deleted AS (
                        DELETE FROM run
                        USING pending
                        WHERE run.run_id = pending.run_id
                        AND %(action)s = 'rollback'
                        RETURNING run.run_id
                    ),

                    unioned as (
                        select run_id, true as done
                        from updated
                        union all
                        select run_id, true as done
                        from deleted
                        union all
                        select run_id, false as done
                        from running
                    )
                    
                    select run_id, bool_and(done) as done
                    from unioned
                    group by run_id
                    """,
                    params,
                    binary=True,
                ),
                pipe.execute(),
            )
        found = [row["run_id"] async for row in cur]
        if len(found) == len(run_ids):
            logger.info(
                "Cancelled runs", run_ids=run_ids, thread_id=thread_id, action=action
            )
            pass
        else:
            raise HTTPException(status_code=404, detail="Run not found")

    @staticmethod
    async def search(
        conn: AsyncConnection[DictRow],
        thread_id: UUID,
        *,
        limit: int = 10,
        offset: int = 0,
        metadata: MetadataInput,
        status: RunStatus | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Run]:
        """List all runs by thread."""
        metadata = metadata if metadata is not None else {}
        filters = await Runs.handle_event(
            ctx,
            "search",
            Auth.types.ThreadsSearch(thread_id=thread_id, metadata=metadata),
        )
        filters_clause, filter_params = _build_filter_query(
            filters=filters, table_alias="thread"
        )
        threads_join = "" if not filter_params else "JOIN thread USING (thread_id)"
        query = f"""SELECT run.*
        FROM run
        {threads_join}
        WHERE run.thread_id = %(thread_id)s AND run.metadata @> %(metadata)s {filters_clause}"""
        params = {**filter_params, "thread_id": thread_id, "metadata": Jsonb(metadata)}

        if status is not None:
            query += " AND run.status = %(status)s::text"
            params["status"] = status

        query += " ORDER BY run.created_at DESC LIMIT %(limit)s OFFSET %(offset)s"
        params["limit"] = limit
        params["offset"] = offset

        cur = await conn.execute(query, params, binary=True)
        return (row async for row in cur)

    @staticmethod
    async def set_status(
        conn: AsyncConnection[DictRow], run_id: UUID, status: RunStatus
    ) -> None:
        """Set the status of a run."""
        # Internal method - no auth
        await conn.execute(
            "UPDATE run SET status = %s WHERE run_id = %s",
            (status, run_id),
            binary=True,
        )
        if status == "pending":
            await wake_up_worker()

    class Stream(Authenticated):
        resource = "threads"

        @staticmethod
        async def subscribe(
            run_id: UUID,
            *,
            stream_mode: StreamMode | None = None,
        ) -> StreamHandler:
            """Subscribe to the run stream, returning a stream handler.
            The stream handler must be passed to `join` to receive messages."""
            pubsub = get_pubsub()
            control_channel = CHANNEL_RUN_CONTROL.format(run_id)
            if stream_mode is None:
                await pubsub.psubscribe(
                    CHANNEL_RUN_STREAM.format(run_id, "*"), control_channel
                )
            else:
                await pubsub.subscribe(
                    CHANNEL_RUN_STREAM.format(run_id, stream_mode), control_channel
                )
            return pubsub

        @staticmethod
        async def join(
            run_id: UUID,
            *,
            thread_id: UUID,
            ignore_404: bool = False,
            cancel_on_disconnect: bool = False,
            stream_mode: StreamMode | StreamHandler | None = None,
            ctx: Auth.types.BaseAuthContext | None = None,
        ) -> AsyncIterator[tuple[bytes, bytes]]:
            """Stream the run output, either from a stream handler or a stream mode."""
            filters = await Runs.Stream.handle_event(
                ctx,
                "read",
                Auth.types.ThreadsRead(run_id=run_id, thread_id=thread_id),
            )
            filter_clause, filter_params = _build_filter_query(
                filters=filters, table_alias="thread"
            )
            if filter_params:
                # TODO: This should be pushed downwards:
                query = f"""
                SELECT run_id FROM run
                JOIN thread USING (thread_id)
                WHERE run_id = %(run_id)s AND thread_id = %(thread_id)s
                {filter_clause}
                """
                params = {**filter_params, "run_id": run_id, "thread_id": thread_id}
                async with connect() as conn:
                    cur = await conn.execute(query, params, binary=True)
                    if not await cur.fetchone():
                        raise HTTPException(status_code=404, detail="Thread not found")

            log = logger.isEnabledFor(logging.DEBUG)
            pubsub: StreamHandler | None = None
            try:
                pubsub = (
                    stream_mode
                    if isinstance(stream_mode, coredis.commands.pubsub.BasePubSub)
                    else get_pubsub()
                )
                async with pubsub, connect() as conn:
                    # subscribe to events
                    control_channel = CHANNEL_RUN_CONTROL.format(run_id)
                    if stream_mode is pubsub:
                        pass  # already subscribed
                    elif stream_mode is None:
                        await pubsub.psubscribe(
                            CHANNEL_RUN_STREAM.format(run_id, "*"), control_channel
                        )
                    else:
                        await pubsub.subscribe(
                            CHANNEL_RUN_STREAM.format(run_id, stream_mode),
                            control_channel,
                        )
                    logger.info(
                        "Joined run stream",
                        run_id=str(run_id),
                        thread_id=str(thread_id),
                    )
                    # stream events
                    len_prefix = len(CHANNEL_RUN_STREAM.format(run_id, "").encode())
                    timeout = WAIT_TIMEOUT
                    while True:
                        if event := await pubsub.get_message(True, timeout=timeout):
                            if event["channel"] == control_channel.encode():
                                if event["data"] == b"done":
                                    timeout = DRAIN_TIMEOUT
                            else:
                                # mode, data
                                yield event["channel"][len_prefix:], event["data"]
                                if log:
                                    logger.debug(
                                        "Streamed run event",
                                        run_id=str(run_id),
                                        stream_mode=event["channel"][len_prefix:],
                                        data=event["data"],
                                    )
                        elif timeout == DRAIN_TIMEOUT:
                            break
                        else:
                            run_iter = await Runs.get(
                                conn, run_id, thread_id=thread_id, ctx=ctx
                            )
                            run = await anext(run_iter, None)
                            if run is None or run["status"] not in (
                                "pending",
                                "running",
                            ):
                                timeout = DRAIN_TIMEOUT
                            if run is None and not ignore_404:
                                yield (
                                    b"error",
                                    HTTPException(
                                        status_code=404, detail="Run not found"
                                    ),
                                )
            except asyncio.CancelledError:
                if pubsub:
                    pubsub.close()
                if cancel_on_disconnect:
                    create_task(cancel_run(thread_id, run_id))
                raise

        @staticmethod
        async def publish(
            run_id: UUID,
            event: str,
            message: bytes,
        ) -> None:
            await get_redis().publish(CHANNEL_RUN_STREAM.format(run_id, event), message)


class Crons(Authenticated):
    resource = "crons"

    @staticmethod
    async def put(
        conn: AsyncConnection[DictRow],
        *,
        payload: dict,
        schedule: str,
        cron_id: UUID | None = None,
        thread_id: UUID | None = None,
        end_time: datetime | None = None,
        metadata: dict | None = None,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Cron]:
        ctx = get_auth_ctx()
        user_id = ctx.user.identity if ctx is not None else None
        cron_id = cron_id or uuid6()
        try:
            thread_id = str(UUID(thread_id)) if thread_id else None
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid thread ID {thread_id}. Expected a UUID.",
            ) from None
        metadata = metadata if metadata is not None else {}
        payload = payload if payload is not None else {}
        request_data = Auth.types.CronsCreate(
            payload=payload,
            schedule=schedule,
            cron_id=cron_id,
            thread_id=thread_id,
            user_id=user_id,
            end_time=end_time,
        )
        request_data["metadata"] = metadata  # type: ignore
        filters = await Crons.handle_event(ctx, "create", request_data)

        if not croniter.is_valid(schedule):
            raise HTTPException(status_code=422, detail="Invalid cron schedule")

        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="c",
        )

        if filter_params and payload["assistant_id"] not in GRAPHS:
            # Auth filters present and assistant is a custom one (not a generic graph)
            # Need to build assistant filter queries too.
            assistant_filter_clause, assistant_filter_params = _build_filter_query(
                filters=filters,
                table_alias="assistant",
                start=len(filter_params),
            )
            filter_params.update(assistant_filter_params)
            authorized_assistant_cte = f"""
WITH authorized_assistant AS (
    SELECT assistant.assistant_id
    FROM assistant
    WHERE assistant.assistant_id = %(assistant_id)s
    {assistant_filter_clause}  -- only yield assistant if user can see it
), """
            insert_assistant_select = "authorized_assistant.assistant_id"
            insert_assistant_from = "FROM authorized_assistant"
        else:
            # Auth filters not present or assistant is a generic graph
            authorized_assistant_cte = "with "
            insert_assistant_select = "%(assistant_id)s"
            insert_assistant_from = ""

        assistant_id = get_assistant_id(payload["assistant_id"])
        payload["assistant_id"] = assistant_id

        if thread_id:
            thread_filter_clause, thread_filter_params = _build_filter_query(
                filters=filters,
                table_alias="thread",
                start=len(filter_params),
            )
            filter_params.update(thread_filter_params)

            authorized_thread_cte = f"""
{authorized_assistant_cte}authorized_thread AS (
    SELECT thread.thread_id
    FROM thread
    WHERE thread.thread_id = %(thread_id)s
    {thread_filter_clause}
),
"""
            insert_select = "authorized_thread.thread_id"
            insert_from = (
                "CROSS JOIN authorized_thread"
                if insert_assistant_from
                else "FROM authorized_thread"
            )
        else:
            # no thread_id => no separate thread filter needed
            authorized_thread_cte = authorized_assistant_cte
            insert_select = "%(thread_id)s"
            insert_from = ""

        query = f"""
{authorized_thread_cte}inserted_cron AS (
    INSERT INTO cron (
        cron_id, assistant_id, thread_id, user_id, end_time,
        schedule, payload, next_run_date, metadata
    )
    SELECT
        %(cron_id)s, {insert_assistant_select}, {insert_select},
        %(user_id)s, %(end_time)s,
        %(schedule)s, %(payload)s, %(next_run_date)s, %(metadata)s
    {insert_assistant_from}
    {insert_from}
    ON CONFLICT (cron_id) DO NOTHING
    RETURNING *
)
SELECT c.* 
FROM inserted_cron c
UNION ALL
SELECT c.*
FROM cron c
WHERE c.cron_id = %(cron_id)s
{filter_clause if filter_params else ""}
LIMIT 1
        """

        params = {
            "cron_id": cron_id,
            "assistant_id": assistant_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "end_time": end_time,
            "schedule": schedule,
            "payload": Jsonb(payload),
            "next_run_date": next_cron_date(schedule, datetime.now(UTC)),
            "metadata": Jsonb(metadata),
        }
        if filter_params:
            params.update(filter_params)
        cur = await conn.execute(query, params, binary=True)
        results = [row async for row in cur]

        if not results:
            raise HTTPException(
                status_code=404, detail="Thread not found or not authorized"
            )

        async def consume():
            for row in results:
                yield {**row, "payload": await ajson_loads(row["payload"])}

        return consume()

    @staticmethod
    async def delete(
        conn: AsyncConnection[DictRow],
        cron_id: UUID,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[UUID]:
        """Delete a cron by ID."""
        filters = await Crons.handle_event(
            ctx,
            "delete",
            Auth.types.CronsDelete(cron_id=cron_id),
        )

        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias="cron",
        )

        query = """DELETE FROM cron WHERE cron.cron_id = %(cron_id)s"""

        params = {"cron_id": cron_id}
        if filter_params:
            query += filter_clause
            params.update(filter_params)

        query += " RETURNING cron_id"
        cur = await conn.execute(query, params, binary=True)

        return (row["cron_id"] async for row in cur)

    @staticmethod
    async def next(
        conn: AsyncConnection[DictRow],
    ) -> AsyncIterator[Cron]:
        """Get the next cron job to run."""
        # Internal API. Needs on auth.
        query = """select *, now() as now from cron
                where (end_time is null or end_time >= now())
                and next_run_date <= now()
                for no key update skip locked"""

        async with conn.transaction():
            async with await conn.execute(
                query,
                binary=True,
            ) as crons:
                async for row in crons:
                    yield {**row, "payload": await ajson_loads(row["payload"])}

    @staticmethod
    async def set_next_run_date(
        conn: AsyncConnection[DictRow],
        cron_id: UUID,
        next_run_date: datetime,
    ) -> None:
        """Update next run date for a cron job."""
        # Internal API. No auth needed.
        query = "UPDATE cron SET next_run_date = %(next_run_date)s WHERE cron_id = %(cron_id)s"
        params = {"next_run_date": next_run_date, "cron_id": cron_id}
        await conn.execute(query, params)

    @staticmethod
    async def search(
        conn: AsyncConnection[DictRow],
        *,
        assistant_id: UUID | None,
        thread_id: UUID | None,
        limit: int,
        offset: int,
        ctx: Auth.types.BaseAuthContext | None = None,
    ) -> AsyncIterator[Cron]:
        """Search all cron jobs"""
        filters = await Crons.handle_event(
            ctx,
            "search",
            Auth.types.CronsSearch(
                assistant_id=assistant_id,
                thread_id=thread_id,
                limit=limit,
                offset=offset,
            ),
        )

        table_aliases = ("cron", "thread") if thread_id else ("cron",)
        filter_clause, filter_params = _build_filter_query(
            filters=filters,
            table_alias=table_aliases,
        )
        threads_join = (
            "JOIN thread USING (thread_id)" if (thread_id and filter_params) else ""
        )

        # Construct base query with joins
        query = f"SELECT cron.* FROM cron {threads_join} WHERE 1=1"
        params: dict[str, Any] = {}

        if thread_id:
            query += " AND cron.thread_id = %(thread_id)s"
            params["thread_id"] = thread_id
        if assistant_id:
            query += " AND cron.assistant_id = %(assistant_id)s"
            params["assistant_id"] = assistant_id

        if filter_params:
            query += filter_clause
            params.update(filter_params)

        query += " ORDER BY cron.created_at DESC LIMIT %(limit)s OFFSET %(offset)s"
        params.update(
            {
                "limit": limit,
                "offset": offset,
            }
        )

        cur = await conn.execute(query, params, binary=True)
        result = [row async for row in cur]
        all_results = []
        async for r in await conn.execute("SELECT * FROM cron WHERE 1 = 1"):
            all_results.append(r)

        async def consume():
            for row in result:
                yield row

        return consume()


async def cancel_run(
    thread_id: UUID, run_id: UUID, ctx: Auth.types.BaseAuthContext | None = None
) -> None:
    async with connect() as conn:
        await Runs.cancel(conn, [run_id], thread_id=thread_id, ctx=ctx)


async def listen_for_cancellation(
    pubsub: StreamHandler, run_id: UUID, done: ValueEvent
):
    try:
        await pubsub.subscribe(CHANNEL_RUN_CONTROL.format(run_id))
        if start_value := await get_redis().get(STRING_RUN_CONTROL.format(run_id)):
            if start_value == b"rollback":
                done.set(UserRollback())
            elif start_value == b"interrupt":
                done.set(UserInterrupt())
        while True:
            event = await pubsub.listen()
            if event is None:
                break
            if event["type"] != "message":
                continue
            payload = event["data"].decode()
            if payload == "rollback":
                done.set(UserRollback())
            elif payload == "interrupt":
                done.set(UserInterrupt())
    except Exception as exc:
        logger.exception("listen_for_cancellation failed", exc_info=exc)
        done.set(RetryableException("listen_for_cancellation failed"))
        raise


async def heartbeat(run_id: UUID):
    """Heartbeat to keep run from getting sweeped back to pending."""
    redis = get_redis()
    while True:
        await asyncio.sleep(BG_JOB_HEARTBEAT / 2)
        try:
            await redis.set(STRING_RUN_RUNNING.format(run_id), "1", ex=BG_JOB_HEARTBEAT)
        except Exception as exc:
            logger.exception("Heartbeat iterationfailed", exc_info=exc)


async def wake_up_worker(delay: float = 0) -> None:
    if delay:
        await asyncio.sleep(delay)
    await get_redis().lpush(LIST_RUN_QUEUE, [1])


def _build_filter_query(
    *,
    filters: Auth.types.FilterType | None,
    metadata_field: str = "metadata",
    table_alias: str | tuple[str, ...] | None = None,
    prefix: str = " AND ",
    start: int = 0,
) -> tuple[str, dict]:
    if not filters:
        return "", {}

    conditions = []
    params = {}
    aliases = (
        (table_alias,)
        if table_alias is None or isinstance(table_alias, str)
        else table_alias
    )
    for i, (key, value) in enumerate(filters.items(), start=start):
        for alias in aliases:
            param_key = f"filter_{i}"
            field_path = f"{alias + '.' if alias else ''}{metadata_field}"

            if isinstance(value, dict):
                op = next(iter(value))  # $eq or $contains
                filter_value = value[op]
                if op == "$eq":
                    conditions.append(f"{field_path} @> %({param_key})s::jsonb")
                    params[param_key] = json.dumps({key: filter_value})
                elif op == "$contains":
                    # array contains logic
                    # We'll assume metadata[key] is an array and filter_value must be in it
                    # We'll use a containment check: value in array
                    # One approach: ((metadata->key)::jsonb) @> [filter_value]::jsonb
                    # But we must ensure metadata[key] is array. Let's just guess:
                    conditions.append(
                        f"((({field_path} -> %({param_key}_key)s)::jsonb) @> to_jsonb(%({param_key})s))"
                    )
                    params[param_key] = filter_value
                    params[f"{param_key}_key"] = key
            else:
                conditions.append(f"{field_path} @> %({param_key})s::jsonb")
                params[param_key] = json.dumps({key: value})

        if not conditions:
            return "", {}
    conditions_str = " AND ".join(conditions)
    return prefix + conditions_str, params


__all__ = [
    "Assistants",
    "Crons",
    "Runs",
    "Threads",
]
