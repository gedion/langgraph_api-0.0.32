import asyncio
import builtins
from collections.abc import AsyncIterator, Iterator
from hashlib import md5
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.channels.base import BaseChannel
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.constants import TASKS
from langgraph.errors import EmptyChannelError
from langgraph_api import config as api_config
from langgraph_api.asyncio import aclosing_aiter
from langgraph_api.schema import MetadataInput
from langgraph_api.serde import Fragment, Serializer, ajson_loads, json_loads
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

SELECT_SQL = f"""
select
    thread_id,
    checkpoint,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    metadata,
    (
        select array_agg(array[bl.channel::bytea, bl.type::bytea, bl.blob])
        from jsonb_each_text(checkpoint -> 'channel_versions')
        inner join checkpoint_blobs bl
            on bl.thread_id = checkpoints.thread_id
            and bl.checkpoint_ns = checkpoints.checkpoint_ns
            and bl.channel = jsonb_each_text.key
            and bl.version = jsonb_each_text.value
    ) as channel_values,
    (
        select
        array_agg(array[cw.task_id::text::bytea, cw.channel::bytea, cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.checkpoint_id
    ) as pending_writes,
    (
        select array_agg(array[cw.type::bytea, cw.blob] order by cw.task_id, cw.idx)
        from checkpoint_writes cw
        where cw.thread_id = checkpoints.thread_id
            and cw.checkpoint_ns = checkpoints.checkpoint_ns
            and cw.checkpoint_id = checkpoints.parent_checkpoint_id
            and cw.channel = '{TASKS}'
    ) as pending_sends
from checkpoints """


class Checkpointer(BaseCheckpointSaver):
    latest_iter: AsyncIterator[CheckpointTuple] | None

    def __init__(
        self,
        conn: AsyncConnection,
        latest: AsyncIterator[CheckpointTuple] | None = None,
    ) -> None:
        serde = Serializer()
        if api_config.LANGGRAPH_AES_KEY:
            serde = EncryptedSerializer.from_pycryptodome_aes(
                serde, key=api_config.LANGGRAPH_AES_KEY
            )
        super().__init__(serde=serde)
        self.conn = conn
        self.loop = asyncio.get_running_loop()
        self.latest_iter = latest
        self.latest_tuple: CheckpointTuple | None = None

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        where, args = self._search_where(config, filter, before)
        query = SELECT_SQL + where + " ORDER BY checkpoint_id DESC"
        if limit:
            query += f" LIMIT {limit}"
        async with self.conn.cursor(binary=True) as cur:
            async for value in await cur.execute(query, args, binary=True):
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["checkpoint_id"],
                        }
                    },
                    await asyncio.to_thread(
                        self._load_checkpoint,
                        value["checkpoint"],
                        value["channel_values"],
                        value["pending_sends"],
                    ),
                    await ajson_loads(value["metadata"]),
                    {
                        "configurable": {
                            "thread_id": value["thread_id"],
                            "checkpoint_ns": value["checkpoint_ns"],
                            "checkpoint_id": value["parent_checkpoint_id"],
                        }
                    }
                    if value["parent_checkpoint_id"]
                    else None,
                    await asyncio.to_thread(self._load_writes, value["pending_writes"]),
                )

    async def aget_iter(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if checkpoint_id:
            args = (thread_id, checkpoint_ns, checkpoint_id)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s"
        else:
            args = (thread_id, checkpoint_ns)
            where = "WHERE thread_id = %s AND checkpoint_ns = %s ORDER BY checkpoint_id DESC LIMIT 1"

        cur = await self.conn.execute(
            SELECT_SQL + where,
            args,
            binary=True,
        )

        return (
            CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["checkpoint_id"],
                    }
                },
                await asyncio.to_thread(
                    self._load_checkpoint,
                    value["checkpoint"],
                    value["channel_values"],
                    value["pending_sends"],
                ),
                await ajson_loads(value["metadata"]),
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": value["checkpoint_ns"],
                        "checkpoint_id": value["parent_checkpoint_id"],
                    }
                }
                if value["parent_checkpoint_id"]
                else None,
                await asyncio.to_thread(self._load_writes, value["pending_writes"]),
            )
            async for value in aclosing_aiter(cur)
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        if self.latest_iter is not None:
            try:
                latest_tuple = await anext(self.latest_iter, None)
                if not latest_tuple:
                    return None
                elif latest_tuple.config["configurable"]["thread_id"] == config[
                    "configurable"
                ]["thread_id"] and latest_tuple.config["configurable"][
                    "checkpoint_ns"
                ] == config["configurable"].get("checkpoint_ns", ""):
                    return latest_tuple
            finally:
                self.latest_iter = None

        return await anext(await self.aget_iter(config), None)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        next_versions: dict[str, str],
    ) -> RunnableConfig:
        configurable = config["configurable"].copy()
        run_id = configurable.pop("run_id", None)
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        checkpoint_id = configurable.pop("checkpoint_id", None)
        copy = checkpoint.copy()
        copy.pop("pending_sends", None)  # saved in aput_writes
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        blobs = await asyncio.to_thread(
            self._dump_blobs,
            thread_id,
            checkpoint_ns,
            copy.pop("channel_values"),
            next_versions,
        )

        async with self.conn.cursor(binary=True) as cur:
            await cur.executemany(
                """INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO NOTHING""",
                blobs,
            )
            await cur.execute(
                """
                INSERT INTO checkpoints (run_id, thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                DO UPDATE SET
                    checkpoint = EXCLUDED.checkpoint,
                    metadata = EXCLUDED.metadata;""",
                (
                    run_id,
                    thread_id,
                    checkpoint_ns,
                    checkpoint["id"],
                    checkpoint_id,
                    Jsonb(copy),
                    # Merging `configurable` and `metadata` will persist graph_id,
                    # assistant_id, and all assistant and run configurable fields
                    # to the checkpoint metadata.
                    Jsonb(
                        {
                            **{
                                k: v
                                for k, v in configurable.items()
                                if not k.startswith("__")
                            },
                            **config.get("metadata", {}),
                            **metadata,
                        }
                    ),
                ),
            )
        return next_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        params = await asyncio.to_thread(
            self._dump_writes,
            config["configurable"]["thread_id"],
            config["configurable"]["checkpoint_ns"],
            config["configurable"]["checkpoint_id"],
            task_id,
            writes,
        )
        async with self.conn.cursor(binary=True) as cur:
            await cur.executemany(
                """INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx) DO NOTHING""",
                params,
            )

    def get_next_version(self, current: str | None, channel: BaseChannel) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        try:
            next_h = md5(self.serde.dumps_typed(channel.checkpoint())[1]).hexdigest()
        except EmptyChannelError:
            next_h = ""
        return f"{next_v:032}.{next_h}"

    def _load_checkpoint(
        self,
        checkpoint_f: Fragment,
        blob_values: list[tuple[bytes, bytes, bytes]],
        pending_sends: list[tuple[bytes, bytes]],
    ) -> Checkpoint:
        checkpoint = json_loads(checkpoint_f)
        return {
            **checkpoint,
            "pending_sends": [
                self.serde.loads_typed((c.decode(), b)) for c, b in pending_sends or []
            ],
            "channel_values": self._load_blobs(blob_values),
        }

    def _load_blobs(
        self, blob_values: list[tuple[bytes, bytes, bytes]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        return {
            k.decode(): self.serde.loads_typed((t.decode(), v))
            for k, t, v in blob_values
            if t.decode() != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: dict[str, str],
    ) -> list[tuple[str, str, str, str, bytes]]:
        if not versions:
            return []

        return [
            (
                thread_id,
                checkpoint_ns,
                k,
                ver,
                *(
                    self.serde.dumps_typed(values[k])
                    if k in values
                    else ("empty", None)
                ),
            )
            for k, ver in versions.items()
        ]

    def _load_writes(
        self, writes: list[tuple[bytes, bytes, bytes, bytes]]
    ) -> list[tuple[str, str, Any]]:
        return (
            [
                (
                    tid.decode(),
                    channel.decode(),
                    self.serde.loads_typed((t.decode(), v)),
                )
                for tid, channel, t, v in writes
            ]
            if writes
            else []
        )

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        writes: list[tuple[str, Any]],
    ) -> list[tuple[str, str, str, int, str, str, bytes]]:
        return [
            (
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
                channel,
                *self.serde.dumps_typed(value),
            )
            for idx, (channel, value) in enumerate(writes)
        ]

    def _search_where(
        self,
        config: RunnableConfig | None,
        filter: MetadataInput,
        before: RunnableConfig | None = None,
    ) -> tuple[str, list[Any]]:
        """Return WHERE clause predicates for alist() given config, filter, cursor.

        This method returns a tuple of a string and a tuple of values. The string
        is the parametered WHERE clause predicate (including the WHERE keyword):
        "WHERE column1 = $1 AND column2 IS $2". The list of values contains the
        values for each of the corresponding parameters.
        """
        wheres = []
        param_values = []

        # construct predicate for config filter
        if config:
            wheres.append("thread_id = %s ")
            param_values.append(config["configurable"]["thread_id"])
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                wheres.append("checkpoint_ns = %s ")
                param_values.append(checkpoint_ns)

        # construct predicate for metadata filter
        if filter:
            wheres.append("metadata @> %s ")
            param_values.append(Jsonb(filter))

        # construct predicate for `before`
        if before is not None:
            wheres.append("checkpoint_id < %s ")
            param_values.append(before["configurable"]["checkpoint_id"])

        return (
            "WHERE " + " AND ".join(wheres) if wheres else "",
            param_values,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from the Postgres database based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (Optional[RunnableConfig]): Base configuration for filtering checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): Maximum number of checkpoints to return.

        Yields:
            Iterator[CheckpointTuple]: An iterator of matching checkpoint tuples.
        """
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_), self.loop
                ).result()
            except StopAsyncIteration:
                break

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from the database.

        This method retrieves a checkpoint tuple from the Postgres database based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and "checkpoint_id" is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self.loop
        ).result()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict[str, str],
    ) -> RunnableConfig:
        """Save a checkpoint to the database.

        This method saves a checkpoint to the Postgres database. The checkpoint is associated
        with the provided config and its parent config (if any).

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self.loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: builtins.list[tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        This method saves intermediate writes associated with a checkpoint to the database.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id), self.loop
        ).result()


__all__ = ["Checkpointer"]
