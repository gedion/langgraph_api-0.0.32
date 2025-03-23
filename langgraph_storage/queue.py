import asyncio
import concurrent.futures
import functools
from collections.abc import Callable, Coroutine
from contextlib import ExitStack

import structlog
from langgraph.utils.future import chain_future
from langgraph_api.asyncio import AsyncQueue
from langgraph_api.config import (
    BG_JOB_HEARTBEAT,
    BG_JOB_ISOLATED_LOOPS,
    N_JOBS_PER_WORKER,
    STATS_INTERVAL_SECS,
)
from langgraph_api.graph import is_js_graph
from langgraph_api.schema import Run
from langgraph_api.webhook import call_webhook
from langgraph_api.worker import WorkerResult, worker

from langgraph_storage.database import connect, stop_pool
from langgraph_storage.ops import Runs

logger = structlog.stdlib.get_logger(__name__)

WORKERS: set[asyncio.Task] = set()
SHUTDOWN_GRACE_PERIOD_SECS = 5


class BgLoopRunner(asyncio.Runner):
    """
    A runner that runs a loop in a separate thread. It's very important to
    use run the loop always in the same thread, as some objects may be created
    which are bound to the loop's thread.
    """

    executor: concurrent.futures.ThreadPoolExecutor

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix=f"bg-loop-{self.idx}"
        )
        self.executor.submit(self.get_loop).result()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        loop = self.get_loop()
        # cancel all remaining tasks
        for task in asyncio.all_tasks(loop):
            task.cancel()
        # close postgres connection pool
        try:
            if not loop.is_running():
                self.executor.submit(self.run, stop_pool()).result(
                    SHUTDOWN_GRACE_PERIOD_SECS / 2
                )
            else:
                asyncio.run_coroutine_threadsafe(stop_pool(), loop).result(
                    SHUTDOWN_GRACE_PERIOD_SECS / 2
                )
        except TimeoutError:
            pass
        # shutdown
        self.executor.shutdown(wait=False)

    def submit(
        self,
        coro: Coroutine,
        *,
        name: str | None = None,
        callback: Callable[[asyncio.Task], None] | None = None,
    ):
        fut = self.executor.submit(
            self.run,
            coro,
            name=name,
        )
        WORKERS.add(fut)
        fut.add_done_callback(callback)
        return fut

    def run(
        self,
        coro: Coroutine,
        *,
        name: str | None = None,
    ):
        """Run a coroutine inside the embedded event loop.
        Modified from asyncio.Runner.run
        - Removed main thread check (we only use it on bg threads)
        - Added callback and name arguments
        - Added WORKERS set to track tasks
        """

        if asyncio.events._get_running_loop() is not None:
            # fail fast with short traceback
            raise RuntimeError(
                "Runner.run() cannot be called from a running event loop"
            )

        self._lazy_init()

        task = self._loop.create_task(coro, name=name)

        try:
            return self._loop.run_until_complete(task)
        except asyncio.exceptions.CancelledError:
            raise  # CancelledError


async def sweep_loop():
    while True:
        await asyncio.sleep(BG_JOB_HEARTBEAT)
        try:
            async with connect() as conn:
                run_ids = await Runs.sweep(conn)
                logger.info("Sweeped runs", run_ids=run_ids)
        except Exception as exc:
            logger.exception("Sweep iteration failed", exc_info=exc)


async def stats_loop():
    while True:
        await asyncio.sleep(STATS_INTERVAL_SECS)
        # worker stats
        active = len(WORKERS)
        await logger.ainfo(
            "Worker stats",
            max=N_JOBS_PER_WORKER,
            available=N_JOBS_PER_WORKER - active,
            active=active,
        )
        # queue stats
        try:
            async with connect() as conn:
                await Runs.stats(conn)
        except Exception as exc:
            logger.exception("Stats iteration failed", exc_info=exc)


async def queue():
    concurrency = N_JOBS_PER_WORKER
    loop = asyncio.get_running_loop()
    # when not using isolated event loops the queue is used to limit concurrency
    # where every task checks out a token before running, and returns it when done
    runners = AsyncQueue[BgLoopRunner](concurrency)
    with ExitStack() as stack:
        WEBHOOKS: set[asyncio.Task] = set()
        if BG_JOB_ISOLATED_LOOPS:
            executor = stack.enter_context(concurrent.futures.ThreadPoolExecutor())
            RUNNERS = {
                stack.enter_context(BgLoopRunner(idx)) for idx in range(concurrency)
            }
            for r in RUNNERS:
                runners.put_nowait(r)
                r.get_loop().set_default_executor(executor)
        else:
            # when not using isolated event loops the queue items are used as
            # tokens to limit concurrency, so we just create empty objects
            for _ in range(concurrency):
                runners.put_nowait(object())

        def cleanup(task: concurrent.futures.Future, runner: BgLoopRunner):
            WORKERS.remove(task)
            runners.put_nowait(runner)

            try:
                if task.cancelled():
                    return
                if exc := task.exception():
                    if not isinstance(exc, asyncio.CancelledError):
                        logger.exception(
                            f"Background worker failed for task {task}",
                            exc_info=exc,
                        )
                    return
                result: WorkerResult | None = task.result()
                if result and result["webhook"]:
                    hook_task = loop.create_task(
                        call_webhook(result),
                        name=f"webhook-{result['run']['run_id']}",
                    )
                    WEBHOOKS.add(hook_task)
                    hook_task.add_done_callback(WEBHOOKS.remove)
            except Exception as exc:
                logger.exception("Background worker cleanup failed", exc_info=exc)

        await logger.ainfo(f"Starting {concurrency} background workers")
        sweep_task = asyncio.create_task(sweep_loop())
        stats_task = asyncio.create_task(stats_loop())

        try:
            while True:
                # wait for an available runner to respect concurrency
                await runners.wait()
                try:
                    # skip the wait, if 1st time, or got a run last time
                    try:
                        wait = run is None  # noqa: F821
                    except UnboundLocalError:
                        wait = False
                    # try to get a run, handle it
                    run: Run | None = None
                    async for run, attempt in Runs.next(
                        wait=wait, limit=runners.qsize()
                    ):
                        # we know a runner exists, so get it
                        runner = runners.get_nowait()

                        graph_id = (
                            run["kwargs"]
                            .get("config", {})
                            .get("configurable", {})
                            .get("graph_id")
                        )

                        if not BG_JOB_ISOLATED_LOOPS or (
                            graph_id and is_js_graph(graph_id)
                        ):
                            task = asyncio.create_task(
                                worker(run, attempt, loop),
                                name=f"run-{run['run_id']}-attempt-{attempt}",
                            )
                            task.add_done_callback(
                                functools.partial(cleanup, runner=runner)
                            )
                            WORKERS.add(task)
                        else:
                            runner.submit(
                                worker(run, attempt, loop),
                                name=f"run-{run['run_id']}-attempt-{attempt}",
                                callback=functools.partial(cleanup, runner=runner),
                            )
                except Exception as exc:
                    # keep trying to run the scheduler indefinitely
                    logger.exception("Background worker scheduler failed", exc_info=exc)
        finally:
            logger.info("Shutting down background workers")
            sweep_task.cancel()
            stats_task.cancel()
            for task in list(WEBHOOKS):
                task.cancel()
            if BG_JOB_ISOLATED_LOOPS:
                futs = [
                    *(chain_future(f, loop.create_future()) for f in WORKERS),
                    *(chain_future(f, loop.create_future()) for f in WEBHOOKS),
                ]
            else:
                futs = [*WORKERS, *WEBHOOKS]
            await asyncio.wait_for(
                asyncio.gather(
                    sweep_task,
                    stats_task,
                    *futs,
                    return_exceptions=True,
                ),
                SHUTDOWN_GRACE_PERIOD_SECS,
            )
