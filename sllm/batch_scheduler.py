
import asyncio
import json
import uuid
import re
from typing import Optional, List, Dict
from datetime import datetime
from sllm.database import Database, BatchTask, BatchJob
from sllm.logger import init_logger
from sllm.router import Router

logger = init_logger(__name__)

class BatchScheduler:
    def __init__(self, database: Database, router: Router):
        self.database = database
        self.router = router
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None
        self.autoscaler = None  # Will be set by API Gateway

        # Track active batches to prevent premature scale-down
        self._active_batch_jobs: set = set()

        # Prevent concurrent processing of same batch
        self._processing_batches: set = set()
        self._batch_lock = asyncio.Lock()

        # Runtime-configurable scheduling strategy
        self.strategy = "semaphore"  # Options: "sync", "chunked", "semaphore"
        # Read Router's buffer capacity to use as default and ceiling
        self._router_buffer_size = getattr(self.router.config, 'max_buffer_size', 100)
        self.buffer_limit = self._router_buffer_size
        self.enable_model_grouping = True

        # Checkpoint prefetch configuration
        self.storage_manager = None     # Will be set by API Gateway
        self.enable_prefetch = True
        self.prefetch_threshold = 0.8   # Trigger prefetch when 80% of group done

        # Per-batch concurrency limit (prevents one batch from monopolizing buffer)
        self.max_concurrent_tasks_per_batch = max(10, self._router_buffer_size // 2)

        # Global concurrency limit across ALL batches
        # Ensures total in-flight tasks never exceed router capacity
        self._global_semaphore = asyncio.Semaphore(self._router_buffer_size)

    def set_autoscaler(self, autoscaler):
        """Set the AutoScaler reference for proactive scaling."""
        self.autoscaler = autoscaler
        logger.info("BatchScheduler connected to AutoScaler")

    def set_storage_manager(self, storage_manager):
        """Set the StorageManager reference for checkpoint prefetching."""
        self.storage_manager = storage_manager
        logger.info("BatchScheduler connected to StorageManager for prefetch")

    def set_strategy(self, strategy: str, buffer_limit: int = 10, enable_model_grouping: bool = True):
        """Set scheduling strategy at runtime without restarting cluster.

        Args:
            strategy: "sync", "chunked", or "semaphore"
            buffer_limit: Concurrency limit for semaphore/chunked strategies
            enable_model_grouping: Whether to sort tasks by model to avoid thrashing
        """
        valid_strategies = ["sync", "chunked", "semaphore"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

        self.strategy = strategy
        # Cap buffer_limit to Router's max_buffer_size to prevent "buffer full" errors
        effective_limit = min(buffer_limit, self._router_buffer_size)
        if effective_limit != buffer_limit:
            logger.info(f"buffer_limit {buffer_limit} capped to Router max_buffer_size {self._router_buffer_size}")
        self.buffer_limit = effective_limit
        self.enable_model_grouping = enable_model_grouping
        logger.info(f"Strategy updated: {strategy}, buffer_limit={self.buffer_limit}, model_grouping={enable_model_grouping}")

    async def handle_batch_with_deadline(self, batch_id: str, num_tasks: int, completion_window: str):
        """Handle a new batch job with a deadline-driven cost model.

        Args:
            batch_id: Batch job ID
            num_tasks: Total number of tasks
            completion_window: Deadline string (e.g., "1h", "24h", "30m")
        """
        try:
            # Parse completion_window to seconds
            window_seconds = self._parse_completion_window(completion_window)
            if window_seconds is None:
                logger.warning(f"Invalid completion_window format: {completion_window}")
                return

            # Get first task to infer model
            tasks = self.database.get_batch_tasks(batch_id)
            if not tasks:
                logger.warning(f"No tasks found for batch {batch_id}")
                return

            model_name = tasks[0].body.get("model")
            if not model_name:
                logger.warning(f"No model specified in batch {batch_id}")
                return

            # Conservative cost model parameters
            # Based on empirical data from Experiment 4 (to be collected)
            # Using worst-case estimates for now
            max_tokens = tasks[0].body.get("max_tokens", 512)  # Default max_tokens
            throughput_per_worker = 2.0  # Conservative: 2 req/s per worker (will be refined)

            # Calculate workers needed
            estimated_time_per_task = max_tokens / (throughput_per_worker * 100)  # Rough estimate
            total_estimated_time = num_tasks * estimated_time_per_task
            workers_needed = max(1, int(total_estimated_time / window_seconds) + 1)

            logger.info(
                f"Batch {batch_id}: {num_tasks} tasks, window={completion_window} ({window_seconds}s), "
                f"estimated_time={total_estimated_time:.1f}s, workers_needed={workers_needed}"
            )

            # Request proactive scaling via AutoScaler
            if self.autoscaler and workers_needed > 1:
                deployment_id = f"{model_name}:vllm"
                deployment = self.database.get_deployment_by_id(deployment_id)

                if not deployment:
                    # Auto-create deployment
                    logger.info(f"Auto-creating deployment for {model_name}")
                    self.database.create_deployment(
                        model_name=model_name,
                        backend="vllm",
                        min_replicas=0,
                        max_replicas=workers_needed,
                    )
                else:
                    # Update max_replicas if needed
                    if workers_needed > deployment.max_replicas:
                        logger.info(f"Increasing max_replicas for {deployment_id} to {workers_needed}")
                        # Note: Need to add update_max_replicas method to Database

                # Set desired replicas
                logger.info(f"Proactively scaling {deployment_id} to {workers_needed} workers")
                self.database.update_desired_replicas(deployment_id, workers_needed)

                # Register batch as active to prevent premature scale-down
                self.autoscaler.register_active_batch(deployment_id, batch_id)

            # Mark batch as active locally
            self._active_batch_jobs.add(batch_id)

        except Exception as e:
            logger.error(f"Error handling batch deadline: {e}", exc_info=True)

    def _parse_completion_window(self, window: str) -> Optional[int]:
        """Parse completion_window string to seconds.

        Supports formats: "30m", "1h", "24h", "2d"
        """
        match = re.match(r'^(\d+)([mhd])$', window.lower())
        if not match:
            return None

        value, unit = int(match.group(1)), match.group(2)

        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400

        return None

    async def start(self):
        """Start the scheduler loop."""
        if self.running:
            return
        self.running = True
        self._loop_task = asyncio.create_task(self._schedule_loop())
        logger.info("BatchScheduler started")

    async def stop(self):
        """Stop the scheduler gracefully, allowing in-flight batches to drain."""
        logger.info("Stopping BatchScheduler...")
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("BatchScheduler stopped")

    async def _schedule_loop(self):
        """Main scheduling loop.

        Launches batch processing tasks concurrently. Does NOT block on
        completion — new batches are picked up every iteration while
        existing ones continue running in the background.
        """
        background_tasks: set = set()

        while self.running:
            try:
                pending_batch_ids = self.database.get_pending_batch_ids()

                for batch_id in pending_batch_ids:
                    async with self._batch_lock:
                        if batch_id not in self._processing_batches:
                            self._processing_batches.add(batch_id)
                            task = asyncio.create_task(self._process_batch_safe(batch_id))
                            background_tasks.add(task)
                            task.add_done_callback(background_tasks.discard)

                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}", exc_info=True)
                await asyncio.sleep(5)

        # On shutdown, wait for in-flight batches to finish
        if background_tasks:
            logger.info(f"Waiting for {len(background_tasks)} in-flight batches to finish...")
            await asyncio.gather(*background_tasks, return_exceptions=True)

    async def _process_batch_safe(self, batch_id: str):
        """Wrapper to ensure batch is removed from processing set."""
        try:
            await self._process_batch(batch_id)
        finally:
            async with self._batch_lock:
                self._processing_batches.discard(batch_id)

    async def _process_batch(self, batch_id: str):
        tasks = self.database.get_batch_tasks(batch_id)

        # Base version: Ignore DAG. Just check if task is pending.
        pending_tasks = [t for t in tasks if t.status == 'pending']

        if not pending_tasks:
            # Check if all tasks completed, if so mark batch done
            if all(t.status in ('completed', 'failed') for t in tasks):
                 self.database.update_batch_job_status(batch_id, 'completed')
                 # Remove from active batches to allow scale-down
                 if batch_id in self._active_batch_jobs:
                     self._active_batch_jobs.remove(batch_id)
                     logger.info(f"Batch {batch_id} completed, allowing scale-down")

                     # Notify AutoScaler to allow scale-down
                     if self.autoscaler and tasks:
                         model_name = tasks[0].body.get("model")
                         if model_name:
                             deployment_id = f"{model_name}:vllm"
                             self.autoscaler.unregister_active_batch(deployment_id, batch_id)
            return

        # Mark batch as in_progress to prevent reprocessing
        self.database.update_batch_job_status(batch_id, 'in_progress')

        # --- OPTIMIZATION: Model Grouping (Fusion) ---
        # Sort pending tasks by model name to minimize thrashing.
        if self.enable_model_grouping:
            pending_tasks.sort(key=lambda t: t.body.get("model", ""))
            logger.info(f"Model grouping enabled: tasks sorted by model")
        else:
            logger.info(f"Model grouping disabled: preserving original task order")

        # === Decide execution path ===
        # When model grouping is enabled, ALWAYS use group-based execution
        # so that prefetch vs no-prefetch is a fair comparison.
        use_prefetch = (
            self.enable_prefetch
            and self.enable_model_grouping
            and self.storage_manager is not None
        )

        if self.enable_model_grouping:
            # Group-based execution (sequential groups, concurrent within group)
            await self._process_with_prefetch(pending_tasks, do_prefetch=use_prefetch)
        else:
            await self._process_without_prefetch(pending_tasks)

    # --------------------------------------------------------------------- #
    #  Non-prefetch path (original logic, unchanged)                         #
    # --------------------------------------------------------------------- #

    async def _process_without_prefetch(self, pending_tasks: List[BatchTask]):
        """Execute tasks using the selected strategy without prefetch."""
        if self.strategy == "sync":
            logger.info(f"Processing {len(pending_tasks)} tasks with SYNC strategy")
            for task in pending_tasks:
                await self._execute_task(task)

        elif self.strategy == "chunked":
            chunk_size = max(1, self.max_concurrent_tasks_per_batch)
            logger.info(f"Processing {len(pending_tasks)} tasks with CHUNKED strategy (chunk_size: {chunk_size})")
            for i in range(0, len(pending_tasks), chunk_size):
                chunk = pending_tasks[i : i + chunk_size]
                execution_futures = [self._execute_task(task) for task in chunk]
                await asyncio.gather(*execution_futures)

        else:  # semaphore (default)
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks_per_batch)

            async def _sem_execute(task):
                async with semaphore:
                    await self._execute_task(task)

            logger.info(f"Processing {len(pending_tasks)} tasks with SEMAPHORE strategy (limit: {self.max_concurrent_tasks_per_batch})")
            await asyncio.gather(*(_sem_execute(task) for task in pending_tasks))

    # --------------------------------------------------------------------- #
    #  Prefetch path — group-aware execution with checkpoint pre-loading     #
    # --------------------------------------------------------------------- #

    async def _process_with_prefetch(self, pending_tasks: List[BatchTask], do_prefetch: bool = True):
        """Execute tasks grouped by model, optionally prefetching the next model's
        checkpoint into CPU memory.

        When do_prefetch=False, still uses group-based sequential execution
        but without triggering any prefetch operations.

        If prefetch_threshold == 0.0:
            Eagerly read ALL subsequent models into CPU memory right at the start.
        If prefetch_threshold > 0.0:
            Wait until threshold% of current tasks are done before fetching the next one.
        """
        groups = self._extract_model_groups(pending_tasks)
        logger.info(
            f"[PREFETCH] Processing {len(pending_tasks)} tasks "
            f"(prefetch={'ON' if do_prefetch else 'OFF'}, threshold={self.prefetch_threshold}) in {len(groups)} "
            f"model groups: {[g[0] for g in groups]}"
        )

        if do_prefetch and self.prefetch_threshold == 0.0:
            # === EAGER PREFETCH MODE ===
            # Kick off background prefetch for *all* models except the first one right away
            if self.storage_manager and len(groups) > 1:
                # Use fromkeys to preserve order and avoid duplicates
                subsequent_models = list(dict.fromkeys(g[0] for g in groups[1:]))
                for m in subsequent_models:
                    if m != groups[0][0]:  # Skip the one we are running first
                        logger.info(f"[PREFETCH_ALL] Eagerly prefetching {m} to CPU right now")
                        asyncio.create_task(self.storage_manager.prefetch_to_cpu(m))
            
            # Now run everything sequentially without threshold tracking
            for group_idx, (model_name, group_tasks) in enumerate(groups):
                logger.info(
                    f"Starting group {group_idx+1}/{len(groups)}: "
                    f"{model_name} ({len(group_tasks)} tasks) (Eager Mode)"
                )
                await self._execute_group(group_tasks)
                
        else:
            # === OVERLAPPING PREFETCH MODE or NO-PREFETCH GROUP MODE ===
            for group_idx, (model_name, group_tasks) in enumerate(groups):
                next_model = (
                    groups[group_idx + 1][0]
                    if group_idx + 1 < len(groups)
                    else None
                )

                logger.info(
                    f"[PREFETCH] Starting group {group_idx+1}/{len(groups)}: "
                    f"{model_name} ({len(group_tasks)} tasks)"
                )

                if do_prefetch and next_model and next_model != model_name:
                    await self._execute_group_with_prefetch(
                        group_tasks, next_model, self.prefetch_threshold
                    )
                else:
                    # Last group, same model, or prefetch disabled
                    await self._execute_group(group_tasks)

    async def _execute_group_with_prefetch(
        self,
        tasks: List[BatchTask],
        next_model: str,
        threshold: float,
    ):
        """Execute one model group and trigger prefetch of *next_model* once
        *threshold*% of tasks complete.

        Always uses the semaphore strategy internally so we can track
        individual task completions for the threshold trigger.
        """
        prefetch_at = max(1, int(len(tasks) * threshold))
        prefetch_triggered = False
        completed_count = 0
        lock = asyncio.Lock()

        semaphore = asyncio.Semaphore(self.max_concurrent_tasks_per_batch)

        async def _execute_and_track(task):
            nonlocal completed_count, prefetch_triggered

            async with semaphore:
                await self._execute_task(task)

            async with lock:
                completed_count += 1

                if (
                    not prefetch_triggered
                    and completed_count >= prefetch_at
                    and self.storage_manager
                ):
                    prefetch_triggered = True
                    logger.info(
                        f"[PREFETCH] {completed_count}/{len(tasks)} done "
                        f"(threshold {threshold:.0%}), prefetching {next_model}"
                    )
                    # Fire-and-forget: prefetch runs in background
                    asyncio.create_task(
                        self.storage_manager.prefetch_to_cpu(next_model)
                    )

        await asyncio.gather(*[_execute_and_track(t) for t in tasks])

    async def _execute_group(self, tasks: List[BatchTask]):
        """Execute a model group using the currently selected strategy
        (without prefetch tracking).
        """
        if self.strategy == "sync":
            for task in tasks:
                await self._execute_task(task)
        elif self.strategy == "chunked":
            chunk_size = max(1, self.max_concurrent_tasks_per_batch)
            for i in range(0, len(tasks), chunk_size):
                chunk = tasks[i : i + chunk_size]
                await asyncio.gather(*[self._execute_task(t) for t in chunk])
        else:
            semaphore = asyncio.Semaphore(self.max_concurrent_tasks_per_batch)

            async def _sem(task):
                async with semaphore:
                    await self._execute_task(task)

            await asyncio.gather(*[_sem(t) for t in tasks])

    def _extract_model_groups(self, tasks: List[BatchTask]) -> List[tuple]:
        """Split sorted task list into per-model groups.

        Args:
            tasks: Tasks already sorted by model_name.

        Returns:
            [(model_name, [tasks]), ...] preserving the sorted order.

        Example:
            Input:  [A, A, A, B, B, C]
            Output: [("ModelA", [A,A,A]), ("ModelB", [B,B]), ("ModelC", [C])]
        """
        groups = []
        current_model = None
        current_group = []
        for task in tasks:
            model = task.body.get("model", "")
            if model != current_model:
                if current_group:
                    groups.append((current_model, current_group))
                current_model = model
                current_group = [task]
            else:
                current_group.append(task)
        if current_group:
            groups.append((current_model, current_group))
        return groups

    async def _execute_task(self, task: BatchTask):
        """Execute a single batch task, respecting the global concurrency limit."""
        async with self._global_semaphore:
            await self._execute_task_inner(task)

    async def _execute_task_inner(self, task: BatchTask):
        logger.info(f"Executing task {task.id} (Batch: {task.batch_id})")

        # Record Start Time
        started_at = datetime.utcnow().isoformat()

        try:
            # 1. Prepare Request
            # Convert body/url to what Router expects
            # Router expects: payload, path, deployment_id
            
            # Infer deployment_id from body['model']
            model = task.body.get("model")
            if not model:
               raise ValueError("Model not specific in task body")
            
            # Simple assumption: default backend is vllm
            # Real logic might need to look up deployment registry
            deployment_id = f"{model}:vllm" 
            
            # Check if deployment exists, auto-create if not
            deployment = self.database.get_deployment_by_id(deployment_id)
            if not deployment:
                logger.info(f"Auto-creating deployment for {model}")
                self.database.create_deployment(
                    model_name=model,
                    backend="vllm",
                    min_replicas=0,
                    max_replicas=1,
                ) 
            
            # 2. Submit to Router
            # Router.handle_request returns the *result* (JSON dict)
            max_retries = 3
            retry_delay = 1.0

            for attempt in range(max_retries):
                try:
                    result = await self.router.handle_request(
                        payload=task.body,
                        path=task.url,
                        deployment_id=deployment_id
                    )
                    break
                except Exception as e:
                    if "buffer full" in str(e).lower() and attempt < max_retries - 1:
                        logger.warning(f"Task {task.id} buffer full, retry {attempt+1}/{max_retries} after {retry_delay}s")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise
            
            # 3. Handle Success
            completed_at = datetime.utcnow().isoformat()
            self.database.upsert_batch_task(
                task_id=task.id,
                batch_id=task.batch_id,
                custom_id=task.custom_id,
                method=task.method,
                url=task.url,
                body=task.body,
                status="completed",
                output=result,
                started_at=started_at,
                completed_at=completed_at
            )
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            completed_at = datetime.utcnow().isoformat()
            self.database.upsert_batch_task(
                task_id=task.id,
                batch_id=task.batch_id,
                custom_id=task.custom_id,
                method=task.method,
                url=task.url,
                body=task.body,
                status="failed",
                output={"error": str(e)},
                started_at=started_at,
                completed_at=completed_at
            )

