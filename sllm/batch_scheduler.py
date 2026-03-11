
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

        # Runtime-configurable scheduling strategy
        self.strategy = "semaphore"  # Options: "sync", "chunked", "semaphore"
        # Read Router's buffer capacity to use as default and ceiling
        self._router_buffer_size = getattr(self.router.config, 'max_buffer_size', 10)
        self.buffer_limit = self._router_buffer_size
        self.enable_model_grouping = True

    def set_autoscaler(self, autoscaler):
        """Set the AutoScaler reference for proactive scaling."""
        self.autoscaler = autoscaler
        logger.info("BatchScheduler connected to AutoScaler")

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
        """Stop the scheduler loop."""
        self.running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("BatchScheduler stopped")

    async def _schedule_loop(self):
        """Main scheduling loop."""
        while self.running:
            try:
                # 1. Fetch pending jobs
                # In a real system, we'd query for jobs.
                # Here, we'll scan all jobs and their tasks pending execution.
                # For simplicity in this base version, we just loop through *all* tasks 
                # that are pending and execute them if they have NO dependencies.
                
                # Note: This is inefficient but functional for a prototype.
                # Ideally, we should have specific queries for "ready" tasks.
                
                # We need a new DB method: get_pending_tasks()
                # For now, let's iterate active batches.
                pending_batch_ids = self.database.get_pending_batch_ids()
                
                for batch_id in pending_batch_ids:
                    await self._process_batch(batch_id)

                await asyncio.sleep(1)  # Interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}", exc_info=True)
                await asyncio.sleep(5)

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

        # --- OPTIMIZATION: Model Grouping (Fusion) ---
        # Sort pending tasks by model name to minimize thrashing.
        if self.enable_model_grouping:
            pending_tasks.sort(key=lambda t: t.body.get("model", ""))
            logger.info(f"Model grouping enabled: tasks sorted by model")
        else:
            logger.info(f"Model grouping disabled: preserving original task order")

        # Execute tasks based on selected strategy
        if self.strategy == "sync":
            # --- BASELINE: Synchronous Execution ---
            # Pros: Zero overhead, deterministic. Cons: No parallelism.
            logger.info(f"Processing {len(pending_tasks)} tasks with SYNC strategy")
            for task in pending_tasks:
                await self._execute_task(task)

        elif self.strategy == "chunked":
            # --- Chunked Execution ---
            # Pros: Simple, guarantees no buffer overflow. Cons: Stop-and-wait behavior.
            chunk_size = max(1, self.buffer_limit)
            logger.info(f"Processing {len(pending_tasks)} tasks with CHUNKED strategy (chunk_size: {chunk_size})")
            for i in range(0, len(pending_tasks), chunk_size):
                chunk = pending_tasks[i : i + chunk_size]
                execution_futures = [self._execute_task(task) for task in chunk]
                await asyncio.gather(*execution_futures)

        else:  # semaphore (default)
            # --- Async with Semaphore ---
            # Pros: Smoother flow, max resource utilization. Cons: Complex to tune limit.
            semaphore = asyncio.Semaphore(self.buffer_limit)

            async def _sem_execute(task):
                async with semaphore:
                    await self._execute_task(task)

            logger.info(f"Processing {len(pending_tasks)} tasks with SEMAPHORE strategy (limit: {self.buffer_limit})")
            await asyncio.gather(*(_sem_execute(task) for task in pending_tasks))

    async def _execute_task(self, task: BatchTask):
        logger.info(f"Executing task {task.id} (Batch: {task.batch_id})")
        
        # Record Start Time
        started_at = datetime.utcnow().isoformat()
        
        # Mark as in_progress (optional, but good for visibility)
        # self.database.upsert_batch_task(task.id, task.batch_id, task.custom_id, task.method, task.url, task.body, status='in_progress', started_at=started_at)

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
            result = await self.router.handle_request(
                payload=task.body,
                path=task.url,
                deployment_id=deployment_id
            )
            
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

