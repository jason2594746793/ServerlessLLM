
import asyncio
import json
import uuid
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
            return

        # --- OPTIMIZATION: Model Grouping (Fusion) ---
        # Sort pending tasks by model name to minimize thrashing.
        pending_tasks.sort(key=lambda t: t.body.get("model", ""))

        # # --- OPTIMIZATION (Option 1): Async with Semaphore ---
        # # Pros: Smoother flow, max resource utilization. Cons: Complex to tune limit.
        buffer_limit = getattr(self.router.config, 'max_buffer_size', 10)
        semaphore = asyncio.Semaphore(buffer_limit)
        
        async def _sem_execute(task):
            async with semaphore:
                await self._execute_task(task)
        
        logger.info(f"Processing {len(pending_tasks)} tasks with semaphore (limit: {buffer_limit})")
        await asyncio.gather(*(_sem_execute(task) for task in pending_tasks))

        # --- OPTIMIZATION (Option 2): Chunked Execution ---
        # Pros: Simple, guarantees no buffer overflow. Cons: Stop-and-wait behavior reduces throughput.
        # buffer_limit = getattr(self.router.config, 'max_buffer_size', 10)
        # chunk_size = max(1, buffer_limit)
        # logger.info(f"Processing {len(pending_tasks)} tasks in chunks of {chunk_size}")

        # for i in range(0, len(pending_tasks), chunk_size):
        #     chunk = pending_tasks[i : i + chunk_size]
        #     execution_futures = [self._execute_task(task) for task in chunk]
        #     await asyncio.gather(*execution_futures)

        # --- BASELINE (Option 3): Synchronous Execution ---
        # Pros: Zero overhead, deterministic. Cons: Extremely slow (no parallelism).
        # for task in pending_tasks:
        #     await self._execute_task(task)

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

