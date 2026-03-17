#!/usr/bin/env python3
"""
exp_common.py — Shared utilities for all thesis experiments.

Provides:
  - set_strategy(): switch scheduler strategy via admin API
  - submit_batch(): submit tasks and poll until completion
  - extract_task_metrics(): per-task latency from batch response
  - save_result(): save JSON results to disk
"""

import json
import time
import requests
import sys
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8343"

# ─── Models ────────────────────────────────────────────────────────────────
MODEL_SMALL  = "Qwen/Qwen3-0.6B"
MODEL_MEDIUM = "Qwen/Qwen2.5-7B-Instruct"
MODEL_LARGE  = "Qwen/Qwen3-8B"
# Extra‑large model for 32‑B test (requires ~70 GB pinned pool)
MODEL_XLARGE = "Qwen/Qwen1.5-32B"


RESULTS_ROOT = Path(__file__).parent.parent / "results"


def set_strategy(strategy: str = "semaphore",
                 buffer_limit: int = 10,
                 enable_model_grouping: bool = True):
    """Switch scheduler strategy via admin API (no restart needed)."""
    resp = requests.post(f"{API_URL}/admin/set_strategy", json={
        "strategy": strategy,
        "buffer_limit": buffer_limit,
        "enable_model_grouping": enable_model_grouping,
    })
    resp.raise_for_status()
    data = resp.json()
    print(f"  [strategy] {data['strategy']}  buffer={data['buffer_limit']}  "
          f"grouping={'ON' if data['enable_model_grouping'] else 'OFF'}")
    time.sleep(2)  # let the scheduler absorb the change
    return data


def submit_batch(tasks: list, metadata: dict | None = None,
                 completion_window: str | None = None,
                 poll_interval: float = 2.0,
                 quiet: bool = False) -> dict:
    """Submit a batch and block until completed/failed.

    Returns dict with keys:
      batch_id, status, duration, num_tasks, tasks_data, raw
    """
    payload: dict = {"tasks": tasks}
    if metadata:
        payload["metadata"] = metadata
    if completion_window:
        payload.setdefault("metadata", {})["completion_window"] = completion_window
        payload["completion_window"] = completion_window

    resp = requests.post(f"{API_URL}/v1/batches", json=payload)
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    if not quiet:
        print(f"  [batch] submitted {batch_id}  ({len(tasks)} tasks)")

    t0 = time.time()
    last_status = None
    while True:
        resp = requests.get(f"{API_URL}/v1/batches/{batch_id}")
        data = resp.json()
        status = data["status"]

        if status != last_status and not quiet:
            elapsed = time.time() - t0
            print(f"  [{elapsed:6.1f}s] status → {status}")
            last_status = status

        if status == "completed":
            duration = time.time() - t0
            return {
                "batch_id": batch_id,
                "status": "completed",
                "duration": duration,
                "num_tasks": len(tasks),
                "tasks_data": data.get("tasks", []),
                "raw": data,
            }
        if status in ("failed", "cancelled"):
            duration = time.time() - t0
            return {
                "batch_id": batch_id,
                "status": status,
                "duration": duration,
                "num_tasks": len(tasks),
                "tasks_data": data.get("tasks", []),
                "raw": data,
            }
        time.sleep(poll_interval)


def extract_task_metrics(tasks_data: list) -> dict:
    """Extract per-task latencies from batch response.

    Returns dict with:
      latencies: list[float]          — per-task durations (seconds)
      first_latency: float            — longest individual task (likely cold-start)
      avg_latency: float              — mean across all tasks
      avg_subsequent_latency: float   — mean excluding the longest task
      min_start: float                — earliest started_at timestamp
      max_end: float                  — latest completed_at timestamp
      processing_window: float        — max_end - min_start
    """
    latencies = []
    min_start = float("inf")
    max_end = 0.0

    for t in tasks_data:
        if t.get("status") != "completed":
            continue
        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        if not s_str or not e_str:
            continue
        try:
            s_ts = datetime.fromisoformat(s_str.replace("Z", "+00:00")).timestamp()
            e_ts = datetime.fromisoformat(e_str.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        dur = e_ts - s_ts
        latencies.append(dur)
        min_start = min(min_start, s_ts)
        max_end = max(max_end, e_ts)

    if not latencies:
        return {"latencies": [], "first_latency": 0, "avg_latency": 0,
                "avg_subsequent_latency": 0, "min_start": 0, "max_end": 0,
                "processing_window": 0}

    sorted_lat = sorted(latencies)
    first_lat = max(latencies)
    avg_lat = sum(latencies) / len(latencies)
    avg_sub = (sum(sorted_lat[:-1]) / max(1, len(sorted_lat) - 1)
               if len(sorted_lat) > 1 else avg_lat)

    return {
        "latencies": latencies,
        "first_latency": first_lat,
        "avg_latency": avg_lat,
        "avg_subsequent_latency": avg_sub,
        "min_start": min_start,
        "max_end": max_end,
        "processing_window": max_end - min_start,
    }


def save_result(result: dict, filepath: Path):
    """Save result dict as pretty-printed JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [saved] {filepath}")


def make_task(model: str, prompt: str, task_id: str,
              max_tokens: int = 50) -> dict:
    """Create one batch task dict."""
    return {
        "custom_id": task_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
    }
