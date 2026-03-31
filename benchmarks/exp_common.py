#!/usr/bin/env python3
"""
exp_common.py — Shared utilities for all thesis experiments.

Provides:
  - set_strategy(): switch scheduler strategy via admin API
  - clear_deployments(): robust cold-start reset with verification
  - submit_batch(): submit tasks and poll until completion
  - extract_task_metrics(): per-task latency from batch response
  - make_task(): create a single batch task dict
  - save_result(): persist JSON results to disk
  - get_gpu_count(): query cluster GPU count
  - aggregate_runs(): compute mean/std across multiple runs
"""

import json
import time
import statistics
import requests
import sys
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8343"

# ─── Models ────────────────────────────────────────────────────────────────
MODEL_SMALL  = "Qwen/Qwen3-0.6B"           # ~1.2 GB checkpoint
MODEL_MEDIUM = "Qwen/Qwen2.5-7B-Instruct"  # ~14.3 GB checkpoint
MODEL_LARGE  = "Qwen/Qwen3-8B"             # ~15.4 GB checkpoint

RESULTS_ROOT = Path(__file__).parent / "results"

# ─── Defaults ──────────────────────────────────────────────────────────────
DEFAULT_RUNS = 3
CLEAR_TIMEOUT = 180       # seconds to wait for GPU release
CLEAR_STABILIZE = 15      # seconds to let CUDA context fully clean up
DEFAULT_PROMPT = "Explain the concept of {topic} in two sentences."


# ===========================================================================
#  Strategy Control
# ===========================================================================

def set_strategy(
    strategy: str = "semaphore",
    buffer_limit: int = 10,
    enable_model_grouping: bool = True,
    enable_johnsons_rule: bool = True,
    enable_prefetch: bool = True,
    prefetch_threshold: float = 0.0,
):
    """Set scheduler strategy via admin API (no restart needed).

    Parameters match the /admin/set_strategy endpoint exactly.
    """
    payload = {
        "strategy": strategy,
        "buffer_limit": buffer_limit,
        "enable_model_grouping": enable_model_grouping,
        "enable_johnsons_rule": enable_johnsons_rule,
        "enable_prefetch": enable_prefetch,
        "prefetch_threshold": prefetch_threshold,
    }
    resp = requests.post(f"{API_URL}/admin/set_strategy", json=payload)
    resp.raise_for_status()
    data = resp.json()
    flags = (
        f"grouping={'ON' if data.get('enable_model_grouping') else 'OFF'}  "
        f"johnson={'ON' if data.get('enable_johnsons_rule') else 'OFF'}  "
        f"prefetch={'ON' if data.get('enable_prefetch') else 'OFF'}"
    )
    print(
        f"  [strategy] {data['strategy']}  buffer={data['buffer_limit']}  "
        f"{flags}  threshold={data.get('prefetch_threshold', 'N/A')}"
    )
    time.sleep(2)  # let the scheduler absorb the change
    return data


# ===========================================================================
#  Deployment Lifecycle
# ===========================================================================

def clear_deployments(timeout: float = CLEAR_TIMEOUT,
                      stabilize: float = CLEAR_STABILIZE) -> bool:
    """Delete ALL deployments and verify GPUs are fully released.

    Three-phase cleanup:
      1. Send DELETE for every deployment (including 'deleting' ones).
      2. Poll until: 0 deployments remaining AND all GPUs free.
      3. Wait `stabilize` seconds for CUDA contexts to fully release.

    Returns True if cold state was verified, False on timeout.
    """
    # Phase 1: delete
    try:
        resp = requests.get(f"{API_URL}/v1/models", timeout=10)
        resp.raise_for_status()
        deployments = resp.json().get("data", [])
    except Exception as e:
        print(f"  [clear] WARNING: Could not list deployments: {e}")
        time.sleep(stabilize)
        return False

    if not deployments:
        print("  [clear] No active deployments")
    else:
        for d in deployments:
            dep_id = d["id"]
            try:
                requests.delete(f"{API_URL}/deployments/{dep_id}", timeout=10)
                print(f"  [clear] Requested deletion: {dep_id}")
            except Exception as e:
                print(f"  [clear] WARN: delete {dep_id} failed: {e}")

    # Phase 2: verify
    print(f"  [clear] Verifying GPU release (timeout={timeout:.0f}s)...",
          end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        try:
            # Check remaining deployments
            resp_models = requests.get(f"{API_URL}/v1/models", timeout=5)
            all_deps = resp_models.json().get("data", [])
            # Re-poke any stuck "deleting" deployments
            for d in all_deps:
                try:
                    requests.delete(
                        f"{API_URL}/deployments/{d['id']}", timeout=5
                    )
                except Exception:
                    pass
            # Zombie entries (desired=0, ready=0) are inactive — ignore them
            active = [
                d for d in all_deps
                if d.get("desired_replicas", 0) > 0
                or d.get("ready_replicas", 0) > 0
            ]

            # Check GPU status
            resp_status = requests.get(f"{API_URL}/status", timeout=5)
            status = resp_status.json()
            nodes = status.get("nodes", [])
            total = sum(n.get("total_gpus", 0) for n in nodes)
            free = sum(n.get("available_gpus", 0) for n in nodes)

            if len(active) == 0 and total > 0 and free >= total:
                # Phase 2b: verify no zombie vllm process still holds GPU.
                # Only check when CUDA_VISIBLE_DEVICES is set (shared cluster
                # has other users' processes that would false-positive).
                import subprocess, os
                gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if gpu_ids:
                    try:
                        smi = subprocess.run(
                            ["nvidia-smi",
                             "--query-compute-apps=pid,used_memory",
                             "--format=csv,noheader,nounits",
                             "--id=" + gpu_ids],
                            capture_output=True, text=True, timeout=10
                        )
                        heavy = [
                            line.strip() for line in smi.stdout.splitlines()
                            if line.strip() and int(line.split(",")[1].strip()) > 1024
                        ]
                        if heavy:
                            print("z", end="", flush=True)
                            continue
                    except Exception:
                        pass
                elapsed = time.time() - t0
                print(f" OK ({elapsed:.0f}s, {free}/{total} GPUs free)")
                break
        except Exception:
            pass
        print(".", end="", flush=True)
    else:
        elapsed = time.time() - t0
        print(f" TIMEOUT ({elapsed:.0f}s) — results may be contaminated!")
        return False

    # Phase 3: stabilize
    print(f"  [clear] Stabilizing {stabilize:.0f}s for CUDA cleanup...",
          end="", flush=True)
    time.sleep(stabilize)
    print(" ready")
    return True


def verify_cold_state() -> bool:
    """Verify no models are loaded (0 deployments, all GPUs free)."""
    try:
        resp = requests.get(f"{API_URL}/v1/models", timeout=5)
        all_deps = resp.json().get("data", [])
        # Zombie entries (desired=0, ready=0) hold no GPU — ignore them
        deployments = [
            d for d in all_deps
            if d.get("desired_replicas", 0) > 0
            or d.get("ready_replicas", 0) > 0
        ]
        resp = requests.get(f"{API_URL}/status", timeout=5)
        nodes = resp.json().get("nodes", [])
        total = sum(n.get("total_gpus", 0) for n in nodes)
        free = sum(n.get("available_gpus", 0) for n in nodes)
        ok = len(deployments) == 0 and free >= total and total > 0
        if not ok:
            print(f"  [WARN] Not cold: {len(deployments)} active deployments, "
                  f"{free}/{total} GPUs free")
        return ok
    except Exception as e:
        print(f"  [WARN] Could not verify cold state: {e}")
        return False


# ===========================================================================
#  Batch Submission
# ===========================================================================

def submit_batch(tasks: list, metadata: dict | None = None,
                 poll_interval: float = 2.0,
                 quiet: bool = False,
                 allow_failures: bool = False) -> dict:
    """Submit a batch and block until completed/failed.

    Args:
        allow_failures: If True, return results even when tasks fail
            instead of raising RuntimeError. The caller can check
            the 'failed_count' key in the returned dict.

    Returns dict with keys:
      batch_id, status, duration, num_tasks, tasks_data, raw,
      failed_count (only when allow_failures=True and failures exist)
    """
    payload: dict = {"tasks": tasks}
    if metadata:
        payload["metadata"] = metadata

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
            print(f"  [{elapsed:6.1f}s] status -> {status}")
            last_status = status

        if status in ("completed", "failed", "cancelled"):
            duration = time.time() - t0
            tasks_data = data.get("tasks", [])
            failed_tasks = [t for t in tasks_data if t.get("status") == "failed"]
            if failed_tasks and not allow_failures:
                first_err = failed_tasks[0].get("output", {}).get("error", "unknown")
                raise RuntimeError(
                    f"Batch {batch_id}: {len(failed_tasks)}/{len(tasks)} tasks failed. "
                    f"First error: {first_err}"
                )
            result = {
                "batch_id": batch_id,
                "status": status,
                "duration": round(duration, 2),
                "num_tasks": len(tasks),
                "tasks_data": tasks_data,
                "raw": data,
            }
            if failed_tasks:
                result["failed_count"] = len(failed_tasks)
                if not quiet:
                    print(f"  [WARN] {len(failed_tasks)}/{len(tasks)} tasks failed")
            return result
        time.sleep(poll_interval)


# ===========================================================================
#  Task & Metrics
# ===========================================================================

def make_task(model: str, prompt: str, task_id: str,
              max_tokens: int = 50) -> dict:
    """Create one batch task dict (OpenAI-compatible)."""
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


def extract_task_metrics(tasks_data: list) -> dict:
    """Extract per-task latencies from batch response.

    Returns dict with:
      latencies, avg_latency, min_latency, max_latency,
      processing_window, completed, failed
    """
    latencies = []
    min_start = float("inf")
    max_end = 0.0
    completed = 0
    failed = 0

    for t in tasks_data:
        if t.get("status") == "failed":
            failed += 1
            continue
        if t.get("status") != "completed":
            continue
        completed += 1

        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        if not s_str or not e_str:
            continue
        try:
            s_ts = datetime.fromisoformat(
                s_str.replace("Z", "+00:00")
            ).timestamp()
            e_ts = datetime.fromisoformat(
                e_str.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            continue
        dur = max(0, e_ts - s_ts)
        latencies.append(dur)
        min_start = min(min_start, s_ts)
        max_end = max(max_end, e_ts)

    if not latencies:
        return {
            "latencies": [], "avg_latency": 0, "min_latency": 0,
            "max_latency": 0, "processing_window": 0,
            "completed": completed, "failed": failed,
        }

    return {
        "latencies": [round(l, 3) for l in latencies],
        "avg_latency": round(statistics.mean(latencies), 3),
        "min_latency": round(min(latencies), 3),
        "max_latency": round(max(latencies), 3),
        "processing_window": round(max_end - min_start, 3),
        "completed": completed,
        "failed": failed,
    }


def analyze_model_groups(tasks_data: list) -> dict:
    """Analyze per-model-group timing from task data.

    Returns {model_name: {count, first_start, last_end, duration, avg_lat}}.
    Useful for measuring transition gaps between groups.
    """
    groups = {}
    for t in tasks_data:
        model = t.get("body", {}).get("model", "unknown")
        if model not in groups:
            groups[model] = {"starts": [], "ends": [], "latencies": [],
                             "completed": 0, "failed": 0}
        if t.get("status") == "failed":
            groups[model]["failed"] += 1
            continue
        if t.get("status") != "completed":
            continue
        groups[model]["completed"] += 1

        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        if not s_str or not e_str:
            continue
        try:
            s_ts = datetime.fromisoformat(
                s_str.replace("Z", "+00:00")
            ).timestamp()
            e_ts = datetime.fromisoformat(
                e_str.replace("Z", "+00:00")
            ).timestamp()
            groups[model]["starts"].append(s_ts)
            groups[model]["ends"].append(e_ts)
            groups[model]["latencies"].append(e_ts - s_ts)
        except ValueError:
            pass

    result = {}
    for model, g in groups.items():
        short = model.split("/")[-1]
        entry = {
            "completed": g["completed"],
            "failed": g["failed"],
        }
        if g["starts"]:
            entry["first_start"] = min(g["starts"])
            entry["last_end"] = max(g["ends"])
            entry["duration"] = round(entry["last_end"] - entry["first_start"], 3)
            entry["avg_latency"] = round(statistics.mean(g["latencies"]), 3)
        result[short] = entry
    return result


# ===========================================================================
#  Results I/O
# ===========================================================================

def make_analytical_result(name: str, makespan: float, num_tasks: int) -> dict:
    """Create a synthetic batch result from an analytical makespan estimate.

    Produces the same dict shape as submit_batch() so experiment scripts can
    use it uniformly for baseline configs without hitting the live cluster.
    """
    throughput = round(num_tasks / makespan, 4) if makespan > 0 else 0.0
    return {
        "batch_id": f"analytical-{name}",
        "status": "completed",
        "duration": round(makespan, 2),
        "num_tasks": num_tasks,
        "tasks_data": [],
        "source": "analytical",
        "throughput": throughput,
    }


def save_result(result: dict, filepath: Path):
    """Save result dict as pretty-printed JSON."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [saved] {filepath}")


def aggregate_runs(runs: list[dict], key: str = "duration") -> dict:
    """Compute mean/std/min/max across multiple runs for a given key.

    Also returns the full list of per-run values.
    """
    values = [r[key] for r in runs if key in r]
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "values": [],
                "runs": runs}
    return {
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0,
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "values": [round(v, 2) for v in values],
        "runs": runs,
    }


# ===========================================================================
#  GPU Query
# ===========================================================================

def get_gpu_count() -> int:
    """Query the server for total GPU count across all workers."""
    try:
        resp = requests.get(f"{API_URL}/status", timeout=10)
        resp.raise_for_status()
        nodes = resp.json().get("nodes", [])
        total = sum(n.get("total_gpus", 0) for n in nodes)
        return max(1, total)
    except Exception as e:
        print(f"  [warn] Could not query GPU count: {e}, assuming 1")
        return 1


# ===========================================================================
#  Printing
# ===========================================================================

def print_header(title: str, params: dict | None = None):
    """Print a formatted experiment header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    if params:
        for k, v in params.items():
            print(f"    {k}: {v}")
    print("=" * 70)


def print_separator(label: str = ""):
    print(f"\n{'─' * 50}")
    if label:
        print(f"  {label}")
        print(f"{'─' * 50}")
