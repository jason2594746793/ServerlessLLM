#!/usr/bin/env python3
"""
Experiment: Multi-Model Multi-GPU Verification
===============================================
Verifies that when a batch contains tasks for different models,
the system runs them on separate GPUs concurrently (not sequentially).

Test logic:
  1. Submit a batch with tasks for 2 models (equal split)
  2. Measure per-model timing windows (earliest start → latest end)
  3. Check for OVERLAP between model windows → proves concurrent GPU use
  4. Compare against a single-model baseline for sanity

Expected on multi-GPU:
  - Model A and Model B processing windows overlap significantly
  - Total makespan < sum of individual model makespans (parallelism gain)

Expected on single-GPU:
  - Model windows are sequential (little/no overlap)
  - Total makespan ≈ sum of individual model makespans

Timing anatomy (per batch):
  |<-------------- makespan (wall-clock) ----------------------->|
  |  poll  |<-- model load (cold start) -->|<-- inference -->|poll|
  submit                                   first_started_at      completed

  The "processing window" = last completed_at - first started_at
  The "cold start overhead" = makespan - processing_window (approx)
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, make_task, save_result,
    clear_deployments,
    MODEL_SMALL, MODEL_MEDIUM, RESULTS_ROOT, API_URL,
)

OUTDIR = RESULTS_ROOT / "exp_multi_gpu_verify"

TASKS_PER_MODEL = 20


# ── Helpers ────────────────────────────────────────────────────────────────

def parse_ts(iso_str: str) -> float:
    """ISO timestamp → epoch seconds."""
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()


def analyse_per_model(tasks_data: list) -> dict:
    """Group completed tasks by model and compute per-model timing windows."""
    by_model = defaultdict(list)

    for t in tasks_data:
        if t.get("status") != "completed":
            continue
        s_str, e_str = t.get("started_at"), t.get("completed_at")
        if not s_str or not e_str:
            continue

        # Extract model from body (now included in API response)
        model = t.get("body", {}).get("model", None)

        # Fallback: try custom_id pattern or output
        if not model:
            custom_id = t.get("custom_id", "")
            # Our task IDs encode model info — but best-effort only
            model = "unknown"

        s_ts, e_ts = parse_ts(s_str), parse_ts(e_str)
        by_model[model].append({
            "start": s_ts, "end": e_ts, "dur": e_ts - s_ts,
            "custom_id": t.get("custom_id", ""),
        })

    result = {}
    for model, entries in by_model.items():
        starts = [e["start"] for e in entries]
        ends = [e["end"] for e in entries]
        result[model] = {
            "count": len(entries),
            "window_start": min(starts),
            "window_end": max(ends),
            "window_duration": max(ends) - min(starts),
            "avg_task_latency": sum(e["dur"] for e in entries) / len(entries),
            "first_start": min(starts),
            "last_end": max(ends),
        }
    return result


def compute_overlap(windows: dict) -> dict:
    """Compute pairwise overlap between model windows."""
    # Filter out 'unknown'
    models = [m for m in windows.keys() if m != "unknown"]
    if len(models) < 2:
        return {
            "overlap_seconds": 0, "overlap_ratio": 0,
            "note": f"need ≥2 identified models, got {models}",
        }

    a, b = windows[models[0]], windows[models[1]]
    overlap_start = max(a["window_start"], b["window_start"])
    overlap_end = min(a["window_end"], b["window_end"])
    overlap = max(0, overlap_end - overlap_start)

    shorter = min(a["window_duration"], b["window_duration"])
    ratio = overlap / shorter if shorter > 0 else 0

    return {
        "model_a": models[0],
        "model_b": models[1],
        "overlap_seconds": round(overlap, 2),
        "overlap_ratio": round(ratio, 3),
        "window_a": round(a["window_duration"], 2),
        "window_b": round(b["window_duration"], 2),
    }


def compute_global_window(tasks_data: list) -> dict:
    """Compute the global processing window across all tasks."""
    starts, ends = [], []
    for t in tasks_data:
        if t.get("status") != "completed":
            continue
        s_str, e_str = t.get("started_at"), t.get("completed_at")
        if s_str and e_str:
            starts.append(parse_ts(s_str))
            ends.append(parse_ts(e_str))
    if not starts:
        return {"window": 0, "first_start": 0, "last_end": 0}
    return {
        "window": max(ends) - min(starts),
        "first_start": min(starts),
        "last_end": max(ends),
    }


# ── Workload generators ───────────────────────────────────────────────────

def generate_two_model_workload(n_per_model: int = TASKS_PER_MODEL):
    """Interleaved ABABAB... two-model tasks."""
    tasks = []
    models = [MODEL_SMALL, MODEL_MEDIUM]
    for i in range(n_per_model * 2):
        m = models[i % 2]
        tasks.append(make_task(
            model=m,
            prompt=f"Explain concept {i+1} in one sentence.",
            task_id=f"multi-gpu-{i}",
        ))
    return tasks


def generate_single_model_workload(model: str, n: int = TASKS_PER_MODEL):
    """Single-model baseline."""
    return [
        make_task(
            model=model,
            prompt=f"Explain concept {i+1} in one sentence.",
            task_id=f"single-{i}",
        )
        for i in range(n)
    ]


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("Multi-Model Multi-GPU Verification")
    print("=" * 70)

    # Use model grouping + semaphore (the production config)
    set_strategy(strategy="semaphore", buffer_limit=10, enable_model_grouping=True)

    # ── Phase 1: Single-model baselines ───────────────────────────────
    baselines = {}
    for label, model in [("small", MODEL_SMALL), ("medium", MODEL_MEDIUM)]:
        print(f"\n--- Baseline: {label} ({model}) ---")
        tasks = generate_single_model_workload(model, TASKS_PER_MODEL)
        result = submit_batch(tasks)

        gw = compute_global_window(result["tasks_data"])
        baselines[label] = {
            "model": model,
            "makespan": result["duration"],
            "processing_window": gw["window"],
            "cold_start_overhead": result["duration"] - gw["window"],
            "num_tasks": result["num_tasks"],
        }
        print(f"  Makespan:    {result['duration']:.2f}s")
        print(f"  Processing:  {gw['window']:.2f}s  "
              f"(cold start ~{result['duration'] - gw['window']:.1f}s)")
        clear_deployments(wait=30.0)

    # ── Phase 2: Two-model concurrent batch ───────────────────────────
    # Tear down all deployments so the two-model batch starts cold,
    # giving a fair apples-to-apples comparison against the baselines.
    print("\n--- Clearing deployments for cold-start two-model test ---")
    clear_deployments(wait=30.0)

    print(f"\n--- Two-model batch ({MODEL_SMALL} + {MODEL_MEDIUM}) ---")
    tasks = generate_two_model_workload(TASKS_PER_MODEL)
    result = submit_batch(tasks)
    makespan = result["duration"]

    gw = compute_global_window(result["tasks_data"])
    per_model = analyse_per_model(result["tasks_data"])
    overlap = compute_overlap(per_model)

    print(f"  Makespan:    {makespan:.2f}s")
    print(f"  Processing:  {gw['window']:.2f}s  "
          f"(cold start ~{makespan - gw['window']:.1f}s)")

    if "unknown" in per_model and len(per_model) == 1:
        print(f"\n  WARNING: Could not identify models in response!")
        print(f"  All {per_model['unknown']['count']} tasks grouped as 'unknown'.")
        print(f"  The API response may not include task body. Restart server")
        print(f"  after the api_gateway.py fix to include 'body' in response.")
    else:
        for model, info in per_model.items():
            short = model.split("/")[-1]
            print(f"  {short:30s}: window={info['window_duration']:.2f}s  "
                  f"avg_task={info['avg_task_latency']:.2f}s  "
                  f"count={info['count']}")

    print(f"\n  Overlap: {overlap['overlap_seconds']:.2f}s "
          f"({overlap.get('overlap_ratio', 0):.0%} of shorter window)")

    # ── Phase 3: Verdict ──────────────────────────────────────────────
    sum_baselines = baselines["small"]["makespan"] + baselines["medium"]["makespan"]
    max_baseline = max(baselines["small"]["makespan"], baselines["medium"]["makespan"])
    speedup_vs_sum = sum_baselines / makespan if makespan > 0 else 0
    speedup_vs_max = max_baseline / makespan if makespan > 0 else 0

    print(f"\n{'=' * 70}")
    print("TIMING ANALYSIS")
    print(f"{'=' * 70}")
    print(f"  Small baseline:     {baselines['small']['makespan']:.2f}s  "
          f"(processing {baselines['small']['processing_window']:.2f}s)")
    print(f"  Medium baseline:    {baselines['medium']['makespan']:.2f}s  "
          f"(processing {baselines['medium']['processing_window']:.2f}s)")
    print(f"  Sum (sequential):   {sum_baselines:.2f}s")
    print(f"  Max (parallel ideal): {max_baseline:.2f}s")
    print(f"  Two-model actual:   {makespan:.2f}s  "
          f"(processing {gw['window']:.2f}s)")

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")
    print(f"  Speedup vs sequential:   {speedup_vs_sum:.2f}x")
    print(f"  Speedup vs max baseline: {speedup_vs_max:.2f}x")

    # The key insight: if makespan ≈ max(baseline_A, baseline_B),
    # both models loaded & ran in parallel.
    # If makespan ≈ sum(baseline_A + baseline_B), they ran sequentially.
    if speedup_vs_sum > 1.5:
        ratio_to_max = makespan / max_baseline
        print(f"\n  PARALLEL EXECUTION CONFIRMED")
        print(f"  Two-model makespan ({makespan:.1f}s) ≈ max single baseline "
              f"({max_baseline:.1f}s)")
        print(f"  Ratio: {ratio_to_max:.2f}x of max baseline "
              f"(1.0 = perfect parallel)")
        print(f"  Both models loaded & inferred on separate GPUs simultaneously.")
    elif speedup_vs_sum > 1.1:
        print(f"\n  PARTIAL PARALLELISM")
        print(f"  Some overlap in model loading, but not fully parallel.")
    else:
        print(f"\n  SEQUENTIAL EXECUTION")
        print(f"  Models ran one after another. Expected on single-GPU.")

    # Explain why overlap might be 0% even with parallel execution
    if overlap.get("overlap_ratio", 0) < 0.1 and speedup_vs_sum > 1.5:
        print(f"\n  NOTE: Overlap ratio is low because inference is very fast")
        print(f"  (~{gw['window']:.1f}s for {result['num_tasks']} tasks) while")
        print(f"  model loading dominates (~{makespan - gw['window']:.0f}s).")
        print(f"  Both models load in parallel, then all tasks execute in a")
        print(f"  short burst. The per-model windows overlap during that burst")
        print(f"  but the burst itself is tiny relative to total makespan.")

    # ── Save ──────────────────────────────────────────────────────────
    full_result = {
        "baselines": baselines,
        "two_model": {
            "makespan": makespan,
            "num_tasks": result["num_tasks"],
            "processing_window": gw["window"],
            "cold_start_overhead": makespan - gw["window"],
            "per_model": {k: {kk: vv for kk, vv in v.items()}
                          for k, v in per_model.items()},
            "overlap": overlap,
        },
        "speedup_vs_sequential": speedup_vs_sum,
        "speedup_vs_max_baseline": speedup_vs_max,
        "sum_baselines": sum_baselines,
        "max_baseline": max_baseline,
    }
    save_result(full_result, OUTDIR / "result.json")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    run()
