#!/usr/bin/env python3
"""
Experiment 7 — Multi-GPU Scaling
================================
Thesis §4.8: Measure how the Shared-Aware scheduler scales across
multiple GPUs (1, 2, 4, 8).

Methodology:
  - 200 tasks: ~67 per model × 3 models (interleaved)
  - Full strategy (grouping + JR + prefetch) for all GPU counts
  - GPU count controlled by starting cluster with N workers
  - 3 runs per GPU count, report mean ± std
  - Cold-start verified between every run

NOTE: This experiment requires MANUALLY restarting the cluster with
different numbers of GPU workers for each data point.  The script
pauses and prompts the operator to reconfigure the cluster.

Expected:
  - Near-linear scaling up to the number of model groups (3 models → 3 GPUs)
  - Beyond 3 GPUs, diminishing returns (more GPUs than distinct models)
  - 8 GPUs may help if the router balances inference across replicas
"""

import sys
import time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
    analyze_model_groups,
    save_result,
    make_task,
    clear_deployments,
    verify_cold_state,
    aggregate_runs,
    print_header,
    print_separator,
    get_gpu_count,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    RESULTS_ROOT,
    API_URL,
    DEFAULT_RUNS,
)

OUTDIR = RESULTS_ROOT / "exp7_multi_gpu"
NUM_TASKS = 200
GPU_COUNTS = [1, 2, 4, 8]


# ── Workload ───────────────────────────────────────────────────────────
def generate_workload(n: int = NUM_TASKS):
    """Round-robin S M L S M L ... across 3 models."""
    tasks = []
    models = [MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE]
    for i in range(n):
        m = models[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(
            make_task(m, f"Explain topic {i + 1}.", f"task-{i}-{tag}")
        )
    return tasks


def wait_for_cluster(expected_gpus: int, timeout: float = 120):
    """Wait for the cluster to report expected number of GPUs."""
    print(f"  Waiting for cluster to report {expected_gpus} GPUs...",
          end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = requests.get(f"{API_URL}/status", timeout=5)
            nodes = resp.json().get("nodes", [])
            total = sum(n.get("total_gpus", 0) for n in nodes)
            if total >= expected_gpus:
                print(f" OK ({total} GPUs detected)")
                return True
        except Exception:
            pass
        time.sleep(3)
        print(".", end="", flush=True)
    print(f" TIMEOUT")
    return False


# ── Main ───────────────────────────────────────────────────────────────
def run():
    print_header("EXP 7 — Multi-GPU Scaling", {
        "GPU counts": GPU_COUNTS,
        "Tasks": NUM_TASKS,
        "Models": "3 (interleaved)",
        "Strategy": "Full (grouping + JR + prefetch)",
        "Runs": DEFAULT_RUNS,
    })

    all_results = {}

    for ngpu in GPU_COUNTS:
        print_separator(f"GPU Count: {ngpu}")

        # Prompt operator to reconfigure cluster
        print(f"\n  ╔═════════════════════════════════════��════════════════╗")
        print(f"  ║  ACTION REQUIRED: Start cluster with {ngpu} GPU(s)       ║")
        print(f"  ║  Then press ENTER to continue...                    ║")
        print(f"  ╚══════════════════════════════════════════════════════╝")
        input()

        # Verify cluster has expected GPUs
        if not wait_for_cluster(ngpu):
            print(f"  [WARN] Could not verify {ngpu} GPUs, proceeding anyway")

        detected = get_gpu_count()
        print(f"  Detected GPUs: {detected}")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

            if not clear_deployments():
                print("  [WARN] Cold-start verification failed")
            if not verify_cold_state():
                print("  [WARN] System may not be fully cold")

            set_strategy(
                strategy="semaphore",
                buffer_limit=8,
                enable_model_grouping=True,
                enable_johnsons_rule=True,
                enable_prefetch=True,
                prefetch_threshold=0.0,
            )

            tasks = generate_workload()
            batch = submit_batch(tasks)
            metrics = extract_task_metrics(batch["tasks_data"])
            groups = analyze_model_groups(batch["tasks_data"])

            makespan = batch["duration"]
            throughput = batch["num_tasks"] / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "gpu_count": ngpu,
                "detected_gpus": detected,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "groups": groups,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result,
                        OUTDIR / f"gpu{ngpu}_run{run_idx + 1}.json")

            print(f"  Makespan:   {makespan:.2f}s")
            print(f"  Throughput: {throughput:.2f} req/s")

        agg = aggregate_runs(runs, key="makespan")
        tput_agg = aggregate_runs(runs, key="throughput")
        all_results[ngpu] = {
            "gpu_count": ngpu,
            "makespan_agg": agg,
            "throughput_agg": tput_agg,
            "runs": runs,
        }
        save_result(all_results[ngpu],
                     OUTDIR / f"gpu{ngpu}_aggregate.json")

        print(f"\n  ── Aggregate (GPU={ngpu}) ──")
        print(f"  Makespan:   {agg['mean']:.2f} ± {agg['std']:.2f}s")
        print(f"  Throughput: {tput_agg['mean']:.2f} ± {tput_agg['std']:.2f}")

    # ── Summary ────────────────────────────────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    print("\n" + "=" * 70)
    print("SUMMARY — Multi-GPU Scaling")
    print(f"{'GPUs':<8} {'Makespan (s)':<16} {'Throughput':<16} {'Speedup':<10}")
    print("-" * 50)

    baseline_ms = None
    for ngpu in GPU_COUNTS:
        r = all_results.get(ngpu)
        if r:
            ms = r["makespan_agg"]["mean"]
            ms_std = r["makespan_agg"]["std"]
            tp = r["throughput_agg"]["mean"]
            tp_std = r["throughput_agg"]["std"]
            if baseline_ms is None:
                baseline_ms = ms
            speedup = baseline_ms / ms if ms > 0 else 0
            print(f"{ngpu:<8} {ms:.1f}±{ms_std:.1f}{'':>4} "
                  f"{tp:.2f}±{tp_std:.2f}{'':>4} {speedup:.2f}x")

    if baseline_ms and len(all_results) > 1:
        print(f"\nScaling efficiency (ideal = linear):")
        for ngpu in GPU_COUNTS:
            r = all_results.get(ngpu)
            if r:
                ms = r["makespan_agg"]["mean"]
                speedup = baseline_ms / ms if ms > 0 else 0
                efficiency = speedup / ngpu * 100
                print(f"  {ngpu} GPUs: {speedup:.2f}x speedup "
                      f"({efficiency:.0f}% efficiency)")
    print("=" * 70)


if __name__ == "__main__":
    run()
