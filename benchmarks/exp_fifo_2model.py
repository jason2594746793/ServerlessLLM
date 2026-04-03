#!/usr/bin/env python3
"""
Exp A — 2-Model FIFO Partial Validation
=========================================
Runs real FIFO scheduling on a 2-model batch (0.6B + 8B, no 32B) to
cross-validate the 83.3s analytical per-switch cost used in Exp 1.

With 20 interleaved tasks (10 per model), there are 19 switches
alternating between 0.6B→8B (~61.9s from Exp 2) and 8B→0.6B (~34s).
Weighted mean expected: ~48s — lower than 83.3s because 32B transitions
(115.2s) are excluded.

Runtime: ~15 minutes for 1 run.
GPU requirement: 1–2 GPUs (0.6B and 8B each fit on 1 GPU, no TP needed).

Results saved to: results/exp_fifo_2model/
"""

import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
    save_result,
    make_task,
    clear_deployments,
    verify_cold_state,
    evict_model_page_cache,
    print_header,
    MODEL_SMALL,
    MODEL_LARGE,
    RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp_fifo_2model"
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE]   # 0.6B and 8B only — no 32B
NUM_TASKS = 20                             # 10 per model, interleaved
NUM_RUNS = 1                               # single run is sufficient


def generate_workload():
    """Round-robin interleaved: 0.6B, 8B, 0.6B, 8B, ..."""
    tasks = []
    labels = ["S", "L"]
    for i in range(NUM_TASKS):
        m = EXP_MODELS[i % 2]
        tag = labels[i % 2]
        tasks.append(make_task(m, f"Explain concept {i + 1} in one sentence.", f"task-{i}-{tag}"))
    return tasks


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print_header("Exp A — 2-Model FIFO Partial Validation (W1 cross-check)", {
        "Models":    f"{MODEL_SMALL}  +  {MODEL_LARGE}",
        "Tasks":     f"{NUM_TASKS} interleaved ({NUM_TASKS // 2}/model) = {NUM_TASKS - 1} switches",
        "Strategy":  "FIFO (no grouping, no prefetch)",
        "Expected":  "~900s makespan, ~48s/switch avg (0.6B↔8B only)",
        "Purpose":   "Cross-validate 83.3s analytical estimate from Exp 1",
    })

    all_runs = []
    for run_idx in range(NUM_RUNS):
        print(f"\n  ── Run {run_idx + 1}/{NUM_RUNS} ──")

        # Cold start — evict page cache for both models
        for m in EXP_MODELS:
            evict_model_page_cache(m)
        if not clear_deployments():
            print("  [WARN] clear_deployments incomplete")
        if not verify_cold_state():
            print("  [WARN] System may not be fully cold")

        # No register_model_deployment needed — 0.6B and 8B each fit on 1 GPU

        set_strategy(
            strategy="semaphore",
            buffer_limit=8,
            enable_model_grouping=False,
            enable_johnsons_rule=False,
            enable_prefetch=False,
            prefetch_threshold=0.0,
        )

        batch = submit_batch(generate_workload(), allow_failures=True)
        makespan = batch["duration"]
        failed = batch.get("failed_count", 0)
        completed = NUM_TASKS - failed
        metrics = extract_task_metrics(batch["tasks_data"])
        throughput = completed / makespan if makespan > 0 else 0

        # Compute implied per-switch cost
        n_switches = NUM_TASKS - 1  # 19 for 20 tasks
        avg_infer = statistics.mean(metrics["latencies"]) if metrics.get("latencies") else 0.5
        implied_switch = (makespan - NUM_TASKS * avg_infer) / n_switches if n_switches > 0 else 0

        result = {
            "run": run_idx + 1,
            "makespan": makespan,
            "throughput": round(throughput, 2),
            "failed": failed,
            "n_switches": n_switches,
            "avg_inference_latency": round(avg_infer, 3),
            "implied_switch_cost": round(implied_switch, 1),
            "batch_id": batch["batch_id"],
            **metrics,
        }
        all_runs.append(result)
        save_result(result, OUTDIR / f"fifo_2model_run{run_idx + 1}.json")

        print(f"  Makespan:              {makespan:.1f}s")
        print(f"  Throughput:            {throughput:.2f} req/s")
        print(f"  Implied switch cost:   {implied_switch:.1f}s  "
              f"(expected ~48s for 0.6B↔8B; 83.3s is 3-model weighted avg)")
        print(f"  Avg inference latency: {avg_infer:.3f}s")
        if failed:
            print(f"  Failed tasks:          {failed}")

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print(f"  2-model FIFO implied switch cost: "
          f"{all_runs[0]['implied_switch_cost']:.1f}s/switch (0.6B+8B only)")
    print(f"  Exp 2 measured transitions:       "
          f"0.6B→8B = 61.9s,  8B→0.6B ≈ 34s,  weighted avg ≈ 48s")
    print(f"  Exp 1 analytical 3-model avg:     83.3s  "
          f"(includes 8B→32B at 115.2s)")
    print("=" * 60)

    save_result({"runs": all_runs}, OUTDIR / "summary.json")


if __name__ == "__main__":
    run()
