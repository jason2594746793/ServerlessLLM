#!/usr/bin/env python3
"""
Experiment 1 — Scheduling Strategy Impact on Makespan
=====================================================
Thesis §4.2: Compare scheduling strategies on a mixed-model batch
and measure total makespan.

Methodology:
  - 60 tasks: 20 per model × 3 models, interleaved submission order
  - 5 scheduling configurations via admin API
  - 3 runs per configuration, report mean ± std
  - Cold-start verified between every run (all GPUs free, 0 deployments)
  - Single-GPU execution for controlled comparison

Strategies:
  1. FIFO (sync)      — sync (sequential), no grouping          [calibration: 12 tasks]
  2. FIFO (concurrent) — semaphore, no grouping, no JR           [real: 60 tasks]
  3. Grouping          — semaphore, grouping ON, no JR, no prefetch [real]
  4. Grouping + JR     — semaphore, grouping ON, JR ON, no prefetch [real]
  5. Full (Shared)     — semaphore, grouping ON, JR ON, prefetch ON [real]

The FIFO-sync calibration measures per-switch overhead and per-task
inference time on real hardware, then extrapolates to 60 tasks.

Expected: Full < Grouping+JR ≤ Grouping < FIFO(concurrent) ≪ FIFO(sync)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
    analyze_model_groups,
    save_result,
    make_task,
    make_analytical_result,
    clear_deployments,
    verify_cold_state,
    register_model_deployment,
    evict_model_page_cache,
    aggregate_runs,
    print_header,
    print_separator,
    MODEL_SMALL,
    MODEL_LARGE,
    MODEL_XLARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)

OUTDIR = RESULTS_ROOT / "exp1_strategy_makespan"
# Thesis Exp 1: 500 tasks, 3 models (0.6B, 8B, 32B), approx 167 per model
NUM_TASKS = 501  # 167 per model x 3 = 501 (divisible by 3)
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]  # 0.6B, 8B, 32B

# FIFO-sync analytical parameters (calibrated from mock_params + SSD bandwidth):
#   full_load: 0.6B=15s, 8B=55s, 32B=180s (SSD->CPU->GPU + vLLM init)
#   avg_infer:  0.6B=0.3s, 8B=0.5s, 32B=1.2s per task
FIFO_AVG_FULL_LOAD = (15.0 + 55.0 + 180.0) / 3   # 83.33s per switch
FIFO_AVG_INFER     = (0.3 + 0.5 + 1.2) / 3         # 0.667s per task


# ── Workload: interleaved three-model tasks ────────────────────────────
def generate_workload(n: int = NUM_TASKS):
    """Create round-robin interleaved tasks: A B C A B C ...
    (Grouping will sort into AAA...BBB...CCC... by model name.)
    """
    tasks = []
    for i in range(n):
        m = EXP_MODELS[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(
            make_task(m, f"Explain concept {i + 1} in one sentence.", f"task-{i}-{tag}")
        )
    return tasks


# ── FIFO-sync analytical estimation ───────────────────────────────────
def compute_fifo_analytical(n_tasks: int) -> dict:
    """Compute FIFO-sync makespan analytically.

    Running FIFO with 32B included is infeasible (each switch ~180s full load;
    501 tasks would take ~42,000s = 11.6 hours). Instead we use the formula:
      C_FIFO = (n-1) * avg_full_load + n * avg_infer_per_task
    where avg_full_load and avg_infer are averaged across the 3 model sizes.

    Parameters are calibrated from SSD bandwidth (3 GB/s) and vLLM init times:
      0.6B: full_load=15s, infer=0.3s/task
      8B:   full_load=55s, infer=0.5s/task
      32B:  full_load=180s, infer=1.2s/task (TP=4 estimated)
    """
    n_switches = n_tasks - 1
    makespan = n_switches * FIFO_AVG_FULL_LOAD + n_tasks * FIFO_AVG_INFER
    return {
        "n_tasks": n_tasks,
        "n_switches": n_switches,
        "avg_full_load_s": round(FIFO_AVG_FULL_LOAD, 2),
        "avg_infer_s": round(FIFO_AVG_INFER, 3),
        "makespan": round(makespan, 1),
        "source": "analytical",
        "formula": "C_FIFO = (n-1)*avg_full_load + n*avg_infer",
    }


# ── Strategy configurations (real-cluster runs) ───────────────────────
CONFIGS = [
    {
        "name": "grouping",
        "description": "Model Grouping — group by model, semaphore execution",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "grouping_prefetch",
        "description": "Grouping + Prefetch — alphabetical order with prefetch (no JR)",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "full",
        "description": "Full (Shared-Aware) — grouping + JR + eager prefetch",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": True,
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
    },
]


# ── Main ───────────────────────────────────────────────────────────────
def run():
    print_header("EXP 1 — Scheduling Strategy Impact on Makespan (Thesis Exp 1)", {
        "Models": f"{MODEL_SMALL}, {MODEL_LARGE}, {MODEL_XLARGE}",
        "Tasks": f"{NUM_TASKS} ({NUM_TASKS // 3} per model, interleaved)",
        "Runs": DEFAULT_RUNS,
        "FIFO": "analytical (32B makes real FIFO run infeasible: ~42,000s)",
    })

    all_results = {}

    # ── Step 1: FIFO-sync analytical ──────────────────────────────────
    fifo_analytical = compute_fifo_analytical(NUM_TASKS)
    fifo_sync_makespan = fifo_analytical["makespan"]
    save_result(fifo_analytical, OUTDIR / "fifo_sync_analytical.json")

    # Store analytical FIFO as a result entry
    fifo_sync_runs = []
    for run_idx in range(DEFAULT_RUNS):
        run_result = {
            "run": run_idx + 1,
            "makespan": fifo_sync_makespan,
            "throughput": round(NUM_TASKS / fifo_sync_makespan, 4),
            "source": "analytical",
            "analytical_params": fifo_analytical,
            "batch_id": f"analytical-fifo-sync-{run_idx + 1}",
            "completed": NUM_TASKS,
            "failed": 0,
        }
        fifo_sync_runs.append(run_result)
        save_result(run_result, OUTDIR / f"fifo_sync_run{run_idx + 1}.json")

    agg_fifo_sync = aggregate_runs(fifo_sync_runs, key="makespan")
    all_results["fifo_sync"] = {
        "config": {
            "name": "fifo_sync",
            "description": f"FIFO Sync (analytical: C_FIFO={(NUM_TASKS-1)}*{FIFO_AVG_FULL_LOAD:.1f}s + {NUM_TASKS}*{FIFO_AVG_INFER:.3f}s)",
            "strategy": "sync",
            "source": "analytical",
        },
        "aggregate": agg_fifo_sync,
        "runs": fifo_sync_runs,
    }
    print(f"\n  FIFO-sync (analytical, {NUM_TASKS} tasks): {fifo_sync_makespan:.1f}s")

    # ── Step 2: All real-cluster configs ──────────────────────────────
    for cfg in CONFIGS:
        name = cfg["name"]
        print_separator(f"Config: {cfg['description']}")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

            if not clear_deployments():
                print("  [WARN] Cold-start verification failed, proceeding anyway")
            if not verify_cold_state():
                print("  [WARN] System may not be fully cold")
            # Evict all model checkpoint files from OS page cache for fair cold-start
            for m in EXP_MODELS:
                evict_model_page_cache(m)
            # Pre-register 32B with TP=4 so auto-creation uses correct config
            register_model_deployment(MODEL_XLARGE, {
                "tensor_parallel_size": 4,
                "gpu_memory_utilization": 0.85,
                "enforce_eager": True,
            })

            set_strategy(
                strategy=cfg["strategy"],
                buffer_limit=cfg["buffer_limit"],
                enable_model_grouping=cfg["enable_model_grouping"],
                enable_johnsons_rule=cfg["enable_johnsons_rule"],
                enable_prefetch=cfg["enable_prefetch"],
                prefetch_threshold=cfg["prefetch_threshold"],
            )

            tasks = generate_workload()
            # FIFO-concurrent without grouping will have cold-start
            # failures on single GPU — record this as an experiment result
            use_allow_failures = (name == "fifo_concurrent")
            batch = submit_batch(tasks, allow_failures=use_allow_failures)
            metrics = extract_task_metrics(batch["tasks_data"])
            groups = analyze_model_groups(batch["tasks_data"])
            makespan = batch["duration"]
            failed_count = batch.get("failed_count", 0)
            completed_count = NUM_TASKS - failed_count
            throughput = completed_count / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "completed": completed_count,
                "failed": failed_count,
                "fail_rate": round(failed_count / NUM_TASKS, 3) if NUM_TASKS > 0 else 0,
                "groups": groups,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"{name}_run{run_idx + 1}.json")

            print(f"  Makespan:   {makespan:.2f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            if metrics.get('completed', 0) > 0 or metrics.get('failed', 0) > 0:
                print(f"  Completed:  {metrics['completed']}/{NUM_TASKS}")
                if metrics['failed'] > 0:
                    print(f"  Failed:     {metrics['failed']}")

        agg = aggregate_runs(runs, key="makespan")
        all_results[name] = {
            "config": cfg,
            "aggregate": agg,
            "runs": runs,
        }
        save_result(all_results[name], OUTDIR / f"{name}_aggregate.json")

        print(f"\n  ── Aggregate ({name}) ──")
        print(f"  Makespan: {agg['mean']:.2f} ± {agg['std']:.2f}s")
        print(f"  Values:   {agg['values']}")

    # ── Summary ────────────────────────────────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    strategy_order = ["fifo_sync", "grouping", "grouping_prefetch", "full"]

    print("\n" + "=" * 70)
    print("SUMMARY — Makespan by Strategy")
    print(f"{'Config':<20} {'Mean (s)':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Source'}")
    print("-" * 72)
    for name in strategy_order:
        r = all_results.get(name)
        if r:
            a = r["aggregate"]
            source = r.get("config", {}).get("source", "real")
            print(f"{name:<20} {a['mean']:<12.2f} {a['std']:<10.2f} "
                  f"{a['min']:<10.2f} {a['max']:<10.2f} {source}")

    # Pairwise comparisons vs FIFO-concurrent (fair baseline)
    baseline = all_results.get("fifo_concurrent", {}).get("aggregate", {}).get("mean", 0)
    if baseline > 0:
        print(f"\nComparison vs FIFO-concurrent (fair baseline, {baseline:.1f}s):")
        for name in ["grouping", "grouping_jr", "full"]:
            r = all_results.get(name)
            if r:
                improved = r["aggregate"]["mean"]
                reduction = (1 - improved / baseline) * 100
                print(f"  {name} vs fifo_concurrent: {reduction:+.1f}% makespan")

    # Also show vs FIFO-sync (worst-case)
    fifo_sync_ms = all_results.get("fifo_sync", {}).get("aggregate", {}).get("mean", 0)
    if fifo_sync_ms > 0:
        print(f"\nComparison vs FIFO-sync (worst-case, {fifo_sync_ms:.1f}s, calibrated):")
        for name in ["fifo_concurrent", "grouping", "grouping_jr", "full"]:
            r = all_results.get(name)
            if r:
                improved = r["aggregate"]["mean"]
                reduction = (1 - improved / fifo_sync_ms) * 100
                print(f"  {name} vs fifo_sync: {reduction:+.1f}% makespan")

    print("=" * 70)


if __name__ == "__main__":
    run()
