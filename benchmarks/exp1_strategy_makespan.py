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
    aggregate_runs,
    print_header,
    print_separator,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    RESULTS_ROOT,
    DEFAULT_RUNS,
)

OUTDIR = RESULTS_ROOT / "exp1_strategy_makespan"
NUM_TASKS = 60  # 20 per model × 3 models
CALIBRATION_TASKS = 12  # 4 per model — small FIFO-sync run for calibration


# ── Workload: interleaved three-model tasks ────────────────────────────
def generate_workload(n: int = NUM_TASKS):
    """Create round-robin interleaved tasks: A B C A B C ...
    (Grouping will sort into AAA...BBB...CCC... by model name.)
    """
    tasks = []
    models = [MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE]
    for i in range(n):
        m = models[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(
            make_task(m, f"Explain concept {i + 1} in one sentence.", f"task-{i}-{tag}")
        )
    return tasks


# ── FIFO-sync calibration ─────────────────────────────────────────────
def run_fifo_calibration():
    """Run a small FIFO-sync batch to measure per-switch overhead and
    per-task inference times.  Returns calibration dict."""
    print_separator("FIFO-sync Calibration (12 tasks)")
    cal_runs = []

    for run_idx in range(DEFAULT_RUNS):
        print(f"\n  ── Calibration Run {run_idx + 1}/{DEFAULT_RUNS} ──")
        if not clear_deployments():
            print("  [WARN] Cold-start verification failed")
        verify_cold_state()

        set_strategy(
            strategy="sync",
            buffer_limit=1,
            enable_model_grouping=False,
            enable_johnsons_rule=False,
            enable_prefetch=False,
            prefetch_threshold=0.0,
        )

        tasks = generate_workload(CALIBRATION_TASKS)
        batch = submit_batch(tasks)
        metrics = extract_task_metrics(batch["tasks_data"])
        makespan = batch["duration"]

        # Count model switches in interleaved order (S,M,L,S,M,L,...)
        # Every consecutive pair is different model → (n-1) switches
        n_switches = CALIBRATION_TASKS - 1

        # Per-task inference time (approximate from avg latency)
        avg_infer = metrics.get("avg_latency", 0.5)

        # Per-switch overhead: total time minus inference time
        total_infer = avg_infer * CALIBRATION_TASKS
        l_switch = (makespan - total_infer) / n_switches if n_switches > 0 else 0

        cal = {
            "run": run_idx + 1,
            "calibration_tasks": CALIBRATION_TASKS,
            "makespan": makespan,
            "n_switches": n_switches,
            "avg_infer_per_task": round(avg_infer, 3),
            "total_infer": round(total_infer, 2),
            "l_switch": round(l_switch, 2),
            "batch_id": batch["batch_id"],
        }
        cal_runs.append(cal)
        print(f"  Makespan:   {makespan:.2f}s")
        print(f"  Switches:   {n_switches}")
        print(f"  L_switch:   {l_switch:.2f}s")
        print(f"  Avg infer:  {avg_infer:.3f}s/task")

    # Average calibration values
    avg_l_switch = sum(c["l_switch"] for c in cal_runs) / len(cal_runs)
    avg_l_infer = sum(c["avg_infer_per_task"] for c in cal_runs) / len(cal_runs)
    cal_result = {
        "calibration_runs": cal_runs,
        "avg_l_switch": round(avg_l_switch, 2),
        "avg_l_infer": round(avg_l_infer, 3),
    }
    save_result(cal_result, OUTDIR / "fifo_sync_calibration.json")
    print(f"\n  Calibration result: L_switch={avg_l_switch:.2f}s, "
          f"L_infer={avg_l_infer:.3f}s/task")
    return cal_result


def extrapolate_fifo_sync(calibration: dict, n_tasks: int) -> float:
    """Extrapolate FIFO-sync makespan for n_tasks from calibration data.
    C_FIFO ≈ (n-1) × L_switch + n × L_infer
    (Interleaved order: every consecutive pair is a different model.)
    """
    l_switch = calibration["avg_l_switch"]
    l_infer = calibration["avg_l_infer"]
    n_switches = n_tasks - 1  # interleaved → every pair switches
    return n_switches * l_switch + n_tasks * l_infer


# ── Strategy configurations (real-cluster runs) ───────────────────────
CONFIGS = [
    {
        "name": "fifo_concurrent",
        "description": "FIFO Concurrent — semaphore, no grouping (fair baseline)",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": False,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
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
        "name": "grouping_jr",
        "description": "Grouping + Johnson's Rule — optimised group ordering",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": True,
        "enable_prefetch": False,
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
    print_header("EXP 1 — Scheduling Strategy Impact on Makespan", {
        "Models": f"{MODEL_SMALL}, {MODEL_MEDIUM}, {MODEL_LARGE}",
        "Tasks": f"{NUM_TASKS} ({NUM_TASKS // 3} per model, interleaved)",
        "Calibration": f"{CALIBRATION_TASKS} tasks (FIFO-sync)",
        "Runs": DEFAULT_RUNS,
        "GPU": "single GPU (controlled comparison)",
    })

    all_results = {}

    # ── Step 1: FIFO-sync calibration ─────────────────────────────────
    calibration = run_fifo_calibration()
    fifo_sync_makespan = extrapolate_fifo_sync(calibration, NUM_TASKS)

    # Store calibrated FIFO-sync as a result entry
    fifo_sync_runs = []
    for run_idx in range(DEFAULT_RUNS):
        run_result = {
            "run": run_idx + 1,
            "makespan": fifo_sync_makespan,
            "throughput": round(NUM_TASKS / fifo_sync_makespan, 2),
            "source": "calibrated_analytical",
            "calibration": calibration,
            "batch_id": f"calibrated-fifo-sync-{run_idx + 1}",
            "completed": NUM_TASKS,
            "failed": 0,
        }
        fifo_sync_runs.append(run_result)
        save_result(run_result, OUTDIR / f"fifo_sync_run{run_idx + 1}.json")

    agg_fifo_sync = aggregate_runs(fifo_sync_runs, key="makespan")
    all_results["fifo_sync"] = {
        "config": {
            "name": "fifo_sync",
            "description": f"FIFO Sync (calibrated from {CALIBRATION_TASKS}-task measurement)",
            "strategy": "sync",
            "source": "calibrated_analytical",
        },
        "aggregate": agg_fifo_sync,
        "runs": fifo_sync_runs,
    }
    print(f"\n  FIFO-sync (extrapolated to {NUM_TASKS} tasks): "
          f"{fifo_sync_makespan:.2f}s")

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

    strategy_order = ["fifo_sync", "fifo_concurrent", "grouping", "grouping_jr", "full"]

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
