#!/usr/bin/env python3
"""
Experiment 5 — Makespan Scalability across Batch Sizes
======================================================
Thesis §4.6: Test how makespan scales as batch size grows for
different scheduling strategies.

Methodology:
  - Batch sizes: 10, 30, 60, 120, 200
  - 3 models (interleaved), single GPU
  - 3 strategies:
    a) FIFO-sync (calibrated analytical from exp1 measurement)
    b) Grouping only (real: semaphore, grouping ON, no JR/prefetch)
    c) Shared-Aware (real: grouping + JR + prefetch)
  - 3 runs per real data point, report mean ± std
  - Cold-start verified between every run

Expected:
  - FIFO-sync scales linearly with switches (calibrated extrapolation)
  - FIFO-concurrent benefits from implicit batching but no explicit grouping
  - Shared-Aware: grouping amortises switch cost over more tasks
  - The gap between strategies grows with batch size
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy,
    submit_batch,
    extract_task_metrics,
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

OUTDIR = RESULTS_ROOT / "exp5_scalability"
BATCH_SIZES = [10, 30, 60, 120, 200]

# Path to exp1 calibration data (run exp1 first!)
EXP1_CALIBRATION = RESULTS_ROOT / "exp1_strategy_makespan" / "fifo_sync_calibration.json"

STRATEGIES = [
    {
        "name": "grouping",
        "description": "Grouping only — semaphore, grouping ON, no JR, no prefetch",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "shared_aware",
        "description": "Shared-Aware — semaphore, grouping + JR + prefetch",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": True,
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
    },
]


# ── Workload ───────────────────────────────────────────────────────────
def generate_interleaved(n: int):
    """Round-robin S M L S M L ... across 3 models."""
    models = [MODEL_SMALL, MODEL_MEDIUM, MODEL_LARGE]
    tasks = []
    for i in range(n):
        m = models[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(make_task(m, f"Explain item {i + 1}.", f"task-{i}-{tag}"))
    return tasks


def load_fifo_calibration():
    """Load FIFO-sync calibration from exp1 results."""
    if not EXP1_CALIBRATION.exists():
        print("  [WARN] No FIFO-sync calibration found at "
              f"{EXP1_CALIBRATION}")
        print("  [WARN] Run exp1 first! Using fallback L_switch=103, L_infer=0.5")
        return {"avg_l_switch": 103.0, "avg_l_infer": 0.5}
    with open(EXP1_CALIBRATION) as f:
        return json.load(f)


def extrapolate_fifo_sync(calibration: dict, n_tasks: int) -> float:
    """Extrapolate FIFO-sync makespan for n interleaved tasks.
    Interleaved order → (n-1) model switches.
    """
    l_switch = calibration["avg_l_switch"]
    l_infer = calibration["avg_l_infer"]
    n_switches = n_tasks - 1
    return n_switches * l_switch + n_tasks * l_infer


# ── Main ───────────────────────────────────────────────────────────────
def run():
    print_header("EXP 5 — Makespan Scalability across Batch Sizes", {
        "Batch sizes": BATCH_SIZES,
        "Models": "3 (interleaved)",
        "Strategies": "FIFO-sync (calibrated), Grouping only (real), Shared-Aware (real)",
        "Runs per point": DEFAULT_RUNS,
    })

    all_results = {}

    # Load FIFO-sync calibration from exp1
    calibration = load_fifo_calibration()
    print(f"  FIFO-sync calibration: L_switch={calibration['avg_l_switch']:.2f}s, "
          f"L_infer={calibration['avg_l_infer']:.3f}s")

    # ── FIFO-sync calibrated analytical ───────────────────────────────
    sname = "fifo_sync"
    all_results[sname] = {}
    print_separator("Strategy: FIFO-SYNC (calibrated analytical)")
    for n in BATCH_SIZES:
        makespan = extrapolate_fifo_sync(calibration, n)
        throughput = n / makespan if makespan > 0 else 0
        runs = []
        for run_idx in range(DEFAULT_RUNS):
            run_result = {
                "run": run_idx + 1,
                "batch_size": n,
                "strategy": sname,
                "makespan": round(makespan, 2),
                "throughput": round(throughput, 2),
                "source": "calibrated_analytical",
                "batch_id": f"calibrated-fifo-sync-{n}-{run_idx + 1}",
                "completed": n,
                "failed": 0,
            }
            runs.append(run_result)
        agg = aggregate_runs(runs, key="makespan")
        tput_agg = aggregate_runs(runs, key="throughput")
        all_results[sname][n] = {
            "batch_size": n,
            "strategy": sname,
            "source": "calibrated_analytical",
            "makespan_agg": agg,
            "throughput_agg": tput_agg,
            "runs": runs,
        }
        print(f"  n={n:>4}: makespan={makespan:.1f}s (calibrated)  "
              f"tput={throughput:.2f}")

    # ── Real-cluster strategies ───────────────────────────────────────
    for strat in STRATEGIES:
        sname = strat["name"]
        all_results[sname] = {}
        print_separator(f"Strategy: {sname.upper()} — {strat['description']}")

        for n in BATCH_SIZES:
            print(f"\n  ── Batch size = {n} ──")

            runs = []
            for run_idx in range(DEFAULT_RUNS):
                print(f"    Run {run_idx + 1}/{DEFAULT_RUNS}:", end=" ")

                if not clear_deployments():
                    print("[WARN: cold-start failed]", end=" ")
                verify_cold_state()

                set_strategy(
                    strategy=strat["strategy"],
                    buffer_limit=strat["buffer_limit"],
                    enable_model_grouping=strat["enable_model_grouping"],
                    enable_johnsons_rule=strat["enable_johnsons_rule"],
                    enable_prefetch=strat["enable_prefetch"],
                    prefetch_threshold=strat["prefetch_threshold"],
                )

                tasks = generate_interleaved(n)
                batch = submit_batch(tasks, quiet=True)
                metrics = extract_task_metrics(batch["tasks_data"])
                makespan = batch["duration"]
                throughput = n / makespan if makespan > 0 else 0
                print(f"makespan={makespan:.1f}s  tput={throughput:.2f}")

                run_result = {
                    "run": run_idx + 1,
                    "batch_size": n,
                    "strategy": sname,
                    "makespan": makespan,
                    "throughput": round(throughput, 2),
                    **metrics,
                    "batch_id": batch["batch_id"],
                }
                runs.append(run_result)
                save_result(
                    run_result,
                    OUTDIR / f"{sname}_{n}tasks_run{run_idx + 1}.json"
                )

            agg = aggregate_runs(runs, key="makespan")
            tput_agg = aggregate_runs(runs, key="throughput")
            all_results[sname][n] = {
                "batch_size": n,
                "strategy": sname,
                "makespan_agg": agg,
                "throughput_agg": tput_agg,
                "runs": runs,
            }
            save_result(
                all_results[sname][n],
                OUTDIR / f"{sname}_{n}tasks_aggregate.json"
            )
            print(f"  Aggregate: makespan={agg['mean']:.2f}±{agg['std']:.2f}s  "
                  f"tput={tput_agg['mean']:.2f}±{tput_agg['std']:.2f}")

    # ── Summary ────────────────────────────────────────────────────────
    save_result(all_results, OUTDIR / "summary.json")

    all_strat_names = ["fifo_sync", "grouping", "shared_aware"]

    print("\n" + "=" * 70)
    print("SUMMARY — Makespan by Strategy x Batch Size")
    print(f"{'Size':<8}", end="")
    for s in all_strat_names:
        print(f"  {s:<22}", end="")
    print()
    print("-" * 78)

    for n in BATCH_SIZES:
        row = f"{n:<8}"
        vals = {}
        for s in all_strat_names:
            r = all_results.get(s, {}).get(n)
            ms = r["makespan_agg"]["mean"] if r else float("nan")
            std = r["makespan_agg"]["std"] if r else 0
            vals[s] = ms
            source = "c" if s == "fifo_sync" else ""
            row += f"  {ms:.1f}±{std:.1f}{source:>3}{'':>8}"
        print(row)

    print("\n  c = calibrated analytical (from exp1 FIFO-sync measurement)")
    print("=" * 70)


if __name__ == "__main__":
    run()
