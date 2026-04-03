#!/usr/bin/env python3
"""
Exp 3a — Scalability anchor at n=99 (small batch)
===================================================
Runs Grouping-only and Shared-Aware at n=99 (33 per model × 3 models)
to complement the n=501 data from Exp 1.

Together these two measured points validate the analytical scaling model.

Run with:
    python benchmarks/exp3a_scalability_anchor.py
"""

import sys
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

OUTDIR = RESULTS_ROOT / "exp3a_scalability"
OUTDIR.mkdir(parents=True, exist_ok=True)

NUM_TASKS = 99  # 33 per model × 3 models
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]

CONFIGS = [
    {
        "name": "grouping",
        "description": "Grouping only — no prefetch, no JR",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": False,
        "enable_prefetch": False,
        "prefetch_threshold": 0.0,
    },
    {
        "name": "shared_aware",
        "description": "Shared-Aware — grouping + JR + prefetch",
        "strategy": "semaphore",
        "buffer_limit": 8,
        "enable_model_grouping": True,
        "enable_johnsons_rule": True,
        "enable_prefetch": True,
        "prefetch_threshold": 0.0,
    },
]


def generate_workload():
    tasks = []
    for i in range(NUM_TASKS):
        m = EXP_MODELS[i % 3]
        tag = ["S", "M", "L"][i % 3]
        tasks.append(make_task(m, f"Explain concept {i + 1} in one sentence.", f"t-{i}-{tag}"))
    return tasks


def run():
    print_header("EXP 3a — Scalability anchor n=99 (3 models: 0.6B, 8B, 32B)", {
        "Tasks": f"{NUM_TASKS} (33 per model, interleaved)",
        "Configs": "Grouping, Shared-Aware",
        "Runs": DEFAULT_RUNS,
        "Purpose": "Anchor point for scalability analysis; n=501 data from Exp 1",
    })

    all_results = {}

    for cfg in CONFIGS:
        name = cfg["name"]
        print_separator(f"Config: {cfg['description']}")

        runs = []
        for run_idx in range(DEFAULT_RUNS):
            print(f"\n  ── Run {run_idx + 1}/{DEFAULT_RUNS} ──")

            if not clear_deployments():
                print("  [WARN] Cold-start verification failed")
            if not verify_cold_state():
                print("  [WARN] System may not be fully cold")
            for m in EXP_MODELS:
                evict_model_page_cache(m)
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
            batch = submit_batch(tasks)
            metrics = extract_task_metrics(batch["tasks_data"])
            makespan = batch["duration"]
            failed_count = batch.get("failed_count", 0)
            completed = NUM_TASKS - failed_count
            throughput = completed / makespan if makespan > 0 else 0

            run_result = {
                "run": run_idx + 1,
                "batch_size": NUM_TASKS,
                "makespan": makespan,
                "throughput": round(throughput, 2),
                "completed": completed,
                "failed": failed_count,
                **metrics,
                "batch_id": batch["batch_id"],
            }
            runs.append(run_result)
            save_result(run_result, OUTDIR / f"{name}_n{NUM_TASKS}_run{run_idx + 1}.json")

            print(f"  Makespan:   {makespan:.1f}s")
            print(f"  Throughput: {throughput:.2f} req/s")
            print(f"  Completed:  {completed}/{NUM_TASKS}")

        agg = aggregate_runs(runs, key="makespan")
        all_results[name] = {"config": cfg, "aggregate": agg, "runs": runs}
        save_result(all_results[name], OUTDIR / f"{name}_n{NUM_TASKS}_aggregate.json")
        print(f"\n  {name} n={NUM_TASKS}: {agg['mean']:.1f} ± {agg['std']:.1f}s")

    print("\n\n=== SCALABILITY SUMMARY ===")
    print(f"{'Config':<20} {'n=99 (measured)':<22} {'n=501 (Exp 1)':<22}")
    print("-" * 64)

    exp1_base = RESULTS_ROOT / "exp1_strategy_makespan"
    for name in ["grouping", "shared_aware"]:
        n99_agg = all_results.get(name, {}).get("aggregate", {})
        exp1_name = "full" if name == "shared_aware" else "grouping"
        exp1_path = exp1_base / f"{exp1_name}_aggregate.json"
        n501_str = "N/A"
        if exp1_path.exists():
            import json
            d = json.load(open(exp1_path))
            agg501 = d.get("aggregate", {})
            n501_str = f"{agg501.get('mean', '?'):.1f} ± {agg501.get('std', '?'):.1f}s"
        n99_str = f"{n99_agg.get('mean', '?'):.1f} ± {n99_agg.get('std', '?'):.1f}s" if n99_agg else "N/A"
        print(f"{name:<20} {n99_str:<22} {n501_str:<22}")


if __name__ == "__main__":
    run()
