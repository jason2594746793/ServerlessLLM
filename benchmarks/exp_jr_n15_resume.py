#!/usr/bin/env python3
"""
Resume n=15 from perm 3 (perms 1-2 already saved).
Uses reduced gpu_memory_utilization (0.70) to fit on shared cluster.
"""
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_common import (
    set_strategy, submit_batch, extract_task_metrics, save_result,
    make_task, clear_deployments, verify_cold_state,
    register_model_deployment, aggregate_runs, print_separator,
    evict_model_page_cache, MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE,
    RESULTS_ROOT,
)

OUTDIR = RESULTS_ROOT / "exp4_task_sensitivity" / "n15"
N_PER_MODEL = 15
EXP_MODELS = [MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE]
NUM_RUNS = 3
ALL_PERMS = list(itertools.permutations(EXP_MODELS))
MODEL_LABELS = {MODEL_SMALL: "0.6B", MODEL_LARGE: "8B", MODEL_XLARGE: "32B"}

# Start from perm 3 (0-indexed: perm_idx 2)
START_PERM_IDX = 2  # perms 1-2 already done


def order_str(order):
    return " → ".join(MODEL_LABELS[m] for m in order)


def generate_workload():
    tasks = []
    total = N_PER_MODEL * len(EXP_MODELS)
    for i in range(total):
        model = EXP_MODELS[i % len(EXP_MODELS)]
        tasks.append(make_task(model, f"Explain concept {i+1} in one sentence.",
                               f"task-{i}-{MODEL_LABELS[model]}"))
    return tasks


def cold_start():
    for m in EXP_MODELS:
        evict_model_page_cache(m)
    if not clear_deployments():
        print("  [WARN] Cold-start verification failed, proceeding anyway")
    if not verify_cold_state():
        print("  [WARN] System may not be fully cold")
    # Use lower gpu_memory_utilization to fit with partial GPU occupancy
    # (GPUs 2,3 have ~6-7GB from other users; 0.70 × 24GB = 16.8GB < 17.7GB free)
    register_model_deployment(MODEL_LARGE, {
        "gpu_memory_utilization": 0.70,
        "enforce_eager": True,
    })
    register_model_deployment(MODEL_XLARGE, {
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.70,
        "enforce_eager": True,
    })


def run_perm(perm, perm_idx):
    label = order_str(perm)
    print_separator(f"n=15 Perm {perm_idx+1}/6 — {label}")
    runs = []
    for run_idx in range(NUM_RUNS):
        cold_start()
        set_strategy(
            strategy="semaphore", buffer_limit=8,
            enable_model_grouping=True, enable_johnsons_rule=False,
            enable_prefetch=True, prefetch_threshold=0.0,
            force_group_order=list(perm),
        )
        batch = submit_batch(generate_workload())
        makespan = batch["duration"]
        metrics = extract_task_metrics(batch["tasks_data"])
        run_result = {
            "run": run_idx + 1, "n_per_model": N_PER_MODEL,
            "perm_idx": perm_idx, "order": label,
            "makespan": makespan, "throughput": round(N_PER_MODEL*3/makespan, 2),
            "batch_id": batch["batch_id"], **metrics,
        }
        runs.append(run_result)
        save_result(run_result, OUTDIR / f"perm{perm_idx+1}_run{run_idx+1}.json")
        print(f"    Run {run_idx+1}: {makespan:.1f}s")
    agg = aggregate_runs(runs, key="makespan")
    save_result({"order": label, "n_per_model": N_PER_MODEL, "aggregate": agg, "runs": runs},
                OUTDIR / f"perm{perm_idx+1}_aggregate.json")
    print(f"  → {label}: {agg['mean']:.1f} ± {agg['std']:.1f}s")
    return agg


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Resuming n=15 from perm {START_PERM_IDX+1} (perms 1-{START_PERM_IDX} already done)")

    for perm_idx in range(START_PERM_IDX, 6):
        perm = ALL_PERMS[perm_idx]
        run_perm(perm, perm_idx)

    # Now regenerate summary.json with all 6 perms
    JR_ORDER_N5  = (MODEL_SMALL, MODEL_LARGE, MODEL_XLARGE)
    JR_ORDER_N30 = (MODEL_SMALL, MODEL_XLARGE, MODEL_LARGE)
    results = []
    for perm_idx, perm in enumerate(ALL_PERMS):
        f = OUTDIR / f"perm{perm_idx+1}_aggregate.json"
        if f.exists():
            import json
            d = json.loads(f.read_text())
            results.append({
                "perm_idx": perm_idx, "order": d["order"],
                "is_jr_n5": list(perm) == list(JR_ORDER_N5),
                "is_jr_n30": list(perm) == list(JR_ORDER_N30),
                "aggregate": d["aggregate"],
            })
    results_sorted = sorted(results, key=lambda r: r["aggregate"]["mean"])
    jr_n5_rank  = next((i+1 for i,r in enumerate(results_sorted) if r["is_jr_n5"]), None)
    jr_n30_rank = next((i+1 for i,r in enumerate(results_sorted) if r["is_jr_n30"]), None)
    save_result({"n_per_model": N_PER_MODEL, "ranked": results_sorted,
                 "jr_n5_rank": jr_n5_rank, "jr_n30_rank": jr_n30_rank},
                OUTDIR / "summary.json")

    print(f"\nn=15 COMPLETE: JR@n5 rank={jr_n5_rank}/6, JR@n30 rank={jr_n30_rank}/6")
    for i, r in enumerate(results_sorted):
        jr5=" [JR@n5]" if r["is_jr_n5"] else ""
        jr30=" [JR@n30]" if r["is_jr_n30"] else ""
        print(f"  {i+1}. {r['order']:25} {r['aggregate']['mean']:.1f} ± {r['aggregate']['std']:.1f}s{jr5}{jr30}")


if __name__ == "__main__":
    run()
