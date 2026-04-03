#!/usr/bin/env python3
"""
Exp W1 — Manual FIFO Switch Cost Measurement
=============================================
Directly measures per-switch cost by orchestrating a tight A→B→A→B→A→B loop:

  cold_start()
  for cycle in range(N_CYCLES):   # N_CYCLES = 3
      submit 1 task to MODEL_A; wait; record time
      clear_deployments()          # force eviction
      submit 1 task to MODEL_B; wait; record time
      clear_deployments()

This bypasses ServerlessLLM's broken FIFO router (which cannot handle live
model eviction during concurrent dispatch) and directly measures the cold
per-switch cost for 0.6B↔8B transitions.

Design:
  - N_CYCLES = 3 → 6 tasks total, 5 switches (A→B ×3, B→A ×2)
  - Models: 0.6B and 8B only (no 32B, no TP=4)
  - Each switch = clear_deployments() + load next model cold
  - Implied per-switch cost = (total_time - 6*avg_infer) / 5

Cross-validates:
  - Exp 2 measured: 0.6B→8B = 61.9s, 8B→0.6B ≈ 34–37s, weighted mean ≈ 48s
  - Exp 1 analytical 3-model avg: 83.3s (includes expensive 8B→32B = 115.2s)

GPU requirement: 2 GPUs (GPUs 2,3 recommended if 0,1 occupied).
Runtime: ~6×30s + 5×50s ≈ 6 minutes.
Results saved to: results/exp_fifo_manual/
"""

import sys
import time
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

OUTDIR = RESULTS_ROOT / "exp_fifo_manual"
MODEL_A = MODEL_SMALL   # 0.6B
MODEL_B = MODEL_LARGE   # 8B
N_CYCLES = 3            # 6 tasks, 5 switches: A→B×3, B→A×2
MODEL_LABELS = {MODEL_A: "0.6B", MODEL_B: "8B"}


def cold_start():
    """Full cold start: evict caches, clear deployments, verify."""
    for m in [MODEL_A, MODEL_B]:
        evict_model_page_cache(m)
    if not clear_deployments():
        print("  [WARN] Cold-start verification failed, proceeding anyway")
    if not verify_cold_state():
        print("  [WARN] System may not be fully cold")


def submit_one(model: str, task_id: str) -> tuple[float, float]:
    """Submit a single task to model, return (makespan, infer_latency)."""
    task = make_task(model, "Explain neural networks in one sentence.", task_id)
    batch = submit_batch([task])
    makespan = batch["duration"]
    metrics = extract_task_metrics(batch["tasks_data"])
    infer = metrics["latencies"][0] if metrics.get("latencies") else makespan
    return makespan, infer


def run():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print_header("Exp W1 — Manual FIFO Switch Cost Measurement", {
        "Design":    f"0.6B → 8B → 0.6B → 8B → ... ({N_CYCLES} cycles = {N_CYCLES*2} tasks, {N_CYCLES*2-1} switches)",
        "Strategy":  "No grouping/prefetch. clear_deployments() between every task.",
        "Expected":  "~48s/switch avg (0.6B→8B: 62s, 8B→0.6B: ~34s, weighted ~48s)",
        "Purpose":   "Cross-validate Exp 1 analytical 83.3s FIFO estimate (3-model avg includes 8B→32B=115.2s)",
        "GPUs":      "2 (0.6B and 8B each fit on 1 GPU, no TP needed)",
    })

    # Configure strategy: single-task buffer, no grouping, no prefetch
    set_strategy(
        strategy="semaphore",
        buffer_limit=1,
        enable_model_grouping=False,
        enable_johnsons_rule=False,
        enable_prefetch=False,
        prefetch_threshold=0.0,
    )

    cold_start()

    sequence = []  # list of {model, task_id, load_time, infer_latency}
    t_experiment_start = time.time()

    for cycle_idx in range(N_CYCLES):
        for step, model in enumerate([MODEL_A, MODEL_B]):
            task_id = f"cycle{cycle_idx+1}-{'A' if step==0 else 'B'}"
            label = MODEL_LABELS[model]
            print(f"\n  ── Cycle {cycle_idx+1}/{N_CYCLES} Step {'A' if step==0 else 'B'} ({label}) ──")

            t_load_start = time.time()
            makespan, infer = submit_one(model, task_id)
            t_load_end = time.time()

            # load_time = total task time (includes vLLM init + inference)
            load_time = makespan
            print(f"  Task makespan: {makespan:.1f}s  |  Inference: {infer:.3f}s")

            sequence.append({
                "cycle": cycle_idx + 1,
                "step": "A" if step == 0 else "B",
                "model": label,
                "task_id": task_id,
                "makespan": round(makespan, 2),
                "infer_latency": round(infer, 3),
            })

            # Clear between tasks (force model eviction)
            if not (cycle_idx == N_CYCLES - 1 and step == 1):
                print(f"  Clearing deployments (evicting {label})...")
                for m in [MODEL_A, MODEL_B]:
                    evict_model_page_cache(m)
                ok = clear_deployments()
                if not ok:
                    print("  [WARN] clear_deployments timed out, continuing")

    t_total = time.time() - t_experiment_start

    # ── Compute switch costs ─────────────────────────────────────────────────
    n_tasks = N_CYCLES * 2
    n_switches = n_tasks - 1  # 5

    all_makespans = [s["makespan"] for s in sequence]
    all_infers = [s["infer_latency"] for s in sequence]
    avg_infer = statistics.mean(all_infers)
    total_task_time = sum(all_makespans)

    # Separate A→B and B→A switches (by order in sequence)
    # Switches happen between consecutive tasks in sequence
    a_to_b_times = []  # 0.6B→8B: cycles where B follows A
    b_to_a_times = []  # 8B→0.6B: cycles where A follows B (next cycle)

    for i in range(len(sequence) - 1):
        cur = sequence[i]
        nxt = sequence[i + 1]
        # Each task's makespan includes: load time + inference
        # The "switch cost" to get to next model = nxt.makespan - nxt.infer_latency
        # (i.e., the cold-load overhead in the next task)
        switch_cost = nxt["makespan"] - nxt["infer_latency"]
        if cur["step"] == "A" and nxt["step"] == "B":
            a_to_b_times.append(switch_cost)
        elif cur["step"] == "B" and nxt["step"] == "A":
            b_to_a_times.append(switch_cost)

    avg_a_to_b = statistics.mean(a_to_b_times) if a_to_b_times else 0
    avg_b_to_a = statistics.mean(b_to_a_times) if b_to_a_times else 0

    # Overall implied switch cost (traditional method: global)
    # sum of all per-switch costs = total_task_time - n_tasks*avg_infer
    implied_switch_global = (total_task_time - n_tasks * avg_infer) / n_switches

    # Weighted mean from individual switch costs
    n_atob = len(a_to_b_times)
    n_btoa = len(b_to_a_times)
    weighted_switch = (
        (avg_a_to_b * n_atob + avg_b_to_a * n_btoa) / (n_atob + n_btoa)
        if (n_atob + n_btoa) > 0 else implied_switch_global
    )

    result = {
        "experiment": "exp_fifo_manual",
        "n_cycles": N_CYCLES,
        "n_tasks": n_tasks,
        "n_switches": n_switches,
        "sequence": sequence,
        "total_experiment_time": round(t_total, 1),
        "total_task_time": round(total_task_time, 1),
        "avg_inference_latency": round(avg_infer, 3),
        "implied_switch_cost_global": round(implied_switch_global, 1),
        "avg_switch_0.6B_to_8B": round(avg_a_to_b, 1),
        "avg_switch_8B_to_0.6B": round(avg_b_to_a, 1),
        "weighted_switch_cost": round(weighted_switch, 1),
        "n_a_to_b": n_atob,
        "n_b_to_a": n_btoa,
    }
    save_result(result, OUTDIR / "result.json")

    print("\n" + "=" * 64)
    print("RESULTS — Manual FIFO Switch Cost")
    print(f"  {'Metric':<35} {'Value':<12} {'Reference'}")
    print("  " + "-" * 60)
    print(f"  {'0.6B→8B switch cost (avg):':<35} {avg_a_to_b:<12.1f} Exp 2: 61.9s")
    print(f"  {'8B→0.6B switch cost (avg):':<35} {avg_b_to_a:<12.1f} Exp 2: ~34s")
    print(f"  {'Weighted mean switch cost:':<35} {weighted_switch:<12.1f} Exp 2 weighted: ~48s")
    print(f"  {'Implied switch (global method):':<35} {implied_switch_global:<12.1f} (alternative)")
    print(f"  {'Avg inference latency:':<35} {avg_infer:<12.3f}")
    print("  " + "-" * 60)
    print(f"  Exp 1 analytical 3-model avg:     83.3s   (includes 8B→32B=115.2s)")
    print(f"  This experiment: 2-model only (0.6B+8B), excludes 8B→32B transitions")
    print("=" * 64)

    if abs(weighted_switch - 48) < 15:
        print(f"\n  CROSS-VALIDATION: PASS ({weighted_switch:.1f}s vs ~48s expected, "
              f"{abs(weighted_switch-48)/48*100:.0f}% diff)")
    else:
        print(f"\n  CROSS-VALIDATION: NOTE — {weighted_switch:.1f}s vs ~48s expected "
              f"({abs(weighted_switch-48)/48*100:.0f}% diff). Check vLLM init overhead.")


if __name__ == "__main__":
    run()
