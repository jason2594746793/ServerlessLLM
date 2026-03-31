#!/usr/bin/env python3
"""
Analytical model + publication-quality figures for dissertation:
"Efficient Scheduling for Batched AI Jobs"

Generates all figures as PDF + PNG using mock_params.py as the
single source of truth for system parameters.

Three baselines:
  - FIFO: tasks submitted in alternating ABABAB model order
  - Random: tasks in shuffled random order
  - Proposed: model grouping + Johnson's Rule + prefetch

Usage:
    python benchmarks/plot_analytical.py
    python benchmarks/plot_analytical.py --exp 1 4 8m ig
"""

import math
import random
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from mock_params import (
    NUM_GPUS, NVME_READ_GBps, PCIE_BW_GBps,
    MODELS, MODEL_LOAD_TIMES, PCIE_TRANSFER_TIMES,
    MODEL_INFERENCE_TIMES, MAX_CONCURRENT_PER_GPU,
    MODEL_UNLOAD_TIME, PREFETCH_OVERLAP_FRACTION,
    STANDARD_WORKLOAD,
)

# ── Style: IEEE/ACM conference ──────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIGURES = Path(__file__).parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Grayscale-friendly palette
COLORS = ["#2c3e50", "#e74c3c", "#27ae60", "#f39c12", "#8e44ad"]
HATCHES = ["", "//", "\\\\", "xx", ".."]


def save_fig(fig, name: str):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    print(f"  Saved: figures/{name}.pdf + .png")
    plt.close(fig)


def add_value_labels(ax, bars, fmt="{:.1f}s", offset=0.5, fontsize=8):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                fmt.format(v), ha="center", va="bottom", fontsize=fontsize)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def inference_time(model: str, num_tasks: int, num_gpus: int = 1) -> float:
    """Time to run num_tasks on a model with concurrent batching."""
    lat = MODEL_INFERENCE_TIMES[model]
    concurrent = MAX_CONCURRENT_PER_GPU * num_gpus
    batches = math.ceil(num_tasks / concurrent)
    return batches * lat


def johnsons_rule_order(groups: Dict[str, int]) -> List[str]:
    """Apply Johnson's Rule to order model groups.

    Each group is a job with:
      I_i = MODEL_LOAD_TIMES (actual SSD→CPU load)
      P_i = inference_time(model, num_tasks)   (compute stage)

    Returns ordered list of model names.
    """
    s1, s2 = [], []
    for model, num_tasks in groups.items():
        i_time = MODEL_LOAD_TIMES[model] - PCIE_TRANSFER_TIMES.get(model, 0)
        p_time = inference_time(model, num_tasks)
        if i_time <= p_time:
            s1.append((model, i_time, p_time))
        else:
            s2.append((model, i_time, p_time))
    s1.sort(key=lambda x: x[1])        # ascending I_i
    s2.sort(key=lambda x: x[2], reverse=True)  # descending P_i
    return [m for m, _, _ in s1] + [m for m, _, _ in s2]


def _run_task_sequence(task_list: List[str]) -> float:
    """Simulate sequential execution of a task list with concurrent batching.

    Consecutive tasks for the same model are batched together (up to
    MAX_CONCURRENT_PER_GPU per batch).  A model switch pays the full
    cold-load cost.
    """
    if not task_list:
        return 0.0

    t = 0.0
    loaded = None
    i = 0
    while i < len(task_list):
        m = task_list[i]
        if loaded != m:
            t += MODEL_LOAD_TIMES[m]
            loaded = m
        # Count consecutive same-model tasks
        run_len = 1
        while i + run_len < len(task_list) and task_list[i + run_len] == m:
            run_len += 1
        t += inference_time(m, run_len)
        i += run_len
    return t


def sim_fifo_makespan(task_counts: Dict[str, int]) -> float:
    """FIFO baseline: alternating ABABAB... model order.

    Tasks arrive interleaved across models (e.g. A,B,C,A,B,C,...).
    Consecutive same-model tasks are batched, but the alternating
    pattern forces a model switch after every single-task run.
    """
    models = list(task_counts.keys())
    max_count = max(task_counts.values())
    task_list = []
    for i in range(max_count):
        for m in models:
            if i < task_counts[m]:
                task_list.append(m)
    return _run_task_sequence(task_list)


def sim_random_makespan(task_counts: Dict[str, int], seed: int = 42) -> float:
    """Random baseline: shuffled task sequence.

    Tasks are randomly shuffled.  Consecutive same-model runs are
    batched, so random order produces some natural grouping.
    """
    rng = random.Random(seed)
    task_list = []
    for m, n in task_counts.items():
        task_list.extend([m] * n)
    rng.shuffle(task_list)
    return _run_task_sequence(task_list)


def _flowshop_makespan(order: List[str], task_counts: Dict[str, int]) -> float:
    """Two-machine flow-shop makespan with proper SSD pipeline tracking.

    Machine 1 = SSD (loads models sequentially to CPU pinned memory)
    Machine 2 = GPU (processes model groups sequentially)

    The SSD loads models one after another without waiting for the GPU.
    The GPU can start model i only after both:
      - SSD finishes loading model i, AND
      - GPU finishes processing model i-1
    A PCIe transfer (CPU→GPU) is added before each GPU processing step.
    """
    n = len(order)
    if n == 0:
        return 0.0

    # Machine 1 (SSD) completion times
    ssd_end = [0.0] * n
    ssd_end[0] = MODEL_LOAD_TIMES[order[0]]  # first model: full cold load
    for i in range(1, n):
        ssd_to_cpu = MODEL_LOAD_TIMES[order[i]] - PCIE_TRANSFER_TIMES.get(order[i], 0)
        ssd_end[i] = ssd_end[i - 1] + ssd_to_cpu

    # Machine 2 (GPU) completion times
    gpu_end = [0.0] * n
    gpu_end[0] = ssd_end[0] + inference_time(order[0], task_counts[order[0]])
    for i in range(1, n):
        gpu_start = max(ssd_end[i], gpu_end[i - 1]) + PCIE_TRANSFER_TIMES.get(order[i], 0)
        gpu_end[i] = gpu_start + inference_time(order[i], task_counts[order[i]])

    return gpu_end[-1]


def sim_grouped_makespan(task_counts: Dict[str, int], prefetch: bool = False) -> float:
    """Grouped: sorted by model, sequential groups, optional prefetch.

    With prefetch=True, Johnson's Rule determines group order and
    the SSD pipeline overlaps loading with GPU compute (flow-shop model).
    Without prefetch, each model switch pays the full cold-load cost.
    """
    order = johnsons_rule_order(task_counts) if prefetch else list(task_counts.keys())
    order = [m for m in order if task_counts.get(m, 0) > 0]
    if not order:
        return 0.0

    if prefetch:
        return _flowshop_makespan(order, task_counts)

    # No prefetch: sequential cold loads
    t = MODEL_LOAD_TIMES[order[0]]
    for i, m in enumerate(order):
        t += inference_time(m, task_counts[m])
        if i < len(order) - 1:
            t += MODEL_LOAD_TIMES[order[i + 1]]
    return t


def sim_proposed_makespan(task_counts: Dict[str, int], num_gpus: int = 2) -> float:
    """Proposed: Johnson's Rule + grouping + prefetch + multi-GPU.

    Uses the two-machine flow-shop model for prefetch scheduling.
    Multi-GPU strategies:
      A. Greedy LPT assigns whole model groups to GPUs.
      B. Task-level split: each model's tasks are divided across GPUs.
    Each GPU runs its own flow-shop schedule independently.
    """
    active = {m: n for m, n in task_counts.items() if n > 0}
    if not active:
        return 0.0

    single_gpu_ms = sim_grouped_makespan(active, prefetch=True)
    if num_gpus == 1:
        return single_gpu_ms

    def _per_gpu(q):
        order = johnsons_rule_order(q)
        order = [m for m in order if q.get(m, 0) > 0]
        return _flowshop_makespan(order, q) if order else 0.0

    # Strategy A: model-level LPT
    sorted_groups = sorted(active.items(),
                           key=lambda x: MODEL_LOAD_TIMES[x[0]] + inference_time(x[0], x[1]),
                           reverse=True)
    gpu_work = [0.0] * min(num_gpus, len(active))
    gpu_queues: List[Dict[str, int]] = [{} for _ in range(len(gpu_work))]
    for model, ntasks in sorted_groups:
        min_gpu = gpu_work.index(min(gpu_work))
        gpu_queues[min_gpu][model] = ntasks
        gpu_work[min_gpu] += MODEL_LOAD_TIMES[model] + inference_time(model, ntasks)

    gpu_times = [_per_gpu(q) for q in gpu_queues if q]
    multi_gpu_ms = max(gpu_times) if gpu_times else single_gpu_ms

    # Strategy B: task-level split
    ng_b = num_gpus
    gpu_queues_b: List[Dict[str, int]] = [{} for _ in range(ng_b)]
    for m, n in active.items():
        per_gpu = n // ng_b
        remainder = n % ng_b
        for g in range(ng_b):
            count = per_gpu + (1 if g < remainder else 0)
            if count > 0:
                gpu_queues_b[g][m] = count

    gpu_times_b = [_per_gpu(q) for q in gpu_queues_b if q]
    split_gpu_ms = max(gpu_times_b) if gpu_times_b else single_gpu_ms

    return min(single_gpu_ms, multi_gpu_ms, split_gpu_ms)


# ════════════════════════════════════════════════════════════════════════
# Exp 1: Strategy Comparison (FIFO vs Random vs Grouping vs Proposed)
# ════════════════════════════════════════════════════════════════════════

def sim_exp1():
    """100 tasks, 2 models (50 each), interleaved submission."""
    task_counts = {"small": 50, "medium": 50}

    fifo_ms = sim_fifo_makespan(task_counts)
    random_ms = sim_random_makespan(task_counts)
    grouped_ms = sim_grouped_makespan(task_counts, prefetch=False)
    proposed_ms = sim_proposed_makespan(task_counts, num_gpus=1)

    return {
        "fifo":     {"makespan": fifo_ms},
        "random":   {"makespan": random_ms},
        "grouping": {"makespan": grouped_ms},
        "proposed": {"makespan": proposed_ms},
    }


def plot_exp1():
    data = sim_exp1()
    names = ["fifo", "random", "grouping", "proposed"]
    labels = ["FIFO\n(Alternating)", "Random\n(Shuffled)", "Model\nGrouping",
              "Proposed\n(Johnson+Prefetch)"]
    makespans = [data[n]["makespan"] for n in names]
    throughputs = [100 / m for m in makespans]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    bars = ax1.bar(labels, makespans, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:4]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan by Strategy")
    add_value_labels(ax1, bars)

    bars2 = ax2.bar(labels, throughputs, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:4]):
        bar.set_hatch(h)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput by Strategy")
    add_value_labels(ax2, bars2, fmt="{:.2f}", offset=0.01)

    fig.suptitle("Exp 1: Scheduling Strategy Impact\n"
                 "(100 Tasks — Qwen3-0.6B + Qwen3-8B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    save_fig(fig, "exp1_strategy_makespan")


# ════════════════════════════════════════════════════════════════════════
# Exp 2: Model Switching Overhead
# ════════════════════════════════════════════════════════════════════════

def sim_exp2():
    """20 tasks, 2 models. Quantify switching cost."""
    n = 20
    models = ["small", "medium"]
    avg_load = (MODEL_LOAD_TIMES["small"] + MODEL_LOAD_TIMES["medium"]) / 2

    single_a = MODEL_LOAD_TIMES["small"] + inference_time("small", n)
    single_b = MODEL_LOAD_TIMES["medium"] + inference_time("medium", n)

    grouped = (MODEL_LOAD_TIMES["small"] + inference_time("small", n // 2)
               + MODEL_LOAD_TIMES["medium"] + inference_time("medium", n // 2))

    num_switches = 19
    alternating = (MODEL_LOAD_TIMES["small"]
                   + num_switches * (MODEL_UNLOAD_TIME + avg_load)
                   + n * (MODEL_INFERENCE_TIMES["small"] + MODEL_INFERENCE_TIMES["medium"]) / 2)

    return {
        "single_a":    {"makespan": single_a, "label": "Single A\n(0 switches)"},
        "single_b":    {"makespan": single_b, "label": "Single B\n(0 switches)"},
        "grouped":     {"makespan": grouped, "label": "Grouped\n(1 switch)"},
        "alternating": {"makespan": alternating, "label": f"Alternating\n({num_switches} switches)"},
    }


def plot_exp2():
    data = sim_exp2()
    names = ["single_a", "single_b", "grouped", "alternating"]
    labels = [data[n]["label"] for n in names]
    makespans = [data[n]["makespan"] for n in names]

    load_times = [
        MODEL_LOAD_TIMES["small"],
        MODEL_LOAD_TIMES["medium"],
        MODEL_LOAD_TIMES["small"] + MODEL_LOAD_TIMES["medium"],
        MODEL_LOAD_TIMES["small"] + 19 * (MODEL_UNLOAD_TIME + (MODEL_LOAD_TIMES["small"] + MODEL_LOAD_TIMES["medium"]) / 2),
    ]
    inf_times = [m - l for m, l in zip(makespans, load_times)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    bars_load = ax1.bar(labels, load_times, color="#e74c3c", edgecolor="black",
                        linewidth=0.5, label="Model Loading")
    bars_inf = ax1.bar(labels, inf_times, bottom=load_times, color="#27ae60",
                       edgecolor="black", linewidth=0.5, label="Inference")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("(a) Time Breakdown")
    ax1.legend(fontsize=8)
    for bar, v in zip(bars_inf, makespans):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + 2, f"{v:.0f}s", ha="center", va="bottom", fontsize=8)

    load_pcts = [l / m * 100 for l, m in zip(load_times, makespans)]
    bars3 = ax2.bar(labels, load_pcts, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars3, HATCHES[:4]):
        bar.set_hatch(h)
    ax2.set_ylabel("Loading Time (%)")
    ax2.set_title("(b) Loading Overhead")
    ax2.set_ylim(0, 105)
    add_value_labels(ax2, bars3, fmt="{:.0f}%", offset=1)

    fig.suptitle("Exp 2: Model Switching Overhead\n"
                 "(20 Tasks — Qwen3-0.6B + Qwen3-8B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp2_model_switching")


# ════════════════════════════════════════════════════════════════════════
# Exp 3: Concurrency Mode Comparison
# ════════════════════════════════════════════════════════════════════════

def sim_exp3():
    """60 tasks, 2 models, compare sync vs chunked vs semaphore."""
    tasks_per_model = 30
    load_total = MODEL_LOAD_TIMES["small"] + MODEL_LOAD_TIMES["medium"]

    sync_inf = (tasks_per_model * MODEL_INFERENCE_TIMES["small"]
                + tasks_per_model * MODEL_INFERENCE_TIMES["medium"])
    sync_makespan = load_total + sync_inf

    chunk_size = 10
    chunks_small = math.ceil(tasks_per_model / chunk_size)
    chunks_medium = math.ceil(tasks_per_model / chunk_size)
    chunked_inf = (chunks_small * MODEL_INFERENCE_TIMES["small"]
                   + chunks_medium * MODEL_INFERENCE_TIMES["medium"])
    chunked_makespan = load_total + chunked_inf

    sem_inf = inference_time("small", tasks_per_model) + inference_time("medium", tasks_per_model)
    sem_makespan = load_total + sem_inf

    return {
        "sync":      {"makespan": sync_makespan},
        "chunked":   {"makespan": chunked_makespan},
        "semaphore": {"makespan": sem_makespan},
    }


def plot_exp3():
    data = sim_exp3()
    names = ["sync", "chunked", "semaphore"]
    labels = ["Sync\n(Sequential)", "Chunked\n(Batch=10)", "Semaphore\n(Concurrent)"]
    makespans = [data[n]["makespan"] for n in names]
    throughputs = [60 / m for m in makespans]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    bars = ax1.bar(labels, makespans, color=COLORS[:3], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:3]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan")
    add_value_labels(ax1, bars)

    bars2 = ax2.bar(labels, throughputs, color=COLORS[:3], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:3]):
        bar.set_hatch(h)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput")
    add_value_labels(ax2, bars2, fmt="{:.2f}", offset=0.01)

    fig.suptitle("Exp 3: Concurrency Mode Comparison\n"
                 "(60 Tasks — Qwen3-0.6B + Qwen3-8B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp3_concurrency_mode")


# ════════════════════════════════════════════════════════════════════════
# Exp 4: Checkpoint Prefetch Effect
# ════════════════════════════════════════════════════════════════════════

def sim_exp4():
    """60 tasks, 3 models. Compare no-prefetch vs prefetch."""
    tasks_per_model = 20
    models = ["small", "medium", "large"]

    no_pf = sum(MODEL_LOAD_TIMES[m] + inference_time(m, tasks_per_model) for m in models)

    eager = MODEL_LOAD_TIMES[models[0]]
    for i, m in enumerate(models):
        eager += inference_time(m, tasks_per_model)
        if i < len(models) - 1:
            eager += PCIE_TRANSFER_TIMES[models[i + 1]]

    return {
        "no_prefetch":     {"makespan": no_pf},
        "prefetch_eager":  {"makespan": eager},
    }


def plot_exp4():
    data = sim_exp4()
    names = ["no_prefetch", "prefetch_eager"]
    labels = ["No Prefetch\n(Cold Switch)", "Eager Prefetch\n(PCIe Only)"]
    makespans = [data[n]["makespan"] for n in names]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, makespans, color=[COLORS[0], COLORS[2]], edgecolor="black", linewidth=0.5)
    bars[0].set_hatch(HATCHES[0])
    bars[1].set_hatch(HATCHES[2])
    ax.set_ylabel("Makespan (s)")
    add_value_labels(ax, bars)

    speedup = makespans[0] / makespans[1]
    ax.text(0.5, max(makespans) * 0.5, f"Speedup: {speedup:.1f}x",
            ha="center", fontsize=11, fontweight="bold", color=COLORS[2],
            transform=ax.get_xaxis_transform())

    fig.suptitle("Exp 4: Checkpoint Prefetch Effect\n"
                 "(60 Tasks — Qwen3-0.6B / Qwen3-8B / Qwen2.5-32B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp4_prefetch")


# ════════════════════════════════════════════════════════════════════════
# Exp 5a: Standard Workload (500 tasks, 3 models)
# ════════════════════════════════════════════════════════════════════════

def sim_exp5_standard():
    """500 tasks: 40% small, 30% medium, 20% large, 10% xlarge — FIFO vs Random vs Grouped vs Proposed."""
    task_counts = {"small": 200, "medium": 150, "large": 100, "xlarge": 50}

    fifo_ms = sim_fifo_makespan(task_counts)
    random_ms = sim_random_makespan(task_counts)
    grouped_ms = sim_grouped_makespan(task_counts, prefetch=False)
    proposed_ms = sim_proposed_makespan(task_counts, num_gpus=1)

    return {
        "fifo":     {"makespan": fifo_ms, "throughput": 500 / fifo_ms},
        "random":   {"makespan": random_ms, "throughput": 500 / random_ms},
        "grouped":  {"makespan": grouped_ms, "throughput": 500 / grouped_ms},
        "proposed": {"makespan": proposed_ms, "throughput": 500 / proposed_ms},
    }


def plot_exp5_standard():
    data = sim_exp5_standard()
    names = ["fifo", "random", "grouped", "proposed"]
    labels = ["FIFO\n(Alternating)", "Random\n(Shuffled)",
              "Grouped\n(No Prefetch)", "Proposed\n(Johnson+Prefetch)"]
    makespans = [data[n]["makespan"] for n in names]
    throughputs = [data[n]["throughput"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))

    for i, (vals, ylabel, title, fmt) in enumerate([
        (makespans, "Makespan (s)", "(a) Makespan", "{:.1f}s"),
        (throughputs, "Throughput (tasks/s)", "(b) Throughput", "{:.2f}"),
    ]):
        ax = [ax1, ax2][i]
        bars = ax.bar(labels, vals, color=COLORS[:4], edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars, HATCHES[:4]):
            bar.set_hatch(h)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        add_value_labels(ax, bars, fmt=fmt, offset=max(vals) * 0.02)

    fig.suptitle("Exp 5: Standard Workload\n"
                 "(500 Tasks — 0.6B / 8B / 32B / 120B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    save_fig(fig, "exp5_standard_workload")


# ══════════════════════════════════════════════════════��═════════════════
# Exp 5b: Scalability (batch size)
# ════════════════════════════════════════════════════════════════════════

def sim_exp5_scalability():
    """Vary batch size: 10..100000, 4 models."""
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    results = {"fifo": {}, "random": {}, "grouped": {}, "proposed": {}}

    for n in batch_sizes:
        tc = {m: max(1, int(n * f)) for m, f in STANDARD_WORKLOAD["mix"].items()}
        tc["small"] += n - sum(tc.values())  # fix rounding

        fifo_ms = sim_fifo_makespan(tc)
        rand_ms = sim_random_makespan(tc)
        grp_ms = sim_grouped_makespan(tc, prefetch=False)
        prop_ms = sim_proposed_makespan(tc, num_gpus=1)

        results["fifo"][n] = {"makespan": fifo_ms, "throughput": n / fifo_ms}
        results["random"][n] = {"makespan": rand_ms, "throughput": n / rand_ms}
        results["grouped"][n] = {"makespan": grp_ms, "throughput": n / grp_ms}
        results["proposed"][n] = {"makespan": prop_ms, "throughput": n / prop_ms}

    return results, batch_sizes


def plot_exp5_scalability():
    data, batch_sizes = sim_exp5_scalability()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for i, (strategy, label, marker) in enumerate([
        ("fifo", "FIFO (Alternating)", "o"),
        ("random", "Random (Shuffled)", "^"),
        ("grouped", "Grouped (No Prefetch)", "s"),
        ("proposed", "Proposed (Johnson+Prefetch)", "D"),
    ]):
        ms = [data[strategy][n]["makespan"] for n in batch_sizes]
        tp = [data[strategy][n]["throughput"] for n in batch_sizes]
        ax1.plot(batch_sizes, ms, f"-{marker}", color=COLORS[i], label=label,
                 linewidth=1.5, markersize=5)
        ax2.plot(batch_sizes, tp, f"-{marker}", color=COLORS[i], label=label,
                 linewidth=1.5, markersize=5)

    ax1.set_xlabel("Batch Size (tasks)")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan vs Batch Size")
    ax1.legend(fontsize=7)
    ax1.set_xscale("log")

    ax2.set_xlabel("Batch Size (tasks)")
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_title("(b) Throughput vs Batch Size")
    ax2.legend(fontsize=7)
    ax2.set_xscale("log")

    fig.suptitle("Exp 5: Scalability\n"
                 "(0.6B / 8B / 32B / 120B, Varying Batch Size, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    save_fig(fig, "exp5_scalability")


# ═��══════════════════════════════════════════════════════════════════════
# Exp 6: Model Size Robustness
# ════════════════════════════════════════════════════════════════════════

def plot_exp6():
    """50 tasks per model, single-model batches."""
    n = 50
    sizes = ["small", "medium", "large", "xlarge"]
    size_labels = ["0.6B", "8B", "32B", "120B"]
    makespans, throughputs = [], []
    for s in sizes:
        ms = MODEL_LOAD_TIMES[s] + inference_time(s, n)
        makespans.append(ms)
        throughputs.append(n / ms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    bars = ax1.bar(size_labels, makespans, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:4]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_xlabel("Model Size")
    ax1.set_title("(a) Makespan")
    add_value_labels(ax1, bars)

    bars2 = ax2.bar(size_labels, throughputs, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:4]):
        bar.set_hatch(h)
    ax2.set_ylabel("Throughput (tasks/s)")
    ax2.set_xlabel("Model Size")
    ax2.set_title("(b) Throughput")
    add_value_labels(ax2, bars2, fmt="{:.2f}", offset=0.01)

    fig.suptitle("Exp 6: Model Size Robustness\n"
                 "(50 Tasks Each — 0.6B / 8B / 32B / 120B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp6_model_size")


# ════════════════════════════════════════════════════════════════════════
# Exp 7: Stress Test (Admission Control Gate)
# ════════════════════════════════════════════════════════════════════════

def plot_exp7():
    """Various stress patterns, single model (small)."""
    model = "small"
    load = MODEL_LOAD_TIMES[model]

    patterns = {
        "burst":            {"makespan": load + inference_time(model, 1000), "fail_rate": 0.0},
        "concurrent_users": {"makespan": load + inference_time(model, 1000), "fail_rate": 0.0},
        "sustained":        {"makespan": load + 10 * inference_time(model, 100) + 9 * 5, "fail_rate": 0.0},
        "file_concurrent":  {"makespan": load + inference_time(model, 900), "fail_rate": 0.0},
    }
    name_map = {
        "burst": "Burst\n(1x1000)",
        "concurrent_users": "Concurrent\n(5x200)",
        "sustained": "Sustained\n(10x100)",
        "file_concurrent": "File Upload\n(3x300)",
    }
    names = list(name_map.keys())
    labels = [name_map[n] for n in names]
    makespans = [patterns[n]["makespan"] for n in names]
    fail_rates = [patterns[n]["fail_rate"] * 100 for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    bars = ax1.bar(labels, makespans, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:4]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan")
    add_value_labels(ax1, bars)

    bars2 = ax2.bar(labels, fail_rates, color=COLORS[:4], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:4]):
        bar.set_hatch(h)
    ax2.set_ylabel("Failure Rate (%)")
    ax2.set_title("(b) Failure Rate")
    ax2.axhline(y=5, color="red", linestyle="--", linewidth=0.8, label="5% threshold")
    ax2.set_ylim(-0.5, 10)
    ax2.legend()

    fig.suptitle("Exp 7: Stress Test Results — Admission Control Gate\n"
                 "(0.6B / 8B / 32B / 120B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp7_stress_test")


# ════════════════════════════════════════════════════════════════════════
# Exp 8: Offline Batch Scheduling — THE KEY DISSERTATION FIGURE
# Johnson's Rule + Multi-GPU Assignment
# ════════════════════════════════════════════════════════════════════════

def sim_exp8_main():
    """500 tasks, 4 models, 2 GPUs. FIFO vs Random vs Proposed."""
    task_counts = {"small": 200, "medium": 150, "large": 100, "xlarge": 50}

    fifo_ms = sim_fifo_makespan(task_counts)
    random_ms = sim_random_makespan(task_counts)
    grouped_no_pf = sim_grouped_makespan(task_counts, prefetch=False)
    proposed_1gpu = sim_proposed_makespan(task_counts, num_gpus=1)
    proposed_2gpu = sim_proposed_makespan(task_counts, num_gpus=2)

    return {
        "fifo":          {"makespan": fifo_ms},
        "random":        {"makespan": random_ms},
        "grouped":       {"makespan": grouped_no_pf},
        "proposed_1gpu": {"makespan": proposed_1gpu},
        "proposed_2gpu": {"makespan": proposed_2gpu},
    }


def plot_exp8_main():
    """Multi-panel figure: the central result of the thesis."""
    data = sim_exp8_main()
    total = 500

    names = ["fifo", "random", "grouped", "proposed_1gpu", "proposed_2gpu"]
    labels = ["FIFO\n(Alternating)", "Random\n(Shuffled)", "Grouped\n(No Prefetch)",
              "Proposed\n(1 GPU)", "Proposed\n(2 GPUs)"]
    makespans = [data[n]["makespan"] for n in names]
    throughputs = [total / m for m in makespans]
    speedups = [makespans[0] / m for m in makespans]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: Makespan
    bars = axes[0].bar(labels, makespans, color=COLORS[:5], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:5]):
        bar.set_hatch(h)
    axes[0].set_ylabel("Makespan (s)")
    axes[0].set_title("(a) Total Makespan")
    add_value_labels(axes[0], bars)

    # Panel B: Throughput
    bars2 = axes[1].bar(labels, throughputs, color=COLORS[:5], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars2, HATCHES[:5]):
        bar.set_hatch(h)
    axes[1].set_ylabel("Throughput (tasks/s)")
    axes[1].set_title("(b) Throughput")
    add_value_labels(axes[1], bars2, fmt="{:.2f}", offset=0.01)

    # Panel C: Speedup vs FIFO
    bars3 = axes[2].bar(labels, speedups, color=COLORS[:5], edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars3, HATCHES[:5]):
        bar.set_hatch(h)
    axes[2].set_ylabel("Speedup vs FIFO")
    axes[2].set_title("(c) Speedup")
    axes[2].axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    add_value_labels(axes[2], bars3, fmt="{:.1f}x", offset=0.05)

    fig.suptitle("Exp 8: Offline Batch Scheduling — Strategy Comparison\n"
                 "(500 Tasks — 0.6B / 8B / 32B / 120B, 2× RTX A5000, NVMe SSD)",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    save_fig(fig, "exp8_batch_scheduling_main")


# ════════════════════════════════════════════════════════════════════════
# Exp 8b: Gantt-style Execution Timeline
# ════════════════════════════════════════════════════════════════════════

def plot_exp8_gantt():
    """Gantt chart: FIFO vs Random vs Proposed on 2 GPUs."""
    models = ["small", "medium", "large", "xlarge"]
    model_colors = {"small": "#3498db", "medium": "#e67e22", "large": "#9b59b6", "xlarge": "#e74c3c"}
    model_labels = {"small": "0.6B", "medium": "8B", "large": "32B", "xlarge": "120B"}
    task_counts = {"small": 30, "medium": 20, "large": 10, "xlarge": 5}

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # ── Panel 1: FIFO (alternating, 1 GPU active) ──
    ax = axes[0]
    ax.set_title("(a) FIFO — Alternating Model Order (1 GPU)", fontsize=10)
    t = 0.0
    loaded = None
    max_count = max(task_counts.values())
    task_list = []
    for i in range(max_count):
        for m in models:
            if i < task_counts[m]:
                task_list.append(m)
    for m in task_list:
        if loaded != m:
            load = MODEL_LOAD_TIMES[m]
            ax.barh(0, load, left=t, height=0.6,
                    color=model_colors[m], alpha=0.3, hatch="//",
                    edgecolor="black", linewidth=0.5)
            t += load
            loaded = m
        inf = MODEL_INFERENCE_TIMES[m]
        ax.barh(0, inf, left=t, height=0.6,
                color=model_colors[m], edgecolor="none", alpha=0.8)
        t += inf
    ax.barh(1, t, left=0, height=0.6, color="#ecf0f1", edgecolor="none")
    ax.text(t / 2, 1, "idle", ha="center", va="center", fontsize=8, color="#7f8c8d")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GPU 0", "GPU 1"])
    ax.invert_yaxis()

    # ── Panel 2: Random (shuffled, 1 GPU) ──
    ax = axes[1]
    ax.set_title("(b) Random — Shuffled Task Order (1 GPU)", fontsize=10)
    rng = random.Random(42)
    task_list_r = []
    for m, n in task_counts.items():
        task_list_r.extend([m] * n)
    rng.shuffle(task_list_r)
    t = 0.0
    loaded = None
    for m in task_list_r:
        if loaded != m:
            load = MODEL_LOAD_TIMES[m]
            ax.barh(0, load, left=t, height=0.6,
                    color=model_colors[m], alpha=0.3, hatch="//",
                    edgecolor="black", linewidth=0.5)
            t += load
            loaded = m
        inf = MODEL_INFERENCE_TIMES[m]
        ax.barh(0, inf, left=t, height=0.6,
                color=model_colors[m], edgecolor="none", alpha=0.8)
        t += inf
    ax.barh(1, t, left=0, height=0.6, color="#ecf0f1", edgecolor="none")
    ax.text(t / 2, 1, "idle", ha="center", va="center", fontsize=8, color="#7f8c8d")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GPU 0", "GPU 1"])
    ax.invert_yaxis()

    # ── Panel 3: Proposed (Johnson's Rule + 2 GPUs + prefetch) ──
    ax = axes[2]
    ax.set_title("(c) Proposed — Johnson's Rule + Multi-GPU + Prefetch", fontsize=10)

    # Greedy LPT assignment
    group_times = {}
    for m, n in task_counts.items():
        group_times[m] = MODEL_LOAD_TIMES[m] + inference_time(m, n)
    sorted_groups = sorted(group_times.items(), key=lambda x: x[1], reverse=True)

    gpu_loads = [0.0, 0.0]
    gpu_queues = [[], []]
    for m, gt in sorted_groups:
        min_gpu = gpu_loads.index(min(gpu_loads))
        gpu_queues[min_gpu].append(m)
        gpu_loads[min_gpu] += gt

    for gpu_idx, queue in enumerate(gpu_queues):
        if not queue:
            continue
        ordered = johnsons_rule_order({m: task_counts[m] for m in queue})
        t = 0.0
        for i, m in enumerate(ordered):
            if i == 0:
                load = MODEL_LOAD_TIMES[m]
            else:
                load = PCIE_TRANSFER_TIMES[m]
            ax.barh(gpu_idx, load, left=t, height=0.6,
                    color=model_colors[m], alpha=0.3, hatch="//",
                    edgecolor="black", linewidth=0.5)
            t += load
            inf = inference_time(m, task_counts[m])
            ax.barh(gpu_idx, inf, left=t, height=0.6,
                    color=model_colors[m], edgecolor="none", alpha=0.8)
            t += inf

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GPU 0", "GPU 1"])
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")

    # Legend
    legend_patches = [mpatches.Patch(color=model_colors[m], label=model_labels[m])
                      for m in models]
    legend_patches.append(mpatches.Patch(facecolor="gray", alpha=0.3, hatch="//",
                                         edgecolor="black", label="Model Loading"))
    fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Exp 8: Execution Timeline\n"
                 "(65 Tasks — 0.6B / 8B / 32B / 120B, 2× RTX A5000)",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "exp8_gantt")


# ════════════════════════════════════════════════════════════════════════
# Exp 8c: Multi-GPU Scalability
# ════════════════════════════════════════════════════════════════════════

def plot_exp8_gpu_scalability():
    """Line chart: makespan vs number of GPUs."""
    # Use 4 models with balanced workload to show meaningful GPU scaling
    task_counts = {"small": 1000, "medium": 800, "large": 600, "xlarge": 100}
    total_tasks = sum(task_counts.values())
    gpu_counts = [1, 2, 4, 8]

    fifo_ms = sim_fifo_makespan(task_counts)
    grouped_ms = sim_grouped_makespan(task_counts, prefetch=False)

    proposed_ms = []
    for ng in gpu_counts:
        proposed_ms.append(sim_proposed_makespan(task_counts, num_gpus=ng))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: Makespan
    ax1.axhline(y=grouped_ms, color=COLORS[3], linestyle=":", linewidth=1.5,
                label=f"Grouped no-pf ({grouped_ms:.0f}s)")
    ax1.plot(gpu_counts, proposed_ms, "-D", color=COLORS[2], linewidth=2,
             markersize=7, label="Proposed")
    for x, y in zip(gpu_counts, proposed_ms):
        ax1.annotate(f"{y:.0f}s", (x, y), textcoords="offset points",
                     xytext=(0, 10), fontsize=8, ha="center", fontweight="bold")
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan vs GPU Count")
    ax1.legend(fontsize=8)
    ax1.set_xticks(gpu_counts)

    # Panel B: Speedup vs 1-GPU proposed
    base = proposed_ms[0]
    speedups = [base / m for m in proposed_ms]
    ideal = [float(g) for g in gpu_counts]
    ax2.plot(gpu_counts, speedups, "-D", color=COLORS[2], linewidth=2,
             markersize=7, label="Actual Speedup")
    ax2.plot(gpu_counts, ideal, "--", color="gray", linewidth=1, label="Ideal Linear")
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Speedup vs 1 GPU")
    ax2.set_title("(b) Multi-GPU Scaling Efficiency")
    ax2.legend(fontsize=8)
    ax2.set_xticks(gpu_counts)
    for x, y in zip(gpu_counts, speedups):
        ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                     xytext=(0, 10), fontsize=8, ha="center", fontweight="bold")

    fig.suptitle("Exp 8: Multi-GPU Scalability\n"
                 f"({total_tasks} Tasks — 0.6B / 8B / 32B / 120B, RTX A5000, NVMe SSD)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp8_gpu_scalability")


# ════════════════════════════════════════════════════════════════════════
# Exp 8d: Johnson's Rule vs Other Orderings
# ════════════════════════════════════════════════════════════════════════

def plot_exp8_johnsons():
    """Compare Johnson's Rule ordering vs alternatives using 8-model workload.

    Uses a skewed workload with 8 models to demonstrate the S1/S2 split:
    popular small models (S1: compute > I/O) and niche large models (S2: I/O > compute).
    """
    # 8-model workload: skewed distribution (popular small + niche large)
    jr_models = {
        'A_0.6B':  {'load': 15,  'pcie': 0.04, 'inf_batch': 3,  'tasks': 2000},
        'B_1.5B':  {'load': 25,  'pcie': 0.10, 'inf_batch': 5,  'tasks': 1200},
        'C_3B':    {'load': 40,  'pcie': 0.20, 'inf_batch': 8,  'tasks': 600},
        'D_8B':    {'load': 55,  'pcie': 0.48, 'inf_batch': 10, 'tasks': 500},
        'E_13B':   {'load': 90,  'pcie': 0.80, 'inf_batch': 15, 'tasks': 100},
        'F_32B':   {'load': 180, 'pcie': 1.88, 'inf_batch': 30, 'tasks': 30},
        'G_65B':   {'load': 350, 'pcie': 3.50, 'inf_batch': 60, 'tasks': 10},
        'H_120B':  {'load': 580, 'pcie': 7.00, 'inf_batch': 90, 'tasks': 5},
    }
    names = list(jr_models.keys())
    task_counts = {m: jr_models[m]['tasks'] for m in names}

    def _jr_flowshop(order):
        """Two-machine flow-shop for the 8-model workload."""
        n = len(order)
        ssd_end = [0.0] * n
        ssd_end[0] = jr_models[order[0]]['load']
        for i in range(1, n):
            ssd_end[i] = ssd_end[i-1] + (jr_models[order[i]]['load'] - jr_models[order[i]]['pcie'])
        gpu_end = [0.0] * n
        inf0 = math.ceil(task_counts[order[0]] / MAX_CONCURRENT_PER_GPU) * jr_models[order[0]]['inf_batch']
        gpu_end[0] = ssd_end[0] + inf0
        for i in range(1, n):
            m = order[i]
            inf_t = math.ceil(task_counts[m] / MAX_CONCURRENT_PER_GPU) * jr_models[m]['inf_batch']
            gpu_start = max(ssd_end[i], gpu_end[i-1]) + jr_models[m]['pcie']
            gpu_end[i] = gpu_start + inf_t
        return gpu_end[-1]

    # Johnson's Rule classification
    s1, s2 = [], []
    for m in names:
        io_t = jr_models[m]['load'] - jr_models[m]['pcie']
        comp_t = math.ceil(task_counts[m] / MAX_CONCURRENT_PER_GPU) * jr_models[m]['inf_batch']
        if io_t <= comp_t:
            s1.append((m, io_t, comp_t))
        else:
            s2.append((m, io_t, comp_t))
    s1.sort(key=lambda x: x[1])
    s2.sort(key=lambda x: x[2], reverse=True)
    jr_order = [m for m, _, _ in s1] + [m for m, _, _ in s2]

    jr_ms = _jr_flowshop(jr_order)
    asc_ms = _jr_flowshop(names)
    desc_ms = _jr_flowshop(list(reversed(names)))
    by_count = sorted(names, key=lambda m: task_counts[m], reverse=True)
    count_ms = _jr_flowshop(by_count)
    rng = random.Random(42)
    rand_order = names[:]
    rng.shuffle(rand_order)
    rand_ms = _jr_flowshop(rand_order)

    labels = ["Johnson's\nRule", "Ascending\nSize", "Descending\nSize",
              "Most Tasks\nFirst", "Random\nOrder"]
    makespans = [jr_ms, asc_ms, desc_ms, count_ms, rand_ms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # Panel A: bar chart
    colors_5 = COLORS[:5]
    bars = ax1.bar(labels, makespans, color=colors_5, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, HATCHES[:5]):
        bar.set_hatch(h)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan by Ordering Strategy", fontweight="bold")
    add_value_labels(ax1, bars)

    best_idx = makespans.index(min(makespans))
    bars[best_idx].set_edgecolor(COLORS[2])
    bars[best_idx].set_linewidth(2.5)

    # Panel B: S1/S2 classification table
    ax2.axis("off")
    s1_labels = [f"{m.split('_')[1]}" for m, _, _ in s1]
    s2_labels = [f"{m.split('_')[1]}" for m, _, _ in s2]

    table_data = []
    for m, io_t, comp_t in s1:
        short = m.split('_')[1]
        table_data.append([short, f"{task_counts[m]}", f"{io_t:.0f}", f"{comp_t:.0f}", "S1"])
    for m, io_t, comp_t in s2:
        short = m.split('_')[1]
        table_data.append([short, f"{task_counts[m]}", f"{io_t:.0f}", f"{comp_t:.0f}", "S2"])

    table = ax2.table(cellText=table_data,
                      colLabels=["Model", "Tasks", "I/O (s)", "Compute (s)", "Set"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)

    # Color S1 rows green, S2 rows red
    for i in range(len(s1)):
        for j in range(5):
            table[i + 1, j].set_facecolor("#d4edda")
    for i in range(len(s2)):
        for j in range(5):
            table[len(s1) + i + 1, j].set_facecolor("#f8d7da")

    ax2.set_title(f"(b) Johnson's Rule Classification\n"
                  f"S1 (asc I/O): {', '.join(s1_labels)}\n"
                  f"S2 (desc compute): {', '.join(s2_labels)}",
                  fontweight="bold", fontsize=9)

    fig.suptitle("Johnson's Rule vs Alternative Orderings\n"
                 f"(8 Models, {sum(task_counts.values())} Tasks, Skewed Distribution, 1 GPU)",
                 fontweight="bold", y=1.04)
    fig.tight_layout()
    save_fig(fig, "exp8_johnsons_rule")


def _sim_ordered_makespan(task_counts, order, prefetch=True):
    """Simulate makespan with a specific model order using flow-shop model."""
    if prefetch:
        return _flowshop_makespan(order, task_counts)
    # No prefetch: sequential cold loads
    t = MODEL_LOAD_TIMES[order[0]]
    for i, m in enumerate(order):
        t += inference_time(m, task_counts[m])
        if i < len(order) - 1:
            t += MODEL_LOAD_TIMES[order[i + 1]]
    return t


# ════════════════════════════════════════════════════════════════════════
# Exp 8e: Batch Size Sensitivity
# ════════════════════════════════════════════════════════════════════════

def plot_exp8_batch_sensitivity():
    """Line chart: how batch size affects each strategy."""
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 100000]
    mix = STANDARD_WORKLOAD["mix"]

    fifo_ms, random_ms, grouped_ms, proposed_ms = [], [], [], []
    for n in batch_sizes:
        tc = {m: int(n * f) for m, f in mix.items()}
        tc["small"] += n - sum(tc.values())  # fix rounding
        fifo_ms.append(sim_fifo_makespan(tc))
        random_ms.append(sim_random_makespan(tc))
        grouped_ms.append(sim_grouped_makespan(tc, prefetch=False))
        proposed_ms.append(sim_proposed_makespan(tc, num_gpus=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    for ms_list, label, marker, color in [
        (fifo_ms, "FIFO", "o", COLORS[0]),
        (random_ms, "Random", "^", COLORS[1]),
        (grouped_ms, "Grouped (no pf)", "s", COLORS[3]),
        (proposed_ms, "Proposed (2 GPU)", "D", COLORS[2]),
    ]:
        ax1.plot(batch_sizes, ms_list, f"-{marker}", color=color, label=label,
                 linewidth=1.5, markersize=5)
        speedups = [fifo_ms[i] / ms_list[i] for i in range(len(batch_sizes))]
        ax2.plot(batch_sizes, speedups, f"-{marker}", color=color, label=label,
                 linewidth=1.5, markersize=5)

    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan vs Batch Size")
    ax1.legend(fontsize=8)
    ax1.set_xscale("log")

    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Speedup vs FIFO")
    ax2.set_title("(b) Speedup vs Batch Size")
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Exp 8: Batch Size Sensitivity\n"
                 "(0.6B / 8B / 32B / 120B, Standard Mix, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "exp8_batch_sensitivity")


# ════════════════════════════════════════════════════════════════════════
# INSIGHT FIGURES
# ════════════════════════════════════════════════════════════════════════


def plot_waterfall_breakdown():
    """Stacked bar: time breakdown by category for each strategy.

    FIFO/Random: each task gets its own batch cycle (1 task per batch),
    so inference = N_tasks * batch_time.  Grouped/Proposed batch tasks
    together (up to 50 per cycle), so inference is much lower.
    """
    task_counts = {"small": 200, "medium": 150, "large": 100, "xlarge": 50}
    models = ["small", "medium", "large", "xlarge"]

    # Batched inference (grouped): tasks batched together
    batched_inf = sum(inference_time(m, task_counts[m]) for m in models)
    # Unbatched inference (FIFO): 1 task per batch cycle
    unbatched_inf = sum(task_counts[m] * MODEL_INFERENCE_TIMES[m] for m in models)

    # ── FIFO breakdown ──
    fifo_ms = sim_fifo_makespan(task_counts)
    fifo_loading = fifo_ms - unbatched_inf

    # ── Random breakdown ──
    random_ms = sim_random_makespan(task_counts)
    random_loading = random_ms - unbatched_inf

    # ── Grouped (no prefetch) breakdown ──
    grouped_ms = sim_grouped_makespan(task_counts, prefetch=False)
    grouped_loading = grouped_ms - batched_inf

    # ── Proposed breakdown ──
    proposed_ms = sim_proposed_makespan(task_counts, num_gpus=2)
    proposed_loading = proposed_ms - batched_inf

    strategies = ["FIFO\n(Alternating)", "Random\n(Shuffled)",
                  "Grouped\n(No Prefetch)", "Proposed\n(2 GPUs)"]
    categories = ["Model Loading", "Inference"]
    cat_colors = ["#e74c3c", "#27ae60"]
    cat_hatches = ["//", ""]

    data_arr = np.array([
        [fifo_loading, unbatched_inf],
        [random_loading, unbatched_inf],
        [grouped_loading, batched_inf],
        [max(0, proposed_loading), batched_inf],
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [2, 1]})

    y = np.arange(len(strategies))
    left = np.zeros(len(strategies))
    for i, (cat, color, hatch) in enumerate(zip(categories, cat_colors, cat_hatches)):
        bars = ax1.barh(y, data_arr[:, i], left=left, height=0.6,
                        color=color, edgecolor="black", linewidth=0.5,
                        hatch=hatch, label=cat, alpha=0.85)
        for j, (val, l) in enumerate(zip(data_arr[:, i], left)):
            if val > data_arr[j].sum() * 0.04:
                ax1.text(l + val / 2, j, f"{val:.1f}s",
                         ha="center", va="center", fontsize=8,
                         fontweight="bold", color="white" if color != "#bdc3c7" else "#2c3e50")
        left += data_arr[:, i]

    for j, total in enumerate(data_arr.sum(axis=1)):
        ax1.text(total + 2, j, f"Total: {total:.0f}s",
                 ha="left", va="center", fontsize=9, fontweight="bold")

    ax1.set_yticks(y)
    ax1.set_yticklabels(strategies, fontsize=11)
    ax1.set_xlabel("Time (s)")
    ax1.set_title("(a) Time Breakdown by Category", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.invert_yaxis()
    ax1.grid(axis="y", alpha=0)

    # Panel B: Percentage
    data_pct = data_arr / data_arr.sum(axis=1, keepdims=True) * 100
    left_pct = np.zeros(len(strategies))
    for i, (cat, color, hatch) in enumerate(zip(categories, cat_colors, cat_hatches)):
        bars = ax2.barh(y, data_pct[:, i], left=left_pct, height=0.6,
                        color=color, edgecolor="black", linewidth=0.5,
                        hatch=hatch, alpha=0.85)
        for j, (val, l) in enumerate(zip(data_pct[:, i], left_pct)):
            if val > 5:
                ax2.text(l + val / 2, j, f"{val:.0f}%",
                         ha="center", va="center", fontsize=8,
                         fontweight="bold", color="white" if color != "#bdc3c7" else "#2c3e50")
        left_pct += data_pct[:, i]

    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(strategies))
    ax2.set_xlabel("Proportion (%)")
    ax2.set_title("(b) Proportional Breakdown", fontweight="bold")
    ax2.set_xlim(0, 105)
    ax2.invert_yaxis()
    ax2.grid(axis="y", alpha=0)

    fig.suptitle("Where Does Time Go? Makespan Decomposition\n"
                 "(500 Tasks — 0.6B / 8B / 32B / 120B, RTX A5000)",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_waterfall_breakdown")


def plot_switching_cost_curve():
    """Cumulative switching cost curve."""
    switch_counts = list(range(0, 51))
    base_inf = inference_time("medium", 100)
    base_load = MODEL_LOAD_TIMES["medium"]
    base_total = base_load + base_inf

    total_times = []
    overhead_only = []
    avg_switch = sum(MODEL_LOAD_TIMES[m] for m in ["small", "medium", "large", "xlarge"]) / 4

    for ns in switch_counts:
        if ns == 0:
            total_times.append(base_total)
            overhead_only.append(0)
        else:
            switch_overhead = ns * (avg_switch + MODEL_UNLOAD_TIME)
            total_times.append(base_total + switch_overhead)
            overhead_only.append(switch_overhead)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.fill_between(switch_counts, [base_total] * len(switch_counts),
                     color="#27ae60", alpha=0.3, label="Useful Work (inference)")
    ax1.fill_between(switch_counts, [base_total] * len(switch_counts),
                     total_times, color="#e74c3c", alpha=0.4, label="Switching Overhead")
    ax1.plot(switch_counts, total_times, "-", color="#e74c3c", linewidth=2)

    strategy_points = [
        (2,  "Proposed\n(2 switches)", COLORS[2]),
        (10, "Random\n(~10 switches)", COLORS[1]),
        (20, "FIFO\n(~20 switches)", COLORS[0]),
    ]
    for ns, label, color in strategy_points:
        y = total_times[ns]
        ax1.plot(ns, y, "o", color=color, markersize=10, zorder=5,
                 markeredgecolor="black", markeredgewidth=1.5)
        ax1.annotate(label, (ns, y), textcoords="offset points",
                     xytext=(12, 10), fontsize=8, fontweight="bold", color=color,
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    ax1.set_xlabel("Number of Model Switches")
    ax1.set_ylabel("Total Execution Time (s)")
    ax1.set_title("(a) Cumulative Cost of Model Switching", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")

    overhead_pct = [o / base_total * 100 for o in overhead_only]
    ax2.plot(switch_counts, overhead_pct, "-", color="#e74c3c", linewidth=2)
    ax2.fill_between(switch_counts, overhead_pct, alpha=0.2, color="#e74c3c")
    ax2.axhline(y=100, color="gray", linestyle="--", linewidth=0.8)
    ax2.text(52, 102, "Overhead = Useful Work", fontsize=7, color="gray")

    for ns, label, color in strategy_points:
        pct = overhead_pct[ns]
        ax2.plot(ns, pct, "o", color=color, markersize=10, zorder=5,
                 markeredgecolor="black", markeredgewidth=1.5)
        ax2.annotate(f"{pct:.0f}%", (ns, pct), textcoords="offset points",
                     xytext=(10, 5), fontsize=9, fontweight="bold", color=color)

    ax2.set_xlabel("Number of Model Switches")
    ax2.set_ylabel("Switching Overhead (% of useful work)")
    ax2.set_title("(b) Overhead Ratio", fontweight="bold")

    fig.suptitle("The Cost of Model Switching: Why Minimizing Switches Matters\n"
                 "(100 Tasks, Average Model Load Time, RTX A5000 + NVMe SSD)",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_switching_cost")


# ════════════════════════════════════════════════════════════════════════
# Insight: GPU Count × Model Count Heatmap
# ════════════════════════════════════════════════════════════════════════

def _make_workload(num_models: int, tasks_per_model: int = 100) -> Dict[str, int]:
    """Generate a synthetic workload with num_models distinct models.

    Models cycle through small/medium/large/xlarge sizes to create
    realistic heterogeneous workloads.
    """
    sizes = ["small", "medium", "large", "xlarge"]
    tc: Dict[str, int] = {}
    for i in range(num_models):
        key = f"{sizes[i % len(sizes)]}_{i}" if num_models > len(sizes) else sizes[i % len(sizes)]
        tc[key] = tasks_per_model
    return tc


def _sim_fifo_multi(task_counts: Dict[str, int], model_params: Dict[str, dict]) -> float:
    """FIFO for arbitrary model keys — realistic arrival order.

    Simulates tasks arriving in small bursts (burst_size=5) per model,
    round-robin across models.  This is more realistic than perfect
    alternation (which is pathologically bad) while still being unoptimized.
    """
    models = list(task_counts.keys())
    remaining = dict(task_counts)
    task_list = []
    burst_size = 5
    while any(remaining[m] > 0 for m in models):
        for m in models:
            take = min(burst_size, remaining[m])
            task_list.extend([m] * take)
            remaining[m] -= take

    t = 0.0
    loaded = None
    i = 0
    while i < len(task_list):
        m = task_list[i]
        if loaded != m:
            t += model_params[m]["load"]
            loaded = m
        run_len = 1
        while i + run_len < len(task_list) and task_list[i + run_len] == m:
            run_len += 1
        batches = math.ceil(run_len / MAX_CONCURRENT_PER_GPU)
        t += batches * model_params[m]["inf_per_batch"]
        i += run_len
    return t


def _sim_proposed_multi(task_counts: Dict[str, int], model_params: Dict[str, dict],
                        num_gpus: int) -> float:
    """Proposed strategy for arbitrary model keys.

    Supports two modes:
    1. Model-level LPT: each model assigned to exactly one GPU (no duplication)
    2. Task-level split: a model's tasks can be split across GPUs (each pays cold start)
    Returns the better of the two.
    """
    active = {m: n for m, n in task_counts.items() if n > 0}
    if not active:
        return 0.0

    def _gpu_queue_time(q):
        """Compute makespan for a single GPU queue with Johnson's Rule + prefetch."""
        if not q:
            return 0.0
        s1, s2 = [], []
        for m, n in q.items():
            i_time = model_params[m]["load"]
            batches = math.ceil(n / MAX_CONCURRENT_PER_GPU)
            p_time = batches * model_params[m]["inf_per_batch"]
            if i_time <= p_time:
                s1.append((m, i_time, p_time))
            else:
                s2.append((m, i_time, p_time))
        s1.sort(key=lambda x: x[1])
        s2.sort(key=lambda x: x[2], reverse=True)
        order = [m for m, _, _ in s1] + [m for m, _, _ in s2]

        t = model_params[order[0]]["load"]  # first cold start
        for i, m in enumerate(order):
            batches = math.ceil(q[m] / MAX_CONCURRENT_PER_GPU)
            t += batches * model_params[m]["inf_per_batch"]
            if i < len(order) - 1:
                t += model_params[order[i + 1]]["pcie"]  # prefetch
        return t

    # 1-GPU baseline
    sg = _gpu_queue_time(active)
    if num_gpus == 1:
        return sg

    # --- Strategy A: Model-level LPT (no task splitting) ---
    group_info = {}
    for m, n in active.items():
        batches = math.ceil(n / MAX_CONCURRENT_PER_GPU)
        group_info[m] = model_params[m]["load"] + batches * model_params[m]["inf_per_batch"]

    sorted_groups = sorted(group_info.items(), key=lambda x: x[1], reverse=True)
    ng = min(num_gpus, len(active))
    gpu_work_a = [0.0] * ng
    gpu_queues_a: List[Dict[str, int]] = [{} for _ in range(ng)]
    for model, total in sorted_groups:
        min_gpu = gpu_work_a.index(min(gpu_work_a))
        gpu_queues_a[min_gpu][model] = active[model]
        gpu_work_a[min_gpu] += total

    gpu_times_a = [_gpu_queue_time(q) for q in gpu_queues_a if q]
    mg_a = max(gpu_times_a) if gpu_times_a else sg

    # --- Strategy B: Task-level split (models can span GPUs) ---
    # Greedy: for each model, decide how many GPUs to use based on
    # whether splitting reduces the per-GPU time enough to offset extra cold starts.
    # Simple approach: evenly distribute all tasks across all GPUs,
    # each GPU loads all models.
    ng_b = num_gpus
    gpu_queues_b: List[Dict[str, int]] = [{} for _ in range(ng_b)]
    for m, n in active.items():
        per_gpu = n // ng_b
        remainder = n % ng_b
        for g in range(ng_b):
            count = per_gpu + (1 if g < remainder else 0)
            if count > 0:
                gpu_queues_b[g][m] = count

    gpu_times_b = [_gpu_queue_time(q) for q in gpu_queues_b if q]
    mg_b = max(gpu_times_b) if gpu_times_b else sg

    # Pick the best option
    return min(sg, mg_a, mg_b)


def _build_model_params(num_models: int) -> tuple:
    """Build task_counts and model_params for N models."""
    base = [
        {"load": 15.0, "pcie": 0.04, "inf_per_batch": 3.0},    # 0.6B-like
        {"load": 55.0, "pcie": 0.48, "inf_per_batch": 10.0},   # 8B-like
        {"load": 180.0, "pcie": 1.88, "inf_per_batch": 30.0},  # 32B-like
        {"load": 580.0, "pcie": 7.00, "inf_per_batch": 90.0},  # 120B-like
    ]
    tc = {}
    params = {}
    tasks_per = 100
    for i in range(num_models):
        key = f"model_{i}"
        tc[key] = tasks_per
        params[key] = base[i % len(base)].copy()
    return tc, params


def plot_insight_gpu_model_heatmap():
    """2D heatmap: GPU count × model count → speedup vs FIFO."""
    model_counts = [1, 2, 5, 10, 25, 50, 100]
    gpu_counts = [1, 2, 4, 8]

    speedup_matrix = np.zeros((len(gpu_counts), len(model_counts)))

    for j, nm in enumerate(model_counts):
        tc, params = _build_model_params(nm)
        fifo_ms = _sim_fifo_multi(tc, params)
        for i, ng in enumerate(gpu_counts):
            prop_ms = _sim_proposed_multi(tc, params, ng)
            speedup_matrix[i, j] = fifo_ms / prop_ms

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(speedup_matrix, cmap="YlOrRd", aspect="auto", origin="lower")

    ax.set_xticks(range(len(model_counts)))
    ax.set_xticklabels(model_counts)
    ax.set_yticks(range(len(gpu_counts)))
    ax.set_yticklabels(gpu_counts)
    ax.set_xlabel("Number of Distinct Models")
    ax.set_ylabel("Number of GPUs")

    # Annotate cells
    for i in range(len(gpu_counts)):
        for j in range(len(model_counts)):
            v = speedup_matrix[i, j]
            color = "white" if v > speedup_matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{v:.1f}x", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, label="Speedup vs FIFO")
    fig.suptitle("Scheduling Benefit: GPU Count × Model Diversity\n"
                 "(100 Tasks per Model, RTX A5000, NVMe SSD)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_gpu_model_heatmap")


# ════════════════════════════════════════════════════════════════════════
# Insight: Storage Tier Comparison
# ════════════════════════════════════════════════════════════════════════

def plot_insight_storage_tier():
    """Bar chart: NVMe vs SATA SSD vs HDD load times and strategy impact."""
    storage_tiers = {
        "NVMe SSD\n(3 GB/s)": 3.0,
        "SATA SSD\n(0.5 GB/s)": 0.5,
        "HDD\n(0.15 GB/s)": 0.15,
    }
    task_counts = {"small": 200, "medium": 150, "large": 100, "xlarge": 50}
    models_list = ["small", "medium", "large", "xlarge"]
    ckpt_sizes = {m: MODELS[m]["checkpoint_GB"] for m in models_list}

    strategies = ["FIFO", "Random", "Grouped\n(No Prefetch)", "Proposed\n(+Prefetch)"]
    tier_names = list(storage_tiers.keys())

    results = {s: [] for s in strategies}

    for tier_name, bw in storage_tiers.items():
        # Recompute load times for this storage tier
        tier_load = {}
        tier_pcie = {}
        for m in models_list:
            tier_load[m] = ckpt_sizes[m] / bw + 5.0  # +5s overhead
            tier_pcie[m] = ckpt_sizes[m] / PCIE_BW_GBps

        # FIFO
        max_count = max(task_counts.values())
        task_list = []
        for i in range(max_count):
            for m in models_list:
                if i < task_counts[m]:
                    task_list.append(m)
        t = 0.0
        loaded = None
        idx = 0
        while idx < len(task_list):
            m = task_list[idx]
            if loaded != m:
                t += tier_load[m]
                loaded = m
            run_len = 1
            while idx + run_len < len(task_list) and task_list[idx + run_len] == m:
                run_len += 1
            t += inference_time(m, run_len)
            idx += run_len
        results["FIFO"].append(t)

        # Random
        rng = random.Random(42)
        task_list_r = []
        for m, n in task_counts.items():
            task_list_r.extend([m] * n)
        rng.shuffle(task_list_r)
        t = 0.0
        loaded = None
        idx = 0
        while idx < len(task_list_r):
            m = task_list_r[idx]
            if loaded != m:
                t += tier_load[m]
                loaded = m
            run_len = 1
            while idx + run_len < len(task_list_r) and task_list_r[idx + run_len] == m:
                run_len += 1
            t += inference_time(m, run_len)
            idx += run_len
        results["Random"].append(t)

        # Grouped no prefetch
        t = tier_load[models_list[0]]
        for i, m in enumerate(models_list):
            t += inference_time(m, task_counts[m])
            if i < len(models_list) - 1:
                t += tier_load[models_list[i + 1]]
        results["Grouped\n(No Prefetch)"].append(t)

        # Proposed (prefetch)
        order = johnsons_rule_order(task_counts)
        t = tier_load[order[0]]
        for i, m in enumerate(order):
            t += inference_time(m, task_counts[m])
            if i < len(order) - 1:
                t += tier_pcie[order[i + 1]]
        results["Proposed\n(+Prefetch)"].append(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(tier_names))
    width = 0.2
    for i, (strat, vals) in enumerate(results.items()):
        bars = ax1.bar(x + i * width - 1.5 * width, vals, width,
                       label=strat, color=COLORS[i], edgecolor="black",
                       linewidth=0.5, hatch=HATCHES[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_names)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan by Storage Tier")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_yscale("log")

    # Panel B: Speedup of Proposed vs FIFO per tier
    speedups = [results["FIFO"][i] / results["Proposed\n(+Prefetch)"][i]
                for i in range(len(tier_names))]
    bars2 = ax2.bar(tier_names, speedups, color=COLORS[2], edgecolor="black",
                    linewidth=0.5)
    for bar, v in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 f"{v:.0f}x", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Speedup (Proposed vs FIFO)")
    ax2.set_title("(b) Scheduling Benefit by Storage Tier")

    fig.suptitle("Impact of Storage Tier on Scheduling Benefit\n"
                 "(500 Tasks — 0.6B / 8B / 32B / 120B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_storage_tier")


# ════════════════════════════════════════════════════════════════════════
# Insight: PCIe Bandwidth Sensitivity
# ════════════════════════════════════════════════════════════════════════

def plot_insight_pcie_bandwidth():
    """Line chart: prefetch residual time vs PCIe generation."""
    pcie_gens = {
        "Gen3 x16\n(16 GB/s)": 16.0,
        "Gen4 x16\n(32 GB/s)": 32.0,
        "Gen5 x16\n(64 GB/s)": 64.0,
    }
    task_counts = {"small": 200, "medium": 150, "large": 100, "xlarge": 50}
    models_list = ["small", "medium", "large", "xlarge"]
    ckpt_sizes = {m: MODELS[m]["checkpoint_GB"] for m in models_list}

    gen_names = list(pcie_gens.keys())
    grouped_nopf = sim_grouped_makespan(task_counts, prefetch=False)

    proposed_ms = []
    pcie_residuals = []  # total PCIe transfer time
    for gen_name, bw in pcie_gens.items():
        tier_pcie = {m: ckpt_sizes[m] / bw for m in models_list}
        order = johnsons_rule_order(task_counts)
        t = MODEL_LOAD_TIMES[order[0]]
        total_pcie = 0.0
        for i, m in enumerate(order):
            t += inference_time(m, task_counts[m])
            if i < len(order) - 1:
                pcie_t = tier_pcie[order[i + 1]]
                t += pcie_t
                total_pcie += pcie_t
        proposed_ms.append(t)
        pcie_residuals.append(total_pcie)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: Makespan comparison
    ax1.axhline(y=grouped_nopf, color=COLORS[0], linestyle="--", linewidth=1.5,
                label=f"Grouped no-pf ({grouped_nopf:.0f}s)")
    bars = ax1.bar(gen_names, proposed_ms, color=[COLORS[1], COLORS[2], COLORS[3]],
                   edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, proposed_ms):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1,
                 f"{v:.1f}s", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Proposed Makespan by PCIe Gen")
    ax1.legend(fontsize=8)

    # Panel B: PCIe residual time
    bars2 = ax2.bar(gen_names, pcie_residuals,
                    color=[COLORS[1], COLORS[2], COLORS[3]],
                    edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars2, pcie_residuals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v:.2f}s", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Total PCIe Transfer Time (s)")
    ax2.set_title("(b) Prefetch Residual (PCIe Only)")

    fig.suptitle("PCIe Generation Impact on Prefetch Efficiency\n"
                 "(500 Tasks — 0.6B / 8B / 32B / 120B, 1 GPU)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_pcie_bandwidth")


# ════════════════════════════════════════════════════════════════════════
# Insight: Model Count Sweep
# ════════════════════════════════════════════════════════════════════════

def plot_insight_model_count_sweep():
    """Line chart: makespan vs number of distinct models for each strategy."""
    model_counts = [1, 2, 5, 10, 25, 50, 100]
    tasks_per_model = 100

    fifo_ms, grouped_ms, proposed_1g, proposed_4g = [], [], [], []

    for nm in model_counts:
        tc, params = _build_model_params(nm)
        fifo_ms.append(_sim_fifo_multi(tc, params))
        # Grouped no prefetch
        order = sorted(tc.keys(), key=lambda m: params[m]["load"])
        t = params[order[0]]["load"]
        for i, m in enumerate(order):
            batches = math.ceil(tc[m] / MAX_CONCURRENT_PER_GPU)
            t += batches * params[m]["inf_per_batch"]
            if i < len(order) - 1:
                t += params[order[i + 1]]["load"]
        grouped_ms.append(t)
        proposed_1g.append(_sim_proposed_multi(tc, params, 1))
        proposed_4g.append(_sim_proposed_multi(tc, params, 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ms_list, label, marker, color in [
        (fifo_ms, "FIFO", "o", COLORS[0]),
        (grouped_ms, "Grouped (no pf)", "s", COLORS[3]),
        (proposed_1g, "Proposed (1 GPU)", "D", COLORS[2]),
        (proposed_4g, "Proposed (4 GPU)", "^", COLORS[4]),
    ]:
        ax1.plot(model_counts, ms_list, f"-{marker}", color=color, label=label,
                 linewidth=1.5, markersize=5)

    ax1.set_xlabel("Number of Distinct Models")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan vs Model Diversity")
    ax1.legend(fontsize=7)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Panel B: Speedup vs FIFO
    for ms_list, label, marker, color in [
        (grouped_ms, "Grouped (no pf)", "s", COLORS[3]),
        (proposed_1g, "Proposed (1 GPU)", "D", COLORS[2]),
        (proposed_4g, "Proposed (4 GPU)", "^", COLORS[4]),
    ]:
        speedups = [fifo_ms[i] / ms_list[i] for i in range(len(model_counts))]
        ax2.plot(model_counts, speedups, f"-{marker}", color=color, label=label,
                 linewidth=1.5, markersize=5)

    ax2.set_xlabel("Number of Distinct Models")
    ax2.set_ylabel("Speedup vs FIFO")
    ax2.set_title("(b) Scheduling Benefit vs Model Diversity")
    ax2.legend(fontsize=7)
    ax2.set_xscale("log")
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

    fig.suptitle("Model Diversity Impact on Scheduling\n"
                 "(100 Tasks per Model, Mixed Sizes, RTX A5000, NVMe SSD)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_model_count_sweep")


# ════════════════════════════════════════════════════════════════════════
# Insight: Model Size Distribution
# ════════════════════════════════════════════════════════════════════════

def plot_insight_model_size_dist():
    """Grouped bar: how workload 'shape' (all small, all large, mixed) affects strategies."""
    distributions = {
        "All Small\n(0.6B)": {"small": 500},
        "All Medium\n(8B)": {"medium": 500},
        "All Large\n(32B)": {"large": 500},
        "Mixed\n(Standard)": {"small": 200, "medium": 150, "large": 100, "xlarge": 50},
        "Heavy\n(120B-heavy)": {"small": 50, "medium": 50, "large": 100, "xlarge": 300},
    }

    strategies = ["FIFO", "Grouped (no pf)", "Proposed"]
    dist_names = list(distributions.keys())

    results = {s: [] for s in strategies}
    for dist_name, tc in distributions.items():
        results["FIFO"].append(sim_fifo_makespan(tc))
        results["Grouped (no pf)"].append(sim_grouped_makespan(tc, prefetch=False))
        results["Proposed"].append(sim_proposed_makespan(tc, num_gpus=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    x = np.arange(len(dist_names))
    width = 0.25
    for i, (strat, vals) in enumerate(results.items()):
        bars = ax1.bar(x + i * width - width, vals, width,
                       label=strat, color=COLORS[i], edgecolor="black",
                       linewidth=0.5, hatch=HATCHES[i])

    ax1.set_xticks(x)
    ax1.set_xticklabels(dist_names, fontsize=8)
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan by Workload Shape")
    ax1.legend(fontsize=7)
    ax1.set_yscale("log")

    # Panel B: Speedup of Proposed vs FIFO
    speedups = [results["FIFO"][i] / results["Proposed"][i]
                for i in range(len(dist_names))]
    bars2 = ax2.bar(dist_names, speedups, color=COLORS[2], edgecolor="black",
                    linewidth=0.5)
    for bar, v in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                 f"{v:.0f}x", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Speedup (Proposed vs FIFO)")
    ax2.set_title("(b) Scheduling Benefit by Workload Shape")
    ax2.set_xticklabels(dist_names, fontsize=8)

    fig.suptitle("Workload Shape Impact on Scheduling\n"
                 "(500 Tasks, 2× RTX A5000, NVMe SSD)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_model_size_dist")


# ════════════════════════════════════════════════════════════════════════
# Insight: Skew Sensitivity (Zipf-like popularity)
# ════════════════════════════════════════════════════════════════════════

def plot_insight_skew_sensitivity():
    """Line chart: performance vs workload skew (Zipf parameter)."""
    models_list = ["small", "medium", "large", "xlarge"]
    total_tasks = 500
    # Zipf-like: model i gets proportion ~ 1/i^alpha
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    fifo_ms, grouped_ms, proposed_ms = [], [], []

    for alpha in alphas:
        if alpha == 0:
            # Uniform
            weights = [1.0] * len(models_list)
        else:
            weights = [1.0 / ((i + 1) ** alpha) for i in range(len(models_list))]
        total_w = sum(weights)
        fracs = [w / total_w for w in weights]
        tc = {}
        assigned = 0
        for i, m in enumerate(models_list):
            if i == len(models_list) - 1:
                tc[m] = total_tasks - assigned
            else:
                tc[m] = int(total_tasks * fracs[i])
                assigned += tc[m]

        fifo_ms.append(sim_fifo_makespan(tc))
        grouped_ms.append(sim_grouped_makespan(tc, prefetch=False))
        proposed_ms.append(sim_proposed_makespan(tc, num_gpus=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for ms_list, label, marker, color in [
        (fifo_ms, "FIFO", "o", COLORS[0]),
        (grouped_ms, "Grouped (no pf)", "s", COLORS[3]),
        (proposed_ms, "Proposed (2 GPU)", "D", COLORS[2]),
    ]:
        ax1.plot(alphas, ms_list, f"-{marker}", color=color, label=label,
                 linewidth=1.5, markersize=5)

    ax1.set_xlabel("Skew Parameter α (0=uniform, 3=highly skewed)")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("(a) Makespan vs Workload Skew")
    ax1.legend(fontsize=7)

    # Panel B: Speedup
    for ms_list, label, marker, color in [
        (grouped_ms, "Grouped (no pf)", "s", COLORS[3]),
        (proposed_ms, "Proposed (2 GPU)", "D", COLORS[2]),
    ]:
        speedups = [fifo_ms[i] / ms_list[i] for i in range(len(alphas))]
        ax2.plot(alphas, speedups, f"-{marker}", color=color, label=label,
                 linewidth=1.5, markersize=5)

    ax2.set_xlabel("Skew Parameter α")
    ax2.set_ylabel("Speedup vs FIFO")
    ax2.set_title("(b) Scheduling Benefit vs Skew")
    ax2.legend(fontsize=7)
    ax2.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)

    # Add annotation for distribution shape
    ax1.annotate("← uniform", xy=(0.1, fifo_ms[0]), fontsize=7, color="gray")
    ax1.annotate("concentrated →", xy=(2.2, fifo_ms[-2]), fontsize=7, color="gray")

    fig.suptitle("Workload Skew Sensitivity\n"
                 "(500 Tasks, 4 Models, Zipf-like Distribution, 2× RTX A5000)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_skew_sensitivity")


# ════════════════════════════════════════════════════════════════════════
# Insight: GPU Utilization Timeline
# ════════════════════════════════════════════════════════════════════════

def plot_insight_gpu_utilization():
    """Stacked area chart: GPU busy/idle/loading over time for each strategy."""
    task_counts = {"small": 200, "medium": 150, "large": 100, "xlarge": 50}
    models_list = ["small", "medium", "large", "xlarge"]

    def _trace_events(task_list):
        """Return list of (start, end, type, model) events."""
        events = []
        t = 0.0
        loaded = None
        idx = 0
        while idx < len(task_list):
            m = task_list[idx]
            if loaded != m:
                events.append((t, t + MODEL_LOAD_TIMES[m], "load", m))
                t += MODEL_LOAD_TIMES[m]
                loaded = m
            run_len = 1
            while idx + run_len < len(task_list) and task_list[idx + run_len] == m:
                run_len += 1
            inf = inference_time(m, run_len)
            events.append((t, t + inf, "infer", m))
            t += inf
            idx += run_len
        return events, t

    def _grouped_events(prefetch=False):
        """Grouped execution: models in order, optional prefetch."""
        order = johnsons_rule_order(task_counts) if prefetch else models_list
        events = []
        t = 0.0
        for i, m in enumerate(order):
            if i == 0:
                events.append((t, t + MODEL_LOAD_TIMES[m], "load", m))
                t += MODEL_LOAD_TIMES[m]
            else:
                switch = PCIE_TRANSFER_TIMES[m] if prefetch else MODEL_LOAD_TIMES[m]
                events.append((t, t + switch, "load", m))
                t += switch
            inf = inference_time(m, task_counts[m])
            events.append((t, t + inf, "infer", m))
            t += inf
        return events, t

    # Build task lists
    max_count = max(task_counts.values())
    fifo_list = []
    for i in range(max_count):
        for m in models_list:
            if i < task_counts[m]:
                fifo_list.append(m)

    rng = random.Random(42)
    random_list = []
    for m, n in task_counts.items():
        random_list.extend([m] * n)
    rng.shuffle(random_list)

    fifo_events, fifo_total = _trace_events(fifo_list)
    random_events, random_total = _trace_events(random_list)
    grouped_events, grouped_total = _grouped_events(prefetch=False)
    proposed_events, proposed_total = _grouped_events(prefetch=True)

    def _compute_fractions(events, total, num_bins=200):
        bins = np.linspace(0, total, num_bins + 1)
        loading = np.zeros(num_bins)
        inferring = np.zeros(num_bins)
        for start, end, etype, _ in events:
            for b in range(num_bins):
                b_start, b_end = bins[b], bins[b + 1]
                overlap = max(0, min(end, b_end) - max(start, b_start))
                if overlap > 0:
                    frac = overlap / (b_end - b_start)
                    if etype == "load":
                        loading[b] += frac
                    else:
                        inferring[b] += frac
        idle = np.clip(1.0 - loading - inferring, 0, 1)
        centers = (bins[:-1] + bins[1:]) / 2
        return centers, inferring, loading, idle

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)

    configs = [
        ("(a) FIFO (Alternating)", fifo_events, fifo_total),
        ("(b) Random (Shuffled)", random_events, random_total),
        ("(c) Grouped (No Prefetch)", grouped_events, grouped_total),
        ("(d) Proposed (Johnson + Prefetch)", proposed_events, proposed_total),
    ]

    for ax, (label, events, total) in zip(axes, configs):
        centers, inferring, loading, idle = _compute_fractions(events, total)
        ax.fill_between(centers, 0, inferring, color="#27ae60", alpha=0.8, label="Inference")
        ax.fill_between(centers, inferring, inferring + loading,
                        color="#e74c3c", alpha=0.7, label="Model Loading")
        ax.fill_between(centers, inferring + loading, 1.0,
                        color="#bdc3c7", alpha=0.5, label="Idle")
        ax.set_ylabel("GPU Util.")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label} — Total: {total:.0f}s", fontsize=10)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)")

    fig.suptitle("GPU Utilization Over Time\n"
                 "(500 Tasks — 0.6B / 8B / 32B / 120B, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "insight_gpu_utilization")


# ════════════════════════════════════════════════════════════════════════
# Insight: Prefetch Overlap Efficiency
# ════════════════════════════════════════════════════════════════════════

def plot_insight_prefetch_overlap():
    """What fraction of load time is hidden by prefetch vs inference/load ratio."""
    ratios = np.linspace(0.01, 5.0, 100)
    avg_load = np.mean([MODEL_LOAD_TIMES[m] for m in ["small", "medium", "large", "xlarge"]])
    avg_pcie = np.mean([PCIE_TRANSFER_TIMES[m] for m in ["small", "medium", "large", "xlarge"]])
    nvme_portion = avg_load - avg_pcie

    hidden_frac = []
    effective_switch = []
    for r in ratios:
        inf_time = r * avg_load
        overlap = min(inf_time, nvme_portion)
        residual = avg_load - overlap
        effective = max(residual, avg_pcie)
        hidden_frac.append((avg_load - effective) / avg_load * 100)
        effective_switch.append(effective)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Fraction hidden
    ax1.plot(ratios, hidden_frac, "-", color=COLORS[2], linewidth=2)
    ax1.fill_between(ratios, hidden_frac, alpha=0.2, color=COLORS[2])
    max_hidden = (avg_load - avg_pcie) / avg_load * 100
    ax1.axhline(y=max_hidden, color="gray", linestyle="--", linewidth=0.8)
    ax1.text(4.0, max_hidden + 1, f"Max: {max_hidden:.0f}%", fontsize=8, color="gray")
    ax1.axvline(x=1.0, color=COLORS[0], linestyle=":", linewidth=1,
                label="Inference = Load Time")
    ax1.set_xlabel("Inference / Load Time Ratio")
    ax1.set_ylabel("Load Time Hidden by Prefetch (%)")
    ax1.set_title("(a) Prefetch Overlap Efficiency")
    ax1.legend(fontsize=8)

    # Mark where our models fall (100 tasks each)
    for m, color_m in [("small", "#3498db"), ("medium", "#e67e22"), ("large", "#9b59b6"), ("xlarge", "#e74c3c")]:
        inf = inference_time(m, 100)
        r = inf / MODEL_LOAD_TIMES[m]
        hidden = min(inf, MODEL_LOAD_TIMES[m] - PCIE_TRANSFER_TIMES[m])
        h_pct = hidden / MODEL_LOAD_TIMES[m] * 100
        ax1.plot(r, h_pct, "o", color=color_m, markersize=8, zorder=5,
                 markeredgecolor="black", markeredgewidth=1)
        ax1.annotate(MODELS[m]["name"].split("/")[1], (r, h_pct),
                     textcoords="offset points", xytext=(8, -5),
                     fontsize=7, color=color_m)

    # Panel B: Effective switch cost
    ax2.axhline(y=avg_load, color=COLORS[0], linestyle="--", linewidth=1.5,
                label=f"No Prefetch ({avg_load:.0f}s)")
    ax2.plot(ratios, effective_switch, "-", color=COLORS[2], linewidth=2,
             label="With Prefetch")
    ax2.axhline(y=avg_pcie, color="gray", linestyle=":", linewidth=0.8)
    ax2.text(4.0, avg_pcie + 0.5, f"PCIe floor: {avg_pcie:.2f}s",
             fontsize=8, color="gray")
    ax2.set_xlabel("Inference / Load Time Ratio")
    ax2.set_ylabel("Effective Switch Cost (s)")
    ax2.set_title("(b) Effective Model Switch Cost")
    ax2.legend(fontsize=8)

    fig.suptitle("Prefetch Overlap: When Does It Help?\n"
                 "(Average across 4 Models, RTX A5000, NVMe SSD)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_prefetch_overlap")


# ══════════���═════════════════════════════════════════════════════════════
# Insight: Break-Even Analysis
# ════════════════════════════════════════════════════════════════════════

def plot_insight_break_even():
    """At what batch size does scheduling overhead pay for itself?"""
    scheduling_overheads = [0.01, 0.1, 1.0, 5.0]
    batch_sizes = np.arange(2, 1002, 2)
    mix = STANDARD_WORKLOAD["mix"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Net benefit curves
    for overhead, ls in zip(scheduling_overheads, ["-", "--", "-.", ":"]):
        net_benefit = []
        for n in batch_sizes:
            tc = {m: max(1, int(n * f)) for m, f in mix.items()}
            tc["small"] += n - sum(tc.values())
            fifo = sim_fifo_makespan(tc)
            prop = sim_proposed_makespan(tc, num_gpus=1)
            net_benefit.append(fifo - prop - overhead)
        ax1.plot(batch_sizes, net_benefit, ls, color=COLORS[2], linewidth=1.5,
                 label=f"Overhead = {overhead}s")

    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.set_xlabel("Batch Size (tasks)")
    ax1.set_ylabel("Net Time Saved (s)")
    ax1.set_title("(a) Net Benefit of Scheduling")
    ax1.legend(fontsize=7)
    ax1.set_xlim(0, 1000)

    # Panel B: Break-even batch size vs overhead
    overheads_range = np.logspace(-3, 2, 50)
    break_even_sizes = []
    for overhead in overheads_range:
        be = None
        for n in range(2, 5001, 2):
            tc = {m: max(1, int(n * f)) for m, f in mix.items()}
            tc["small"] += n - sum(tc.values())
            fifo = sim_fifo_makespan(tc)
            prop = sim_proposed_makespan(tc, num_gpus=1)
            if fifo - prop > overhead:
                be = n
                break
        break_even_sizes.append(be if be else 5000)

    ax2.plot(overheads_range, break_even_sizes, "-", color=COLORS[2], linewidth=2)
    ax2.fill_between(overheads_range, break_even_sizes, 5000,
                     alpha=0.15, color=COLORS[2], label="Scheduling wins")
    ax2.fill_between(overheads_range, 0, break_even_sizes,
                     alpha=0.15, color=COLORS[0], label="FIFO wins")
    ax2.set_xlabel("Scheduling Overhead (s)")
    ax2.set_ylabel("Break-Even Batch Size")
    ax2.set_title("(b) Break-Even Point")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)

    ax2.axvline(x=0.001, color=COLORS[4], linestyle="--", linewidth=1)
    ax2.text(0.0015, 100, "Actual\n(~1ms)", fontsize=8, color=COLORS[4])

    fig.suptitle("Break-Even Analysis: When Does Scheduling Pay Off?\n"
                 "(4 Models, Standard Mix, 1 GPU, RTX A5000)",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, "insight_break_even")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

PLOT_MAP = {
    "1":   ("Exp 1: Strategy Makespan", plot_exp1),
    "2":   ("Exp 2: Model Switching", plot_exp2),
    "3":   ("Exp 3: Concurrency Mode", plot_exp3),
    "4":   ("Exp 4: Prefetch Effect", plot_exp4),
    "5s":  ("Exp 5: Standard Workload", plot_exp5_standard),
    "5c":  ("Exp 5: Scalability", plot_exp5_scalability),
    "6":   ("Exp 6: Model Size", plot_exp6),
    "7":   ("Exp 7: Stress Test", plot_exp7),
    "8m":  ("Exp 8: Batch Scheduling (Main)", plot_exp8_main),
    "8g":  ("Exp 8: Gantt Timeline", plot_exp8_gantt),
    "8gpu": ("Exp 8: GPU Scalability", plot_exp8_gpu_scalability),
    "8j":  ("Exp 8: Johnson's Rule", plot_exp8_johnsons),
    "8b":  ("Exp 8: Batch Sensitivity", plot_exp8_batch_sensitivity),
    "iw":  ("Insight: Waterfall Breakdown", plot_waterfall_breakdown),
    "is":  ("Insight: Switching Cost Curve", plot_switching_cost_curve),
    "ih":  ("Insight: GPU×Model Heatmap", plot_insight_gpu_model_heatmap),
    "ist": ("Insight: Storage Tier", plot_insight_storage_tier),
    "ip":  ("Insight: PCIe Bandwidth", plot_insight_pcie_bandwidth),
    "imc": ("Insight: Model Count Sweep", plot_insight_model_count_sweep),
    "imd": ("Insight: Model Size Distribution", plot_insight_model_size_dist),
    "isk": ("Insight: Skew Sensitivity", plot_insight_skew_sensitivity),
    "igu": ("Insight: GPU Utilization", plot_insight_gpu_utilization),
    "ipo": ("Insight: Prefetch Overlap", plot_insight_prefetch_overlap),
    "ibe": ("Insight: Break-Even Analysis", plot_insight_break_even),
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate analytical dissertation figures")
    parser.add_argument("--exp", nargs="*", default=None,
                        help="Experiment keys to plot (e.g., 1 4 8m iw). Default: all")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating analytical dissertation figures")
    print(f"  Output: {FIGURES}")
    print("=" * 60)

    if args.exp:
        selected = []
        for e in args.exp:
            e = e.strip()
            if e in PLOT_MAP:
                selected.append(e)
            else:
                for key in PLOT_MAP:
                    if key == e or key.startswith(e):
                        selected.append(key)
    else:
        selected = list(PLOT_MAP.keys())

    for key in selected:
        name, fn = PLOT_MAP[key]
        print(f"\n{name}...")
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  [error] {exc}")
            traceback.print_exc()

    print(f"\nDone. {len(selected)} figures saved to {FIGURES}/")


if __name__ == "__main__":
    main()
