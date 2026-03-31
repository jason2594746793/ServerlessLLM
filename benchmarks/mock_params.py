"""
benchmarks/mock_params.py — Centralized mock system parameters for
analytical benchmarks and dissertation figures.

All experiments and plotting scripts should import from here rather than
hardcoding values.  Changing a number here propagates everywhere.

Values are calibrated from real measurements on the target hardware:
  - 2× NVIDIA RTX A5000 (24 GB VRAM each)
  - 2× AMD EPYC 7453 (128 PCIe Gen4 lanes per socket)
  - 2 TB NVMe SSD (sequential read ~3 GB/s)
  - 512 GB DDR4 system RAM (2 NUMA nodes)
"""

# ════════════════════════════════════════════════════════════════════════
# Hardware
# ════════════════════════════════════════════════════════════════════════

NUM_GPUS = 2                        # GPUs used in default experiments
GPU_VRAM_GB = 24                    # per GPU (RTX A5000)
GPU_VRAM_SMALL_GB = 8               # hypothetical small GPU for comparison
GPU_VRAM_LARGE_GB = 80              # hypothetical A100 for comparison

SYSTEM_RAM_GB = 512                 # total DDR4
SYSTEM_RAM_SMALL_GB = 64            # small-system scenario
SYSTEM_RAM_LARGE_GB = 1024          # large-system scenario

PINNED_MEM_POOL_GB = 72             # sllm-store pinned memory pool
PINNED_MEM_POOL_SMALL_GB = 20       # fits up to ~7B model
PINNED_MEM_POOL_LARGE_GB = 160      # fits up to ~70B model

NVME_READ_GBps = 3.0                # sequential read throughput (GB/s)
PCIE_BW_GBps = 32.0                 # PCIe Gen4 x16 per GPU (GB/s)
PCIE_LANES_PER_SOCKET = 128         # AMD EPYC 7453

# ════════════════════════════��═══════════════════════════════════════════
# Models — checkpoint sizes and names
# ════════════════════════════════════════════════════════════════════════

MODELS = {
    "small": {
        "name": "Qwen/Qwen3-0.6B",
        "params_B": 0.6,
        "checkpoint_GB": 1.2,
    },
    "medium": {
        "name": "Qwen/Qwen3-8B",
        "params_B": 8.0,
        "checkpoint_GB": 15.4,
    },
    "large": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "params_B": 32.0,
        "checkpoint_GB": 60.0,
    },
    "xlarge": {
        "name": "DeepSeek/DeepSeek-V2-Lite-Chat",
        "params_B": 120.0,
        "checkpoint_GB": 224.0,
    },
}

# ════════════════════════════════════════════════════════════════════════
# Model loading times (SSD → CPU pinned mem → GPU, seconds)
# Measured end-to-end via sllm-store on NVMe SSD
# ════════════════════════════════════════════════════════════════════════

MODEL_LOAD_TIMES = {
    "small":  15.0,     # 0.6B  — ~1.2 GB / 3 GB/s + overhead
    "medium": 55.0,     # 8B    — ~15.4 GB / 3 GB/s + overhead
    "large":  180.0,    # 32B   — ~60 GB / 3 GB/s + overhead
    "xlarge": 580.0,    # 120B  — ~224 GB / 3 GB/s + overhead
}

# PCIe transfer only (CPU pinned mem → GPU), used after prefetch
PCIE_TRANSFER_TIMES = {
    "small":  0.04,     # 1.2 GB / 32 GB/s
    "medium": 0.48,     # 15.4 GB / 32 GB/s
    "large":  1.88,     # 60 GB / 32 GB/s
    "xlarge": 7.00,     # 224 GB / 32 GB/s
}

# Fraction of full load time that PCIe-only transfer represents
PREFETCH_RESIDUAL_FRACTION = 0.15   # ~15% of full load after prefetch

# ════════════════════════════════════════════════════════════════════════
# Inference latency (per batch, seconds)
# Measured with vLLM continuous batching, max_tokens=512
# Each "batch" is up to MAX_CONCURRENT_PER_GPU tasks processed together.
# ════════════════════════════════════════════════════════════════════════

MODEL_INFERENCE_TIMES = {
    "small":  3.0,      # 0.6B  — ~512 tokens, fast decoding
    "medium": 10.0,     # 8B    — ~512 tokens
    "large":  30.0,     # 32B   — ~512 tokens
    "xlarge": 90.0,     # 120B  — ~512 tokens, tensor parallel
}

MAX_CONCURRENT_PER_GPU = 50         # vLLM continuous batching capacity

# ════════════════════════════════════════════════════════════════════════
# Scheduling overhead
# ════════════════════════════════════════════════════════════════════════

MODEL_UNLOAD_TIME = 2.0             # GPU memory free (fast)
PREFETCH_OVERLAP_FRACTION = 0.85    # fraction of load time saved by prefetch

# ════════════════════════════════════════════════════════════════════════
# Johnson's Rule estimation parameters
# (used in batch_scheduler.py for I/O vs compute classification)
# ════════════════════════════════════════════════════════════════════════

JOHNSONS_BASE_TASK_TIME_S = 0.5     # baseline per-task time for ~1B model
JOHNSONS_BASE_MODEL_BYTES = 2e9     # reference model size (~1B in fp16)
JOHNSONS_NVME_READ_BPS = 3e9        # NVMe throughput for load time estimate

# ═════════════════════════════════════════════════════════════��══════════
# Default workload mixes
# ════════════════════════════════════════════════════════════════════════

STANDARD_WORKLOAD = {
    "total_tasks": 500,
    "mix": {"small": 0.40, "medium": 0.30, "large": 0.20, "xlarge": 0.10},
}
