
Full code repository: https://github.com/jason2594746793/ServerlessLLM

## Source Code

`source_code/` contains the 9 files added or modified in the ServerlessLLM framework. All other framework files are unchanged from the upstream v1-beta release.

| File | Status |
|---|---|
| `batch_scheduler.py` | New — core scheduler (model grouping, Johnson's Rule, prefetch, multi-GPU) |
| `database.py` | Extended — batch job and file schema |
| `api_gateway.py` | Extended — OpenAI-compatible batch endpoints |
| `storage_manager.py` | Extended — checkpoint prefetch integration |
| `router.py`, `autoscaler.py`, `reconciler.py`, `command_builder.py`, `cli/_cli_utils.py` | Extended — minor batch-aware changes |



## Setup

```bash
pip install serverlessllm

# 2. Download models (once)
python benchmarks/download_models.py

# 3. Start the cluster (head node + worker + sllm-store)
bash scripts/start_cluster_batch.sh
# Logs: pylet_head_batch.log, pylet_worker_batch.log, sllm_head_batch.log

# 4. Verify the cluster is healthy
curl http://localhost:8343/status
```

> **Note for prefetch experiments:** set `--mem-pool-size` to at least the size of the
> largest model checkpoint (≥20 GB for 8B models, ≥72 GB for 32B models) when starting
> sllm-store, otherwise prefetch silently falls back to cold loading.


## Running Experiments

Each script is self-contained and handles cold-start verification between runs automatically.

```bash
python benchmarks/exp1_strategy_makespan.py    # Exp 1 + Exp 3 n=501
python benchmarks/exp3a_scalability_anchor.py  # Exp 3 n=99
python benchmarks/exp4_prefetch.py             # Exp 2
python benchmarks/exp_jr_all_perms.py          # Exp 4
python benchmarks/exp_jr_task_sensitivity.py   # Exp 4b
python benchmarks/exp_prefetch_2model.py       # Exp 5 (NVMe row)
```

Results are written to `benchmarks/results/<exp_name>/`.


