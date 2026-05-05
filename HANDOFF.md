# `experiment/pr328-merge` — Handoff Notes

This branch merges upstream PR #328 (auto-download on deploy) into the
`feat/batch-scheduler` work. It is **partially hardened** — the happy
path is e2e-verified end-to-end (incl. TP=4 on Qwen3-32B with sllm-store
fast load), but several known issues remain. **Read this file before
merging anything from this branch upstream or to `main`.**

---

## ✅ What is done & verified

### Tier 2 — Hygiene
- **Lifespan single-source-of-truth refactor**
  (`sllm/api_gateway.py`, `sllm/cli/_cli_utils.py`):
  `storage_manager` is now injected via `app.state` only. Removed the
  `create_app(storage_manager=...)` kwarg and the `UnboundLocalError`
  defensive fallback. Removed duplicate assignment in `_cli_utils.py`.
- **Dedup of `_ensure_deployment_ready*`** (`sllm/batch_scheduler.py`):
  collapsed the prefetch-lane and single-GPU paths into one function
  with optional `max_replicas`. Net **−110 lines**.
- **HF hub cache fallback for TP detection**
  (`sllm/batch_scheduler.py::_get_checkpoint_size_bytes`):
  if a model isn't in `storage_path` yet, fall back to
  `~/.cache/huggingface/hub/...` so TP auto-detection picks the right
  size on first run instead of defaulting to TP=1.
- **TP propagation through download flow**
  (`sllm/storage_manager.py::download_model_on_node`):
  now accepts `tensor_parallel_size`, passes `--tensor-parallel-size`
  to `sllm-store save`, and requests `gpu=tp` from Pylet. Without
  this, 32B models silently saved as TP=1 and crashed at load.
- **4 red unit tests fixed** (autoscaler, command_builder, router):
  - Autoscaler tests now pass `initial_status='active'` because
    `update_desired_replicas` is gated on `status='active'` (PR #328
    changed the default to `pending`).
  - `test_minimal_config` updated to expect `--max-model-len 4096`
    (deliberate default — Qwen3-8B's native 40k context blows
    24GB cards).
  - Router tests updated to new defaults (`cold_start_timeout=600.0`,
    `request_timeout=600.0`).
  - 3 new tests added for the `--load-format serverless_llm` rewrite.
  - **All 73 touched unit tests green.**

### Tier 1 — sllm-store fast load wired into vLLM
- **`build_vllm_command` now activates the fast loader**
  (`sllm/command_builder.py`):
  - Added `_has_sllm_store_shards(model_dir, tp)` — checks all
    `rank_<i>/tensor.data_0` shards exist for the requested TP.
  - When shards are present, rewrites the model arg to the **local
    directory path** (the vLLM patch's `ServerlessLLMLoader`
    asserts `os.path.isdir`) and adds `--load-format serverless_llm`.
  - Falls back to HF id + safetensors loader when shards are missing
    or partial — so the patch isn't a hard dependency for users who
    haven't applied it.
- **A/B verified on Qwen3-32B + TP=4**:
  | Path | Weight load | Total vLLM load |
  |------|-------------|-----------------|
  | HF safetensors | 53.70 s | 55.38 s |
  | sllm-store fast | **1.92 s** | 1.92 s |
  | Speedup | **~28×** | — |
  Both runs reach steady-state ≈21.4 GiB/GPU and produce valid output.

### End-to-end coverage
- Qwen3-0.6B batch e2e (single GPU): pass.
- Qwen3-32B uncached e2e (TP=4 auto-detected → background download →
  load → infer): pass.
- Qwen3-32B fast-load e2e (sllm-store shards on disk → mmap → infer):
  pass, with the 28× speedup above.

---

## ⚠️ Unresolved issues — READ BEFORE LANDING

### 1. Three patches live in the venv, **not in the repo**

These are required for the fast-load path to work, and **will be lost
the next time anyone re-runs `pip install`**. They must be baked into
`sllm_store/vllm_patch/sllm_load.patch` (or equivalent) before anyone
relies on this branch in CI / a fresh checkout.

| Patch | File in venv | What it does |
|-------|--------------|--------------|
| (A) | `sllm_store/torch.py` (lines 92, 103) | Replaced hardcoded `127.0.0.1:8073` with `os.getenv("SLLM_STORE_ENDPOINT", "127.0.0.1:8073")`. Pylet allocates a dynamic gRPC port (15600 in our test) and exposes it via `SLLM_STORE_ENDPOINT`; without this, `confirm_model_loaded` and `load_into_cpu` connect to the wrong port and hang / fail. |
| (B) | `vllm/model_executor/model_loader/sllm_loader.py` (line 114-ish) | After loading state_dict from sllm-store, filter keys ending in `_k_scale`, `_v_scale`, `_q_scale`, `_prob_scale` before the strict missing-keys check. vLLM 0.15+ unconditionally registers these FP8 KV-cache scale buffers on every attention layer; sllm-store didn't save them (correctly so — they're 1.0 for bf16/fp16 inference and re-initialized in `initialize_model`). Without this filter, every load raises `ValueError: Missing keys (..._k_scale, ..._v_scale, ...)`. |
| (C) | Runtime config | `SLLM_STORE_MEM_POOL_SIZE` must be ≥ largest model size (e.g. `72GB` for 32B). Without this, `PinnedMemoryPool` segfaults under "489 buffers needed, 128 available". The size string must be in `<int>GB` format — bare integers are rejected by `to_num_bytes`. **Where to set this is a config UX question** — currently the operator has to set it manually before `sllm-store start`. |

**Action items:**
- [ ] Bake (A) and (B) into `sllm_store/vllm_patch/sllm_load.patch`
      so `bash sllm_store/vllm_patch/patch.sh` reproduces them on a
      fresh venv.
- [ ] Add a setup-time check that warns if `SLLM_STORE_MEM_POOL_SIZE`
      is unset or smaller than the largest deployed model.
- [ ] Optionally upstream (A) and (B) to `ServerlessLLM/ServerlessLLM`
      and `vllm-project/vllm` respectively.

### 2. Tier 3 of the cleanup plan is **not started**

See `.claude/plans/concurrent-weaving-mist.md` for the full plan. Tier 3
covers:
- **Error-path integration tests** — bogus model name, worker death
  mid-download, download timeout. None of these have automated
  regression coverage yet; the happy path was the only thing exercised.
- **Real-data v5→v6 migration test** — schema-level test exists, but
  no fixture-based test that copies a populated `state_v5.db` and
  asserts row preservation through the upgrade.
- **Fail-loud sweep** — several `logger.warning(...) + silent fallback`
  spots in `sllm/batch_scheduler.py` and `sllm/storage_manager.py`
  should probably be `RuntimeError` (e.g. "could not pick download
  node"). They were left as-is to avoid scope creep.

### 3. Remaining code smells

- `sllm/storage_manager.py::download_model_on_node` allocates `gpu=tp`
  but does **not** double-check that the chosen node actually has `tp`
  free GPUs at submit time. Pylet may reject; we propagate the error
  but don't retry on a different node.
- `_get_checkpoint_size_bytes` HF hub fallback walks the entire
  snapshot dir on every call — fine for current scale, may want
  caching if the deployment list grows.
- `default download_timeout` in `init_storage_manager` is 600 s. May
  be too tight for 70B+ on slow disks or rate-limited HF accounts.

### 4. Untested scenarios

- **Multi-node download** — only tested single-node with one Pylet.
  The download-node selection logic exists but the code path that
  fires when the chosen node ≠ the requesting node hasn't been
  exercised end-to-end.
- **Model that mixes TP sizes across deployments** — e.g. saving a
  model with TP=2 and later requesting TP=4 of the same model. The
  shard check would fail and fall back to HF, but the storage manager
  doesn't currently warn that the on-disk version is stale.
- **vLLM upstream version drift** — patch (B) was written against
  vLLM 0.15. Newer vLLM may register additional buffer types under
  different naming conventions; the `SCALE_SUFFIXES` tuple may need
  expansion.

---

## File map (what changed, where to look)

| File | Tier | Why |
|------|------|-----|
| `sllm/api_gateway.py` | 2.1 | Lifespan → `app.state`-only injection |
| `sllm/cli/_cli_utils.py` | 2.1 | Drop duplicate `app.state` assignment |
| `sllm/batch_scheduler.py` | 2.2 + bugfix | Function dedup; HF hub size fallback; TP propagation in `_await_model_download` |
| `sllm/storage_manager.py` | bugfix | `tensor_parallel_size` arg in `download_model_on_node`; `--tensor-parallel-size` flag passed through |
| `sllm/command_builder.py` | 1.2 | `_has_sllm_store_shards` helper; rewrite model arg + `--load-format serverless_llm` flag |
| `tests/unit/test_router.py` | 2.3 | Updated to new default timeouts |
| `tests/unit/test_command_builder.py` | 2.3 | Updated to `--max-model-len 4096` default; 3 new tests for shard-aware load-format |
| `tests/unit/test_autoscaler.py` | 2.3 | Pass `initial_status='active'` so `update_desired_replicas` SQL gate (`WHERE status='active'`) actually fires |

---

## Reproducing the e2e tests (Qwen3-32B + TP=4)

Prereqs:
1. Apply venv patches (A) and (B) from the table above (or run
   `patch.sh` once it's been updated to include them).
2. `export SLLM_STORE_MEM_POOL_SIZE=72GB`.
3. `sllm-store start --mem-pool-size 72000000000` (in its own terminal).
4. `sllm start --controller-config <yourconf>` and a Pylet worker.

Then:
- Cold (uncached) run hits the auto-download flow and saves shards to
  `<storage_path>/Qwen/Qwen3-32B/rank_{0..3}/tensor.data_0`.
- Warm run (shards present) takes the fast-load path. Look for
  `Model loading took 0.01 GiB memory and ~2 seconds` in the vLLM
  instance log under `~/.pylet/logs/<instance-uuid>.1`.

If you see `Loading weights took ~50 seconds` instead, the fast-load
path didn't activate — most likely because the model arg wasn't
rewritten to a local path, the patch (B) wasn't applied, or
`SLLM_STORE_ENDPOINT` wasn't honored (patch A).
