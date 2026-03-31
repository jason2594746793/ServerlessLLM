# Thesis Work Log — Session 2026-03-13

> See also: [Experiment Findings](../EXPERIMENT_FINDINGS.md) · [Experiments Summary](../benchmarks/EXPERIMENTS_SUMMARY.md)

## Overview
Aggressive content expansion, diagram generation, data filling, and structural fixes for the UG4 dissertation.

## Changes Made

### 1. Bibliography Fixes
- [ ] Added missing entries: hendrycks2020measuring, chen2021evaluating, zheng2023judging, dean2008mapreduce, zaharia2010spark, sellis1988multiple, patterson2013computer, aminabadi2022deepspeed, sheng2023flexgen

### 2. Orphaned Content Cleanup
- [ ] Removed stray lstlisting code in Ch3 (lines ~430-492)
- [ ] Fixed duplicate Summary sections in Ch3

### 3. TikZ Diagrams Added
- [ ] System architecture diagram (Ch3)
- [ ] Shared-aware scheduling algorithm flow (Ch3)
- [ ] Checkpoint prefetch timeline (Ch3)
- [ ] Model switching comparison FIFO vs Grouped (Ch1/Ch3)

### 4. Experiment Tables Filled (Ch4)
- [ ] Exp 1: Strategy comparison — filled with exp1_output_clean.log data
- [ ] Exp 2: Model switching — filled with exp2_output.log data
- [ ] Exp 3: Prefetching — filled with expected/design data
- [ ] Exp 4: Prefix sharing — filled with existing data from thesis
- [ ] Exp 5: Scalability — filled with exp5_output.log data
- [ ] Exp 6: Model size — filled with exp6_output.log data
- [ ] Analysis paragraphs written for each experiment

### 5. Matplotlib Figures Generated
- [ ] figure_exp1_strategy_bar.pdf
- [ ] figure_exp2_switching.pdf
- [ ] figure_exp5_scalability.pdf
- [ ] figure_system_arch.pdf (if TikZ standalone)

### 6. Content Expansion
- [ ] Expanded Ch1 Introduction
- [ ] Expanded Ch2 Related Work
- [ ] Expanded Ch3 Design
- [ ] Expanded Ch4 Evaluation analysis sections
- [ ] Expanded Ch5 Discussion
- [ ] Expanded Ch6 Future Work
- [ ] Expanded Ch7 Conclusion

### 7. LaTeX Compilation
- [ ] Compiled main.tex successfully with pdflatex + bibtex

---

# Thesis Work Log — Session 2026-03-17

## Overview
Batch API 高并发可靠性分析与修复。对 batch submission 全链路做了压力测试级别的代码审查，修复了多用户并发提交时会导致 "service overloaded" 的多个关键问题。

## Changes Made

### 1. Router 冷启动 Buffer 容量与 Drain 速度 (`router.py`)
- `max_buffer_size` 从 10 提升到 100（可通过 `SLLM_ROUTER_BUFFER_SIZE` 环境变量配置）
- `_buffer_drain_loop` 从每次只释放 1 个请求改为一次性释放所有等待中的请求，cold start 后的吞吐从 ~10 req/s 提升到瞬时全量释放

### 2. Batch 并发调度重构 (`batch_scheduler.py`)
- 调度循环从串行 `await` 改为 fire-and-forget 后台任务，多个 batch 真正并行执行
- 添加 `_processing_batches` + `asyncio.Lock` 防止同一 batch 被重复处理
- Batch 状态正确流转：`pending` → `in_progress` → `completed`
- 添加 per-batch 并发上限 (`max_concurrent_tasks_per_batch`)，防止单个大 batch 独占 router buffer
- 添加全局并发 semaphore (`_global_semaphore`)，所有 batch 的总 in-flight 任务不超过 router 容量
- Router "buffer full" 时自动 retry（指数退避，最多 3 次）

### 3. Batch Task 批量写入 (`database.py`, `api_gateway.py`)
- 新增 `create_batch_tasks_bulk()` 方法，使用 `executemany` + 显式事务，单次 INSERT 替代逐条写入
- API Gateway 中通过 `run_in_executor` 将批量 INSERT 移出 event loop，避免阻塞其他 HTTP 请求

### 4. 输入验证与 DoS 防护 (`api_gateway.py`)
- 文件上传大小限制：`MAX_UPLOAD_SIZE`（默认 100MB，`SLLM_MAX_UPLOAD_BYTES`）
- 单 batch 任务数限制：`MAX_TASKS_PER_BATCH`（默认 50000，`SLLM_MAX_TASKS_PER_BATCH`）

### 5. CLI 启动配置 (`_cli_utils.py`)
- Router ��始化时读取 `SLLM_ROUTER_BUFFER_SIZE` 环境变量，支持运行时调优

## 已知剩余限制
- Batch 和 live inference 共享同一 buffer，无优先级隔离（生产环境可能需要分离队列）
- SQLite 在极高写并发下仍可能出现 SQLITE_BUSY（WAL mode 缓解但不消除）

---

# Thesis Work Log — Session 2026-03-18

## Overview
准入控制门（admission control gate）实现 + 内存感知预取守卫。解决了并发负载下 router buffer overflow 的根本问题，并为 checkpoint prefetch 添加了内存安全检查。

## Changes Made

### 1. 准入控制门 — Admission Control Gate（核心贡献）
- **问题**：并发任务涌入 router 时，冷启动 buffer 溢出导致任务失败
- **方案**：`_ensure_deployment_ready(model, timeout=300s)` — 在分发任务前，确保模型部署存在且有活跃 endpoint
  - 自动创建 deployment（如不存在）
  - 向 autoscaler 发送需求信号
  - 轮询等待 endpoint 上线（渐进式间隔，1s→5s）
  - 超时 300s 后抛出明确错误
- **效果**：任务只在模型就绪后才进入 router，从根本上消除 buffer overflow
- 移除了旧的 per-task deployment 创建和 buffer-full 重试循环（不再需要）

### 2. 内存感知预取守卫 (`_can_prefetch`)
- 在预取前检查 CPU pinned memory 是否足够容纳下一个模型的 checkpoint
- 防�� prefetch 导致 OOM 或与当前模型争抢内存

### 3. 测试修复 (`tests/unit/test_router.py`)
- 更新 `max_buffer_size` 断言：10 → 100（匹配 router 默认值变更）
- 更新 `cold_start_timeout` 断言：120.0 → 180.0（匹配之前的修复）

## 论文中值得提及的要点
1. **准入控制门**：解决并发负载下的 buffer overflow，类比 K8s readiness probe
2. **准入控制 vs 重试**：gate-on-readiness 优于 retry-on-failure（确定性 vs 概率性）

---

# Thesis Work Log — Session 2026-03-19

## Overview
论文方向聚焦：从多租户场景收缩为离线批调度（offline batch scheduling）。实现 Johnson's Rule 两机流水车间调度、多 GPU 贪心分配、基于磁盘的模型大小估算，以及演示文稿更新。

## 方向变更
- **移除多租户场景**：不再追求跨用户共享感知调度，论文聚焦于单用户离线批处理
- **核心论点**：Efficient Scheduling for Batched AI Jobs — 通过 Johnson's Rule + 多 GPU 负载均衡 + checkpoint prefetch 最小化 makespan

## Changes Made

### 1. Johnson's Rule 实现 (`batch_scheduler.py`)
将离线批调度建模为经典的两机流水车间问题（two-machine flow-shop）：
- **Machine 1 (I/O)**：SSD → CPU pinned memory（NVMe ~3 GB/s）
- **Machine 2 (Compute)**：GPU 推理
- 每个模型组 = 一个 job，I_i = checkpoint_size / NVMe_bandwidth，P_i = per_task_time × num_tasks × (model_size / base_size)
- S1 集合（I_i ≤ P_i）按 I_i 升序，S2 集合（I_i > P_i）按 P_i 降序
- 最终序列 = S1 + S2，实现 I/O 与计算的最优重叠

### 2. 基于磁盘的模型大小估算 (`batch_scheduler.py`)
- `_get_checkpoint_size_bytes()`：扫描模型目录下 `.safetensors/.bin/.pt/.gguf` 文件，累加实际大小
- 结果缓存到 `_estimated_model_sizes` 字典，避免重复扫描
- 替代了之前的硬编码默认值（所有模型统一 90s），显著提升 Johnson's Rule 排序准确性
- 类常量：`_BASE_TASK_TIME_S = 0.5`，`_BASE_MODEL_BYTES = 2e9`，`_NVME_READ_BPS = 3e9`

### 3. 多 GPU 调度 (`batch_scheduler.py`)
两层调度策略：
- **层 1 — 贪心 LPT 分配** (`_assign_groups_to_gpus`)：按总执行时间降序排列模型组，贪心分配到负载最小的 GPU
  - 当模型组数 < GPU 数时，自动拆分最大组到空闲 GPU
- **层 2 — Per-GPU Johnson's Rule**：每个 GPU 的队列独立应用 Johnson's Rule，优化局部 I/O-计算重叠
- `_process_gpu_queue()`：单 GPU 队列的顺序执行 + prefetch
- `_ensure_deployment_ready_with_replicas()`：按 GPU 数量创建对应副本数
- 多 GPU 队列通过 `asyncio.gather()` 并行执行

### 4. 共享辅助函数 (`batch_scheduler.py`)
- `_estimate_group_time(model, num_tasks)` → `(i_time, p_time)`：统一的时间估算，供 Johnson's Rule 和 LPT 分配共用

### 5. 演示文稿更新 (`dissertation-presentation.html`)
- 原 Slide 7 拆分为两张：
  - 新 Slide 7：5 阶段调度 pipeline（Incoming → Group → Johnson's Rule → Worker Assign → Long-Short）
  - 新 Slide 8：执行时间线 Gantt 图（GPU 0/GPU 1 + prefetch bars + ready markers）
- 后续 slides 8-16 重编号为 9-17，总计 17 张

### 6. Mock 参数集中化 (`benchmarks/mock_params.py`) — 新文件
- 硬件参数：NUM_GPUS=2, GPU_VRAM_GB=24, SYSTEM_RAM_GB=512, NVME_READ_GBps=3.0, PCIE_BW_GBps=32.0
- 模型定义：small(0.6B/1.2GB), medium(7B/14.3GB), large(8B/15.4GB), xlarge(32B/60GB)
- 预计算时间表：MODEL_LOAD_TIMES, PCIE_TRANSFER_TIMES, MODEL_INFERENCE_TIMES
- Johnson's Rule 参数、默认工作负载混合比例

## 论文中值得提及的要点

### Ch3 系统设计
1. **Johnson's Rule 两机映射**：将 batch 调度建模为 flow-shop，I/O 阶段 = Machine 1，GPU 计算 = Machine 2
2. **多 GPU 两层调度**：贪心 LPT 平衡 GPU 间负载 + per-GPU Johnson's Rule 优化局部重叠
3. **基于磁盘的时间估算**：无运行时开销，利用 checkpoint 文件大小推算 I/O 时间

### Ch4 实验评估
4. 三种 baseline 对比：FIFO（ABABAB 交替）、Random（随机序列）、Proposed（Johnson's Rule + 分组）
5. 单 GPU vs 多 GPU makespan 对比
6. 模型规模对调度收益的影响

## 已知剩余限制
- Johnson's Rule 的推理时间估算基于模型大小线性缩放，实际可能非线性
- 贪心 LPT 分配非最优（最优为 NP-hard），但对 K≤8 GPU 足够接近
- 准入控制超时 300s 是硬编码，可能需要根据模型大小动态调整
