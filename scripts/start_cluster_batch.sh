#!/bin/bash
set -e

# === Start Cluster with Batch Processing ENABLED ===
# This script starts the full ServerlessLLM stack with the Batch Scheduler active.
# Use this for testing batch jobs, DAG dependencies, and scheduler optimizations.

# MPS Workaround (User specific)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_vllm

# Fix: CUDA toolkit stub shadows real driver (since CUDA toolkit update 2026-03-12)
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Activate virtual environment explicitly
source .venv_new/bin/activate

# Configuration
PYTHON=.venv_new/bin/python
PYLET_BIN=.venv_new/bin/pylet
SLLM_BIN=.venv_new/bin/sllm
PORT=8343

echo "=== GPU Configuration ==="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES: Not Set (Using all available GPUs)"
fi

echo "Visible GPUs (as seen by PyTorch):"
$PYTHON -c "import torch; print(f'Count: {torch.cuda.device_count()}'); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" || echo "Failed to query PyTorch."
echo "========================="

# Paths (using distinct directories for isolation)
MODELS_DIR="./models_batch"
DB_PATH="./state_batch.db"

# Cleanup
echo "Cleaning up previous run..."
pkill -f "pylet start" || true
pkill -f "sllm start" || true
pkill -f "sllm-store" || true
sleep 2

mkdir -p $MODELS_DIR
rm -f ${DB_PATH}*
rm -f ~/.pylet/pylet.db*

echo "=== 1. Starting Pylet Head (Cluster Manager) ==="
$PYLET_BIN start > pylet_head_batch.log 2>&1 &
echo "Pylet Head PID: $!"
sleep 2

echo "=== 2. Starting Pylet Worker (2 GPUs) ==="
# Connects to localhost:8000
# We explicitly pass the env var again to be absolutely sure
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYLET_BIN start --head localhost:8000 --gpu-units 2 > pylet_worker_batch.log 2>&1 &
echo "Pylet Worker PID: $!"
sleep 2

echo "=== 3. Starting SLLM Head (Gateway + Router + Batch Scheduler) ==="
# Explicitly enable batch scheduler
export ENABLE_BATCH_SCHEDULER=true

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $SLLM_BIN start \
    --host 0.0.0.0 \
    --port $PORT \
    --pylet-endpoint http://localhost:8000 \
    --database-path "$DB_PATH" \
    --storage-path "$MODELS_DIR" > sllm_head_batch.log 2>&1 &
echo "SLLM Head PID: $!"

echo ""
echo "============================================================"
echo "Cluster Started (Batch Mode)"
echo "API Endpoint: http://localhost:$PORT"
echo "Batch Endpoint: http://localhost:$PORT/v1/batches"
echo "Logs: sllm_head_batch.log"
echo "============================================================"
echo "Press Ctrl+C to stop the cluster."

# === Example Usage (Run in another terminal) ===
echo ""
echo "=== To Submit a 10-Task Batch Job (Qwen 1.5B & 7B) ==="
echo "Run the following command in another terminal:"
echo ""
echo "curl -X POST http://localhost:8343/v1/batches \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"tasks\": ["
echo "      {\"custom_id\": \"task-1-llama\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"1+1=?\"}]}},"
echo "      {\"custom_id\": \"task-2-llama\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Name a color.\", \"max_tokens\": 10}]}},"
echo "      {\"custom_id\": \"task-3-llama\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of Italy?\"}]}},"
echo "      {\"custom_id\": \"task-4-llama\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a haiku about code.\"}]}},"
echo "      {\"custom_id\": \"task-5-llama\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"meta-llama/Meta-Llama-3-8B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Is Python a snake?\"}]}},"
echo "      {\"custom_id\": \"task-6-qwen\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Explain quantum entanglement simply.\"}]}},"
echo "      {\"custom_id\": \"task-7-qwen\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"List 3 benefits of exercise.\"}]}},"
echo "      {\"custom_id\": \"task-8-qwen\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Who wrote Hamlet?\"}]}},"
echo "      {\"custom_id\": \"task-9-qwen\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Translate Hello to Spanish.\"}]}},"
echo "      {\"custom_id\": \"task-10-qwen\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the speed of light?\"}]}}"
echo "    ]"
echo "  }'"
echo ""
echo "=== To Check Batch Status ==="
echo "curl http://localhost:8343/v1/batches/<BATCH_ID>"
echo ""

wait
