# ServerlessLLM Batch API Guide

ServerlessLLM provides an OpenAI-compatible Batch API, designed for high-throughput, offline processing of large datasets. By submitting tasks in batches, the system optimizes GPU utilization, reduces model switching overhead, and enables processing millions of documents without holding concurrent HTTP connections.

## Starting the Server

Before submitting any batches, ensure the cluster is running in batch mode:

```bash
# Execute this in the ServerlessLLM directory
./scripts/start_cluster_batch.sh
```

---

## Two Ways to Submit

We support two submission methods to accommodate different workload sizes:
1. **Direct JSON Array (For small batches < 1k tasks)**
2. **File-based Submission / Dataset (For large batches, up to millions of tasks)** - *Recommended & Industrial Standard*

---

## Method 1: File-based Submission (Recommended)
This approach minimizes network payload and memory consumption on the API Gateway by using a `.jsonl` (JSON Lines) file to define the tasks.

### Step 1: Prepare your `.jsonl` file
Create a file where **each line** is a valid JSON object representing a single completion request. E.g., `requests.jsonl`:

```jsonl
{"custom_id": "req-01", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "What is AI?"}]}}
{"custom_id": "req-02", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "Write a short poem about space."}]}}
{"custom_id": "req-03", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Translate hello to French."}]}}
{"custom_id": "req-04", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "Qwen/Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "What is the boiling point of water?"}]}}
```

### Step 2: Upload the file
Use the `/v1/files` endpoint to securely upload your dataset to the server.

```bash
curl -X POST http://localhost:8343/v1/files \
  -H "Content-Type: multipart/form-data" \
  -F "purpose=batch" \
  -F "file=@requests.jsonl"
```
*Save the returned `id` (e.g., `file_abcdef123`).*

### Step 3: Create the Batch Job
Submit the batch by referencing the uploaded file ID. The server will parse the file entirely asynchronously.

```bash
curl -X POST http://localhost:8343/v1/batches \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file_abcdef123",
    "metadata": {
      "description": "My first big dataset"
    }
  }'
```
*Save the returned `id` (e.g., `batch_12345678`).*

---

## Method 2: Direct JSON Array Submission (Quick Testing)
For rapid prototyping or small task sets, you can embed the tasks directly in the POST body.

```bash
curl -X POST http://localhost:8343/v1/batches \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [
      {
        "custom_id": "task-llama-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
          "model": "meta-llama/Meta-Llama-3-8B-Instruct",
          "messages": [{"role": "user", "content": "1+1=?"}]
        }
      },
      {
        "custom_id": "task-llama-2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
          "model": "meta-llama/Meta-Llama-3-8B-Instruct",
          "messages": [{"role": "user", "content": "Name a database."}]
        }
      },
      {
        "custom_id": "task-qwen-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
          "model": "Qwen/Qwen2.5-7B-Instruct",
          "messages": [{"role": "user", "content": "Explain gravity."}]
        }
      },
      {
        "custom_id": "task-qwen-2",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
          "model": "Qwen/Qwen2.5-7B-Instruct",
          "messages": [{"role": "user", "content": "Who is Alan Turing?"}]
        }
      }
    ]
  }'
```

---

## Checking Batch Status

No matter which method you used, you retrieve the progress relying purely on the batch ID.

```bash
curl http://localhost:8343/v1/batches/batch_12345678
```

The response includes high-level progress tracking and individual task outputs.

**Example Response:**
```json
{
  "id": "batch_12345678",
  "status": "in_progress",
  "request_counts": {
    "total": 1000,
    "completed": 450,
    "failed": 0
  },
  "tasks": [
    {
      "custom_id": "req-01",
      "status": "completed",
      "output": { "choices": [...] }
    }
  ]
}
```

*Note: In the future, large job results will be available for download as an aggregated `.jsonl` output file to prevent massive payload sizes during retrieval.*
