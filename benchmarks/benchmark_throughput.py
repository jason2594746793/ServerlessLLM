import json
import time
import requests
import argparse
import sys
import os
from datetime import datetime

# Default configuration
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"  # Fallback
# Use the model mentioned by user if available, otherwise default
# User mentioned Qwen/Qwen3-0.6B in the prompt.

def generate_workload(model_name, num_tasks):
    """
    Generates a homogeneous workload for a single model.
    """
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Count from 1 to 20. Request ID: {i}"} 
                ],
                # Requesting enough tokens to make continuous batching meaningful
                "max_tokens": 100 
            }
        })
    return tasks

def monitor_batch(batch_id, model_name, base_url="http://localhost:8343", num_tasks=0):
    print(f"Monitoring Batch {batch_id}...")
    start_time = time.time()
    
    last_status = None
    
    while True:
        try:
            resp = requests.get(f"{base_url}/v1/batches/{batch_id}")
            resp.raise_for_status()
            data = resp.json()
            status = data['status']
            
            # Record status transition
            if status != last_status:
                print(f"[{time.time() - start_time:.2f}s] Status changed to: {status}")
                if status == "in_progress":
                    print(f"   Batch started processing...")
                last_status = status
            
            # Print dynamic progress
            if status == "in_progress":
                counts = data.get("request_counts", {})
                completed = counts.get("completed", 0)
                failed = counts.get("failed", 0)
                total = counts.get("total", num_tasks)
                
                msg = f"\r[{time.time() - start_time:6.2f}s] Progress: {completed}/{total} (Failed: {failed})"
                if completed > 0 and (time.time() - start_time) > 0:
                    rps = completed / (time.time() - start_time)
                    msg += f" | Current RPS: {rps:.2f}"
                
                sys.stdout.write(msg.ljust(80))
                sys.stdout.flush()

            if status in ["completed", "failed", "cancelled"]:
                print(f"\n\n[{time.time() - start_time:.2f}s] Final Status: {status}")
                if status == "failed":
                    print("!!! BATCH FAILED !!!")
                break
                
            time.sleep(1) # Poll interval
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nError polling status: {e}")
            time.sleep(2)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total Batch Duration: {duration:.2f} seconds")
    
    # Analysis
    print("\n" + "=" * 80)
    print("THROUGHPUT & LATENCY ANALYSIS")
    print("=" * 80)
    
    tasks = data.get("tasks", [])
    if not tasks:
        print("No task details available.")
        return

    # Filter completed tasks
    completed_tasks = [t for t in tasks if t["status"] == "completed"]
    valid_count = len(completed_tasks)
    
    if valid_count == 0:
        print("No tasks completed successfully.")
        return

    # Calculate metrics
    total_duration_sum = 0.0
    min_start = float('inf')
    max_end = 0.0
    
    for t in completed_tasks:
        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        
        if s_str and e_str:
            try:
                # Naive parsing, assuming UTC ISO format from server
                # Python 3.10+ handles 'Z' with fromisoformat, but to be safe:
                s_ts = datetime.fromisoformat(s_str.replace("Z", "+00:00")).timestamp()
                e_ts = datetime.fromisoformat(e_str.replace("Z", "+00:00")).timestamp()
                
                dur = e_ts - s_ts
                total_duration_sum += dur
                
                if s_ts < min_start: min_start = s_ts
                if e_ts > max_end: max_end = e_ts
            except ValueError:
                pass
    
    avg_latency = total_duration_sum / valid_count if valid_count else 0
    # Throughput = Tasks / (Time from First Start to Last End)
    # This excludes queue time before the first task picked up.
    processing_time = max_end - min_start if max_end > min_start else duration
    throughput = valid_count / processing_time if processing_time > 0 else 0
    
    # End-to-End Throughput (including queue/scheduling overhead)
    e2e_throughput = valid_count / duration

    print(f"Model: {model_name}")
    print("-" * 40)
    print(f"Total Requests:         {valid_count}")
    print(f"Successful Requests:    {valid_count}")
    print(f"Average Request Latency:{avg_latency:.4f} s")
    print(f"Processing Window:      {processing_time:.4f} s (First Start -> Last End)")
    print(f"Processing Throughput:  {throughput:.2f} req/s (GPU Concurrency)")
    print(f"End-to-End Throughput:  {e2e_throughput:.2f} req/s (System Performance)")
    print("=" * 80)

    # Save results
    result_filename = "throughput_benchmark_result.json"
    try:
        with open(result_filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n[INFO] Full benchmark result saved to: {result_filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save result file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a Throughput/Concurrency Benchmark")
    parser.add_argument("--url", default="http://localhost:8343", help="Base URL of SLLM")
    parser.add_argument("--tasks", type=int, default=50, help="Number of tasks to submit")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    
    args = parser.parse_args()

    print(f"Generating workload with {args.tasks} tasks for model '{args.model}'...")
    tasks = generate_workload(args.model, args.tasks)
    
    payload = {
        "tasks": tasks,
        "metadata": {
            "type": "throughput_benchmark",
            "model": args.model,
            "num_tasks": args.tasks
        }
    }
    
    try:
        print(f"Submitting batch to {args.url}...")
        resp = requests.post(f"{args.url}/v1/batches", json=payload)
        resp.raise_for_status()
        batch_data = resp.json()
        batch_id = batch_data['id']
        print(f"Batch Submitted! ID: {batch_id}")
        
        monitor_batch(batch_id, args.model, args.url, args.tasks)
        
    except Exception as e:
        print(f"Failed to submit batch: {e}")

if __name__ == "__main__":
    main()
