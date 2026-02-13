import json
import time
import requests
import argparse
import sys
import os
from datetime import datetime

def generate_thrashing_workload(num_pairs=5):
    """
    Generates a workload that alternates between two models.
    """
    # Using two models that are relatively large but will run on the current A5000 setup.
    # We use Qwen-14B and Qwen-7B to simulate "large" models relative to available memory,
    # or just two distinct models to force switching.
    model_a = "Qwen/Qwen3-0.6B" 
    model_b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 

    tasks = []
    for i in range(num_pairs):
        # Task A
        tasks.append({
            "custom_id": f"task-{i}-A",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_a,
                "messages": [{"role": "user", "content": f"what is machine learning{i}"}],
                "max_tokens": 100
            }
        })
        # Task B
        tasks.append({
            "custom_id": f"task-{i}-B",
            "method": "POST",
            "url": "/v1/chat/completions", 
            "body": {
                "model": model_b,
                "messages": [{"role": "user", "content": f"say something in chinese{i}"}],
                "max_tokens": 100
            }
        })
    return tasks

def monitor_batch(batch_id, base_url="http://localhost:8343"):
    print(f"Monitoring Batch {batch_id}...")
    start_time = time.time()
    
    # Store history of status changes
    status_history = []
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
                    # Try to capture progress metrics if available
                    completed = data.get("request_counts", {}).get("completed", 0)
                    print(f"   Progress: {completed}/{data.get('request_counts', {}).get('total', '?')}")
                status_history.append((time.time(), status))
                last_status = status
            
                
            # If in progress, print dynamic status line
            if status == "in_progress":
                counts = data.get("request_counts", {})
                completed = counts.get("completed", 0)
                failed = counts.get("failed", 0)
                total = counts.get("total", "?")
                
                # Check individual task progress if we have the task list
                current_tasks = data.get("tasks", [])
                completed_ids = [t.get("custom_id", t["id"]) for t in current_tasks if t["status"] == "completed"]
                
                msg = f"\r[{time.time() - start_time:6.2f}s] Progress: {completed}/{total} (Failed: {failed}) | " 
                if completed_ids:
                    msg += f"Latest Done: {completed_ids[-1]}"
                
                sys.stdout.write(msg.ljust(100))
                sys.stdout.flush()

            if status in ["completed", "failed", "cancelled"]:
                print(f"\n\n[{time.time() - start_time:.2f}s] Final Status: {status}")
                if status == "failed":
                    print("!!! BATCH FAILED !!!")
                break
                
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\nError polling status: {e}")
            time.sleep(2)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total Batch Duration: {duration:.2f} seconds")
    
    # Analysis using Server-Side Metrics
    print("\n" + "=" * 100)
    print("ANALYSIS OF MODEL SWITCHING OVERHEAD ('THRASHING')")
    print("=" * 100)
    
    tasks = data.get("tasks", [])
    # Sort by started_at if available to ensure chronological order analysis
    tasks.sort(key=lambda x: x.get("started_at") or "9999")

    if not tasks:
        print("No task details available.")
        return

    # Header
    print(f"{'Task ID':<20} | {'Model':<30} | {'Total Duration':<15} | {'Est. Load Time':<15} | {'Inference (Est)':<15} | {'Gap (Teardown)':<15}")
    print("-" * 115)
    
    total_execution_sums = 0.0
    total_load_est = 0.0
    valid_tasks = 0
    prev_end_ts = None
    
    # Heuristic: For trivial tasks (1+1, 10 tokens), inference is usually < 0.2s on A5000.
    # We assume anything above 0.5s is overhead (loading/queuing).
    INFERENCE_BASELINE = 0.5 

    for t in tasks:
        t_id = t.get("custom_id", t["id"])
        
        # Identify model from ID if possible (our script names them task-A/B)
        model_label = "Model A" if "-A" in t_id else "Model B" if "-B" in t_id else "?"
        # Try to extract actual model name if we tracked it (we didn't store it in DB output directly, but we know it from inputs)
        # Just use label for now.
        
        start_str = t.get("started_at")
        end_str = t.get("completed_at")
        
        duration_str = "-"
        load_est_str = "-"
        gap_str = "-"
        
        if start_str and end_str:
            try:
                s_ts = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                e_ts = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                
                duration_val = (e_ts - s_ts).total_seconds()
                duration_str = f"{duration_val:.2f}s"
                
                # Estimate Loading Time
                # If duration is huge, it's mostly loading.
                if duration_val > INFERENCE_BASELINE:
                    load_est = duration_val - INFERENCE_BASELINE
                    load_est_str = f"{load_est:.2f}s"
                    total_load_est += load_est
                else:
                    load_est_str = "~0s"
                
                # Gap from previous task
                if prev_end_ts:
                    gap_val = (s_ts - prev_end_ts).total_seconds()
                    gap_str = f"{gap_val:.2f}s"
                else:
                    gap_str = "0.00s (Start)"
                
                prev_end_ts = e_ts
                total_execution_sums += duration_val
                valid_tasks += 1
                
            except Exception as e:
                duration_str = "Err"

        print(f"{t_id:<20} | {model_label:<30} | {duration_str:<15} | {load_est_str:<15} | {INFERENCE_BASELINE:<15} | {gap_str:<15}")

    print("-" * 115)
    
    if valid_tasks > 0:
        avg_duration = total_execution_sums / valid_tasks
        avg_load = total_load_est / valid_tasks
        thrashing_percentage = (avg_load / avg_duration) * 100 if avg_duration > 0 else 0
        
        print("\nSUMMARY statistics:")
        print(f"1. Average Task Duration:       {avg_duration:.2f}s")
        print(f"2. Average Model Loading Time:  {avg_load:.2f}s")
        print(f"3. Thrashing Overhead:          {thrashing_percentage:.1f}%")
        print(f"4. Throughput:                  {valid_tasks / duration:.2f} tasks/second")
        print(f"5. Total Batch Duration:        {duration:.2f}s")
        print("-" * 115)
    print("=" * 115)

    # Save full result to file
    result_filename = "thrashing_benchmark_result.json"
    try:
        # Fetch final state one last time to be sure
        resp = requests.get(f"{base_url}/v1/batches/{batch_id}")
        final_data = resp.json()
        
        with open(result_filename, "w") as f:
            json.dump(final_data, f, indent=4)
        print(f"\n[INFO] Full benchmark result saved to: {result_filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save result file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a 'Thrashing' Benchmark")
    parser.add_argument("--url", default="http://localhost:8343", help="Base URL of SLLM")
    parser.add_argument("--pairs", type=int, default=10, help="Number of A/B pairs (total tasks = pairs*2)")
    args = parser.parse_args()

    print(f"Generating workload with {args.pairs * 2} alternating tasks...")
    tasks = generate_thrashing_workload(args.pairs)
    
    payload = {
        "tasks": tasks,
        "metadata": {"type": "thrashing_benchmark"}
    }
    
    try:
        print(f"Submitting batch to {args.url}...")
        resp = requests.post(f"{args.url}/v1/batches", json=payload)
        resp.raise_for_status()
        batch_data = resp.json()
        batch_id = batch_data['id']
        print(f"Batch Submitted! ID: {batch_id}")
        
        monitor_batch(batch_id, args.url)
        
    except Exception as e:
        print(f"Failed to submit batch: {e}")

if __name__ == "__main__":
    main()
