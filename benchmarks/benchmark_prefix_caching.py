import json
import time
import requests
import argparse
import sys
import os
from datetime import datetime

# Default configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# -----------------------------------------------------------------------------
# A long document to serve as the shared prefix (simulating RAG or Chat History)
# Approx 1000-1500 tokens.
# -----------------------------------------------------------------------------
LONG_SHARED_PREFIX = """
The History of Artificial Intelligence: From Theory to Reality

The intellectual roots of AI, and the concept of intelligent machines, may be found in Greek mythology. Intelligent artifacts appear in literature since then, with real (and fraudulent) mechanical devices actually demonstrated to behave with some degree of intelligence. After modern computers became available following World War II, it has become possible to create programs that perform difficult intellectual tasks.

1950 - 1980: Early AI
Applying the term artificial intelligence to computers was not a common idea until 1956 when the Dartmouth Conference on Artificial Intelligence took place. At this conference, the field of AI was formally founded. The attendees, including John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon, became the leaders of AI research for decades. They and their students wrote programs that were, to most people, simply astonishing: computers were winning at checkers, solving word problems in algebra, proving logical theorems and speaking English. By the middle of the 1960s, research in the U.S. was heavily funded by the Department of Defense and laboratories had been established around the world. AI's founders were optimistic about the future: Herbert Simon predicted, "machines will be capable, within twenty years, of doing any work a man can do". Marvin Minsky agreed, writing, "within a generation ... the problem of creating 'artificial intelligence' will substantially be solved".

1980 - 1987: Boom 
In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research in the field.

1987 - 1993: Bust
The collapse of the Lisp Machine market in 1987 was the first sign of trouble. "AI Winter" is a term first coined by researchers to describe the series of budget cuts and loss of enthusiasm for AI research that followed.

1993 - 2011: Integration and Quiet Success
In the late 90s and early 21st century, AI began to be used for logistics, data mining, medical diagnosis and other areas. The success was due to increasing computational power (see Moore's law), greater emphasis on solving specific subproblems, the creation of new ties between AI and other fields working on similar problems, and a new commitment by researchers to solid mathematical methods and rigorous scientific standards. Deep Blue became the first computer chess-playing system to beat a reigning world chess champion, Garry Kasparov, on 11 May 1997.

2011 - Present: Deep Learning and Big Data
Deep learning is a branch of machine learning that models high-level abstractions in data by using model architectures, which are composed of multiple non-linear transformations. It is part of a broader family of machine learning methods based on learning representations of data. An observation (e.g., an image) can be represented in many ways such as a vector of intensity values per pixel, or in a more abstract way as a set of edges, regions of particular shape, etc. Some representations make it easier to learn tasks (e.g., face recognition or facial expression recognition) from examples. One of the promises of deep learning is replacing handcrafted features with efficient algorithms for unsupervised or semi-supervised feature learning and hierarchical feature extraction.

Recent advances in deep learning, particularly with the advent of the Transformer architecture in 2017, have revolutionized Natural Language Processing (NLP). Models like BERT, GPT-3, and now Qwen, have demonstrated remarkable capabilities in understanding and generating human-like text. These models rely on massive datasets and enormous computational resources to train. The concept of "Serverless LLM" aims to make these powerful models accessible and efficient by optimizing resource allocation and reducing idle costs.

(This text is simply to fill up the context window to test prefix caching performance.)
""" * 3  # Repeat to ensure it's long enough (~1500 words -> ~2000 tokens)

def generate_prefix_workload(model_name, num_tasks):
    """
    Generates a workload where every request shares the SAME long prefix.
    If Prefix Caching is working, the KV cache for this prefix should be computed ONCE 
    and reused for subsequent requests, drastically reducing Time-To-First-Token (TTFT)
    and overall latency.
    """
    tasks = []
    for i in range(num_tasks):
        tasks.append({
            "custom_id": f"task-prefix-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name,
                "messages": [
                    # System Prompt contains the Long Shared Prefix
                    {"role": "system", "content": f"You are an AI assistant. Context: {LONG_SHARED_PREFIX}"},
                    # User Prompt is short and unique
                    {"role": "user", "content": f"Summarize the section about 'Deep Learning' in one sentence. Query ID: {i}"} 
                ],
                # We want just a short answer to focus benchmark on Prefill time
                "max_tokens": 50 
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
                    print(f"   Batch started processing (Expect faster speeds due to Prefix Caching)...")
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
    
    # Analysis
    print("\n" + "=" * 80)
    print("PREFIX CACHING BENCHMARK RESULTS")
    print("=" * 80)
    
    tasks = data.get("tasks", [])
    if not tasks:
        print("No task details available.")
        return

    completed_tasks = [t for t in tasks if t["status"] == "completed"]
    valid_count = len(completed_tasks)
    
    if valid_count == 0:
        print("No tasks completed successfully.")
        return

    # Calculate metrics
    first_task_latency = 0.0
    subsequent_latencies = []
    
    # Sort by started_at to distinguish the first request (Cache Miss) from others (Cache Hits)
    sorted_tasks = sorted(completed_tasks, key=lambda x: x.get("started_at") or "9999")
    
    for i, t in enumerate(sorted_tasks):
        s_str = t.get("started_at")
        e_str = t.get("completed_at")
        
        if s_str and e_str:
            try:
                s_ts = datetime.fromisoformat(s_str.replace("Z", "+00:00")).timestamp()
                e_ts = datetime.fromisoformat(e_str.replace("Z", "+00:00")).timestamp()
                dur = e_ts - s_ts
                
                if i == 0:
                    first_task_latency = dur
                else:
                    subsequent_latencies.append(dur)
            except ValueError:
                pass

    avg_subsequent = sum(subsequent_latencies) / len(subsequent_latencies) if subsequent_latencies else 0
    
    # Throughput (Requests per Second)
    throughput = valid_count / duration

    print(f"Model: {model_name}")
    print(f"Shared Prefix Length: ~2000 tokens")
    print("-" * 40)
    print(f"Total Requests:         {valid_count}")
    print(f"1. First Request Latency (Cache MISS):  {first_task_latency:.4f} s")
    print(f"2. Avg Subsequent Latency (Cache HIT):  {avg_subsequent:.4f} s")
    if avg_subsequent > 0:
        print(f"   => Speedup Factor: {first_task_latency / avg_subsequent:.2f}x")
    print("-" * 40)
    print(f"End-to-End Throughput:  {throughput:.2f} req/s")
    print("=" * 80)

    # Save results
    result_filename = "prefix_caching_result.json"
    try:
        data["metrics"] = {
            "first_latency": first_task_latency,
            "avg_subsequent_latency": avg_subsequent,
            "speedup": first_task_latency / avg_subsequent if avg_subsequent > 0 else 0
        }
        with open(result_filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"\n[INFO] Full benchmark result saved to: {result_filename}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save result file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a Prefix Caching Benchmark")
    parser.add_argument("--url", default="http://localhost:8343", help="Base URL of SLLM")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks to submit")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model to use")
    
    args = parser.parse_args()

    print(f"Generating PREFIX CACHING workload ({args.tasks} tasks) for '{args.model}'...")
    tasks = generate_prefix_workload(args.model, args.tasks)
    
    payload = {
        "tasks": tasks,
        "metadata": {
            "type": "prefix_caching_benchmark",
            "model": args.model,
            "num_tasks": args.tasks,
            "note": "Shared ~2000 token prefix"
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
