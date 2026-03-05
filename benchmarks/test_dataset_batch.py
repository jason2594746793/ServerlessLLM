import json
import requests
import time
import sys

base_url = "http://localhost:8343"

def main():
    # 1. Create a simulated dataset file (.jsonl)
    filename = "test_dataset.jsonl"
    print(f"Creating local dataset file: {filename}")
    tasks = [
        {
            "custom_id": f"task-file-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "messages": [{"role": "user", "content": f"Calculate {i} + 10"}],
                "max_tokens": 10
            }
        }
        for i in range(5)
    ]
    
    with open(filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
            
    # 2. Upload the file to SLLM File API
    print("\nUploading file to /v1/files...")
    with open(filename, "rb") as f:
        # Standard form-data upload
        resp = requests.post(
            f"{base_url}/v1/files", 
            data={"purpose": "batch"}, 
            files={"file": (filename, f)}
        )
        
    if resp.status_code != 200:
        print(f"Failed to upload file: {resp.text}")
        return
        
    file_info = resp.json()
    file_id = file_info["id"]
    print(f" -> File uploaded successfully! File ID: {file_id}")
    
    # 3. Submit a Batch using ONLY the file id (no inline tasks)
    print(f"\nSubmitting Batch Job using input_file_id: {file_id}...")
    batch_payload = {
        "input_file_id": file_id,
        "metadata": {
            "source": "dataset_test",
            "description": "Testing file-based batch submission"
        }
    }
    
    resp = requests.post(f"{base_url}/v1/batches", json=batch_payload)
    if resp.status_code != 200:
        print(f"Failed to create batch: {resp.text}")
        return
        
    batch_info = resp.json()
    batch_id = batch_info["id"]
    print(f" -> Batch created successfully! Batch ID: {batch_id}")
    
    # 4. Monitor the batch
    print("\nMonitoring batch progress...")
    start_time = time.time()
    last_status = None
    
    while True:
        resp = requests.get(f"{base_url}/v1/batches/{batch_id}")
        if resp.status_code != 200:
            print(f"Error checking status: {resp.text}")
            time.sleep(2)
            continue
            
        data = resp.json()
        status = data['status']
        counts = data.get('request_counts', {})
        
        if status != last_status:
            print(f"\n[{time.time()-start_time:.1f}s] Status change: {status}")
            last_status = status
            
        if status == "in_progress":
            msg = f"\rProgress: {counts.get('completed', 0)}/{counts.get('total', 5)} tasks"
            sys.stdout.write(msg.ljust(50))
            sys.stdout.flush()
            
        if status in ["completed", "failed", "cancelled"]:
            print(f"\n\nBatch Finished with status: {status}")
            break
            
        time.sleep(1)

if __name__ == "__main__":
    main()
