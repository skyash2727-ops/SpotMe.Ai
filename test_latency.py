import time
import requests
import numpy as np

# Generate a dummy 512D vector (simulating a selfie)
dummy_vector = np.random.rand(512)
dummy_vector = dummy_vector / np.linalg.norm(dummy_vector)

# Update this URL if your FastAPI runs on a different port or route!
URL = "http://127.0.0.1:8000/find-matches"  

latencies = []
print("Hitting the FastAPI Backend...")

for i in range(10):
    start_time = time.perf_counter()
    
    # We send the vector as a JSON list. 
    # (If your endpoint expects something else, let me know!)
    try:
        response = requests.post(URL, json={"image_embedding": dummy_vector.tolist()})
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        print(f"Request {i+1}: {latency_ms:.2f} ms")
    except Exception as e:
        print(f"Error hitting endpoint: {e}")
        break

if latencies:
    # We skip the first request because it's usually slower due to "cold start"
    avg_latency = sum(latencies[1:]) / len(latencies[1:])
    print("-" * 30)
    print(f"✅ Average End-to-End Latency: {avg_latency:.2f} ms")