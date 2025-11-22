import requests
import json
import sys

def test_stream():
    url = "http://localhost:8001/rag/stream"
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "collection_name": "default"
    }
    headers = {"Content-Type": "application/json"}

    print(f"Connecting to {url}...")
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=30) as response:
            response.raise_for_status()
            print("Connected. Streaming events:")
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    print(f"Event: {data}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to backend. Is it running on port 8001?")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_stream()
