import requests
import json


port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"



stream_url = f"{BASE_URL}/stream-embeddings"
stream_payload = {"dbName": "ei_rest", "k": 1}

stream_resp = requests.post(stream_url, json=stream_payload)
print("Stream Embeddings:", stream_resp.json())


# Format and pretty-print the JSON response
# response_json = stream_resp.json()
# formatted_json = json.dumps(response_json, indent=2)
# print("Formatted Stream Embeddings:\n", formatted_json)

