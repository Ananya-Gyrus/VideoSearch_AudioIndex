import requests
import json
import pprint


port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

search_url = f"{BASE_URL}/audiosearch"
search_payload = {
    "audio_path": "traffic-327232.mp3",
    "sortBy": "relevance",
    "startIndex": 1,
    "limit": 10,
    "dbName": "openmovies",
    # "sourceIds": ["tos", "wing2"]
}
search_resp = requests.post(search_url, json=search_payload)
print("Search Videos:", search_resp.json())