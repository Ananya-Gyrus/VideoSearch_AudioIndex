import requests
import json

port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

index_url = f"{BASE_URL}/index-videos"
index_payload = {
    "data": [
        {"filepath": "/home/gyrus3pc/Desktop/SMOL Models/videos/meridian.mp4", "sourceId": "mer", "fps": 30, "useAudio": True},
        {"filepath": "/home/gyrus3pc/Desktop/SMOL Models/videos/CosmosLaundromat.mp4", "sourceId": "cos", "fps": 30, "useAudio": True}
    ],
    "isVideo": True,
    "dbName": "ei_vid"
}

index_resp = requests.post(index_url, json=index_payload)
print("Index Videos:", index_resp.json())
