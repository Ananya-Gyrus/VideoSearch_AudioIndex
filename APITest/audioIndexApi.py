import requests

port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

index_url = f"{BASE_URL}/index-audios"
index_payload = {
    "data": [
        {"filepath": "/home/gyrus3pc/Desktop/SMOL Models/videos/tearsofsteel.mp4", "sourceId": "eng", "useAudio": True},
        {"filepath": "/home/gyrus3pc/Desktop/SMOL Models/videos/Sprite.mp4", "sourceId": "eng", "useAudio": True},
        {"filepath": "/home/gyrus3pc/Desktop/SMOL Models/videos/CosmosLaundromat.mp4", "sourceId": "eng", "useAudio": True},
        
    ],
    "isAudio": True,

    "dbName": "ei_audio"
}

resp = requests.post(index_url, json=index_payload)
print("Status code:", resp.status_code)
print("Raw response:\n", resp.text)   
