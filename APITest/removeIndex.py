import requests
import json


port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

remove_url = f"{BASE_URL}/remove-video"
remove_payload = {"sourceId": "img"}
remove_resp = requests.post(remove_url, json=remove_payload)
print("Remove Video:", remove_resp.json())

"""
{'deletionstatus': {'message': 'Removed spr from index', 'removed_clips': 77, 'success': True}}
"""