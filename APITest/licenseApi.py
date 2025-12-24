import requests
import json


port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

# 1. Licence Requirement
licence_url = f"{BASE_URL}/licence-requirement"
licence_resp = requests.post(licence_url)
print("Licence Requirement:", licence_resp.json())


"""{
    "licensestatus": {
        "User Key": "hardware specific key",
        "status": "Key Successfully Generated",
        "success": True
    }
    """
