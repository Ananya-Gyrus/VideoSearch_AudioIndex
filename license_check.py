import requests

port = 5801
resp = requests.post(f"http://127.0.0.1:{port}/licence-requirement")
print(resp.json())
