import requests
import json

BASE_URL = "http://127.0.0.1:5800"

# 1. Licence Requirement
licence_url = f"{BASE_URL}/licence-requirement"
licence_resp = requests.post(licence_url)
print("Licence Requirement:", licence_resp.json())
# exit()
# 2. Index Videos
index_url = f"{BASE_URL}/index-videos"
scenefr = {
    "ToS": [0, 311, 420, 539, 697, 1063, 1107, 1192, 1239, 1292, 1324, 1385, 1457, 1644, 1775, 1875, 2002, 2068, 2133, 2315, 2387, 2454, 2555, 2898, 2974, 3335, 3455, 3536, 3596, 3766, 4232, 4495, 4645, 4731, 4861, 4925, 5057, 5105, 5129, 5145, 5196, 5231, 5270, 5435, 5499, 5727, 5825, 6179, 6287, 7046, 7180, 7213, 7237, 7272, 7318, 7362, 7482, 7551, 7594, 7666, 7776, 7830, 7875, 7980, 8024, 8231, 8286, 8412, 8459, 8502, 8562, 8649, 8788, 8845, 8921, 8952, 8967, 9016, 9070, 9121, 9150, 9169, 9217, 9292, 9347, 9384, 9432, 9461, 9517, 9673, 9707, 9794, 9915, 9982, 10057, 10148, 10309, 10364, 10452, 10517, 10587, 10623, 10686, 10774, 11180, 11337, 11366, 11452, 11592, 11622, 11668, 11699, 11749, 11810, 11843, 11895, 11911, 11937, 11990, 12036, 12083, 12115, 12154, 12186, 12238, 12336, 12526, 12601, 12755, 12846, 13002, 13038, 13133, 13365, 13461, 13640, 13892, 14209, 14235, 14365, 14440, 14532, 14607, 14760, 14895, 14996, 17125, 17625, 17717],
    "ABC": [0, 50, 150, 350, 420]
}
index_payload = {
    "filepaths": ["../../data/news/news_2.mp4", "../../data/sports/football_2.mp4"],
    "sourceIds": ["news_2", "football_2"],
    "isVideo": True,
    "videoFps": 24,
    "sceneFrames": scenefr,
    # "dbName": "tearsofsteel"
}
# index_resp = requests.post(index_url, json=index_payload)
# print("Index Videos:", index_resp.json())

# # 3. Get Status
status_url = f"{BASE_URL}/status"
status_resp = requests.get(status_url)
print("Get Status:", status_resp.json())

# # 4. Remove Video
remove_url = f"{BASE_URL}/remove-video"
remove_payload = {"sourceId": "news_2"}
# remove_resp = requests.post(remove_url, json=remove_payload)
# print("Remove Video:", remove_resp.json())

# 5. Search Videos
search_url = f"{BASE_URL}/textsearch"
search_payload = {
    "query": "missiles",
    "sortBy": "relevance",
    "startIndex": 1,
    "limit": 5,
    # "dbName": "tearsofsteel",
    # "sourceIds" : ["news_2", "football_2"]
}
search_resp = requests.post(search_url, json=search_payload)
print("Search Videos:", search_resp.json())

search_url = f"{BASE_URL}/imagesearch"
search_payload = {
    "image_path": "football.jpg",
    "sortBy": "relevance",
    "startIndex": 1,
    "limit": 5,
    # "dbName": "tearsofsteel",
    # "sourceIds" : ["news_2", "football_2"]
}
search_resp = requests.post(search_url, json=search_payload)
print("Search Videos:", search_resp.json())

search_url = f"{BASE_URL}/audiosearch"
search_payload = {
    "audio_path": "missile-firing.mp3",
    "sortBy": "relevance",
    "startIndex": 1,
    "limit": 5,
    # "dbName": "tearsofsteel",
    # "sourceIds" : ["news_6", "football_2"]
}
search_resp = requests.post(search_url, json=search_payload)
print("Search Videos:", search_resp.json())

# 6. Stream Embeddings
stream_url = f"{BASE_URL}/stream-embeddings"
stream_payload = {"k": 5}
# stream_resp = requests.post(stream_url, json=stream_payload)
# print("Stream Embeddings:", stream_resp.json()["meta"])
