import requests
import json
import pprint


port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

search_url = f"{BASE_URL}/imagesearch"
search_payload = {
    "image_path": "images/12.jpg",
    "sortBy": "relevance",
    "startIndex": 1,
    "limit": 10,
    "dbName": "openmovies",
    # "sourceIds" : ["tos", "wing2"]
}
search_resp = requests.post(search_url, json=search_payload)
print("Search Images:", search_resp.json())

"""
{
'query': 'rocket.jpg',
'results': [
    {
        'metadata': {
            'db_name': 'openmovies',
            'duration_sec': 4.542,
            'embedding_filename': 'tearsofsteel.mp4_sc0002_emb',
            'end_frame': 420,
            'end_time_sec': 17.5,
            'faiss_id': 1,
            'result_number': 1,
            'scene_index': 2,
            'source_id': 'tos',
            'start_frame': 311,
            'start_time_sec': 12.958,
            'total_scenes': 159,
            'video_filename': 'tearsofsteel.mp4',
            'video_path_relative': 'tearsofsteel.mp4'
        },
        'score': 0.0069452011957764626
    }
],
'search_time': 0.062477827072143555,
'total_results': 1
}

"""