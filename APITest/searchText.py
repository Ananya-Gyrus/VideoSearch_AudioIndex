import requests
import json


port_num = 5801
BASE_URL = f"http://127.0.0.1:{port_num}"

search_url = f"{BASE_URL}/textsearch"
search_payload = {
    "query": "bradpitt",
    # "sortBy": "relevance",
    "startIndex": 1,
    "limit": 10,
    "dbName": "ei2",
    # "sourceIds" : ["cos", "wing2"],
    # "indexType": "text"
}
startIndex = search_payload["startIndex"]
search_resp = requests.post(search_url, json=search_payload)
print("Search Videos:", search_resp.json())

"""
{
    'query': 'Man holding a rifle',
    'results': [
        {
            'metadata': {
                'database': 'openmovies.pkl',
                'duration_sec': 2.708,
                'embedding_filename': 'tearsofsteel.mp4_sc0018_emb',
                'end_frame': 2133,
                'end_time_sec': 88.875,
                'faiss_id': 17,
                'result_number': 1,
                'scene_index': 18,
                'source_id': 'tos',
                'start_frame': 2068,
                'start_time_sec': 86.167,
                'total_scenes': 159,
                'video_filename': 'tearsofsteel.mp4',
                'video_path_relative': 'tearsofsteel.mp4'
            },
            'score': 0.21217244863510132
        }
    ],
    'search_time': 0.018146276473999023,
    'total_results': 1
}
"""

results = search_resp.json()
# print(results)
if "results" in results:
    results = results["results"]
else:
    results = []
# Group results by source_id
source_dict = {}
for res in results:
    source_id = res["metadata"]["source_id"]
    if source_id in source_dict:
        source_dict[source_id].append(res)
    else:
        source_dict[source_id] = [res]

# For each source, merge overlapping or nearby segments
for source_id, source_results in source_dict.items():
    merged = []
    source_results.sort(key=lambda x: x["metadata"]["start_time_sec"])
    
    current_group = source_results[0]
    for res in source_results[1:]:
        # Check if current result is close to the current group
        if abs(res["metadata"]["start_time_sec"] - current_group["metadata"]["end_time_sec"]) < 10:
            # Merge by updating bounds
            current_group["metadata"]["start_time_sec"] = min(current_group["metadata"]["start_time_sec"], 
                                                res["metadata"]["start_time_sec"])
            current_group["metadata"]["end_time_sec"] = max(current_group["metadata"]["end_time_sec"], 
                                            res["metadata"]["end_time_sec"])
            current_group["score"] = max(current_group.get("score", 0), 
                                    res.get("score", 0))
        else:
            # Start new group
            merged.append(current_group)
            current_group = res
    
    merged.append(current_group)
    source_dict[source_id] = merged

# Reconstruct results list
merged_results = []
for source_results in source_dict.values():
    merged_results.extend(source_results)

# Sort by score if available
if merged_results and "score" in merged_results[0]:
    merged_results.sort(key=lambda x: x.get("score", 0), reverse=True)

results = merged_results

for result in results:
    metadata = result['metadata']
    metadata["result_number"] = startIndex + results.index(result) + 1
    result = {
        "score": result['score'],
        "metadata": metadata
    }

# print(results)