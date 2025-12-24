import os
import datetime
import sys
sys.path.append('LanguageBind')
from flask import Flask, jsonify, request
import argparse
from utils import *
import threading
app = Flask(__name__)
from utils import initialize_config
from db_utils import initialize_db_config
from setup_db import setup_database


def parse_directories():
    parser = argparse.ArgumentParser(description="Run Flask app, processing files from input_dir and saving results to output_dir.")

    parser.add_argument("-w", "--working_dir", dest="working_dir", default="work_dir",
                        help=f"Path to the working directory (default: work_dir)")
    
    parser.add_argument("-b", "--batch_size",type=int, dest="batch_size", default=8,
                        help=f"Batch size for indexation (default: 2)")

    parser.add_argument("-p", "--port", type=int, dest="port", default=5801,
                        help="Port to run the Flask app on (default: 5801)")
    
    parser.add_argument("-db", "--database_url", type=str, dest="database_url", required=True,
                        help="Database connection URL")
    args = parser.parse_args()

    return args.working_dir, args.batch_size, args.port, args.database_url


working_dir, batch_size, port_num, database_url = parse_directories()
app.config['WORKING_DIR'] = working_dir
app.config['BATCH_SIZE'] = batch_size
app.config['DATABASE_URL'] = database_url
os.makedirs(working_dir, exist_ok=True)
os.makedirs(os.path.join(working_dir, "database"), exist_ok=True)
initialize_config(app)
initialize_db_config(app)
setup_database()

@app.context_processor
def utility_processor():
    def now():
        return datetime.datetime.now()
    return dict(now=now)

@app.route('/licence-requirement', methods=['POST'])
def licence_requirement():
    working_dir = app.config.get('WORKING_DIR', 'work_dir')
    if check_licence_validation():
        Remaining_credit = get_remaining_credit()
        licensestatus = {"status": "License valid", "Remaining Hourly Credits": Remaining_credit}
        status_code = 200
    else:
        
        if not os.path.exists(os.path.join(working_dir, 'client_hardware_info.txt')):        
            licensestatus, status_code = create_licence_requirement()
        else:
            if os.path.exists(os.path.join(working_dir, 'licence_key.txt')):
                licensestatus = {"status": "User Key exists, Existing licence key is invalid."} 
            else:
                licensestatus = {"status": "User Key exists, Please generate licence key."}
    return jsonify({"licensestatus": licensestatus}), 200

@app.route('/index-audios', methods=['POST'])
def index_audios_rest():
    data = request.get_json()
    audio_data = data.get("data", [])

    # Parse fields similar to video indexing
    filepaths = [item["filepath"].rstrip('/') for item in audio_data]
    source_ids = [item.get("sourceId", os.path.basename(item["filepath"]).split(".")[0]) for item in audio_data]
    use_audio_list = [item.get("useAudio", True) for item in audio_data]

    db_name = data.get("dbName", "_default_audio_db")

    # Optional timing
    start_time = time.time()

    # Index the audios
    index_status = []
    for filepath, source_id, use_audio in zip(filepaths, source_ids, use_audio_list):
        result = index_audio_only(filepath, source_id, db_name, batch_size=8)
        index_status.append(result)

    elapsed = round(time.time() - start_time, 2)
    print(f"Indexed {len(audio_data)} audio files in {elapsed} seconds")

    # Return structured response
    return jsonify({"indexingstatus": index_status}), 200


@app.route('/index-videos', methods=['POST'])
def index_videos_rest():
    data = request.get_json()
    video_data = data.get("data", [])
    
    # Clean up filepath by removing trailing slash
    filepaths = [item["filepath"].rstrip('/') for item in video_data]
    source_ids = [item.get("sourceId", os.path.basename(item["filepath"]).split(".")[0]) for item in video_data]
    video_fps_list = [item.get("fps", 30) for item in video_data]
    use_audio_list = [item.get("useAudio", False) for item in video_data]
    scene_frames = {item["sourceId"]: item["sceneFrames"] for item in video_data if "sceneFrames" in item}
    
    is_video = data.get("isVideo", True)
    db_name = data.get("dbName", "_default_db")
    index_status, statuscode = index_videos(filepaths, source_ids, video_fps_list, use_audio_list, is_video, scene_frames, db_name)
    return jsonify({"indexingstatus": index_status}), statuscode

@app.route('/remove-video', methods=['POST'])
def remove_video_rest():
    data = request.get_json()
    db_name = data.get("dbName", None)
    source_id = data.get("sourceId")
    index_type = data.get("indexType", "both")
    deletion_status, status_code = remove_video(source_id, db_name, index_type)
    return jsonify({"deletionstatus": deletion_status}), status_code

@app.route('/status', methods=['GET'])
def get_status_rest():
    status_data = get_status()
    return jsonify(status_data), 200

@app.route('/textsearch', methods=['POST'])
def textsearch():
    data = request.get_json()
    query = data.get("query")
    start_index = data.get("startIndex")
    sort_by = data.get("sortBy", "relevance")
    limit = data.get("limit", 20)
    db_name = data.get("dbName", "*")
    source_ids = data.get("sourceIds", None)
    index_type = data.get("indexType", "video")
    search_res, status_code = search_api(query, 0, start_index, limit, db_name, source_ids, index_type)
    return jsonify(search_res), status_code

@app.route('/imagesearch', methods=['POST'])
def imagesearch():
    data = request.get_json()
    image_path = data.get("image_path")
    start_index = data.get("startIndex")
    sort_by = data.get("sortBy", "relevance")
    limit = data.get("limit", 20)
    db_name = data.get("dbName", "*")
    source_ids = data.get("sourceIds", None)
    search_res, status_code = imagesearch_api(image_path, 0, start_index, limit, db_name, source_ids)
    return jsonify(search_res), status_code

@app.route('/audiosearch', methods=['POST'])
def audiosearch():
    data = request.get_json()
    audio_path = data.get("audio_path")
    start_index = data.get("startIndex")
    sort_by = data.get("sortBy", "relevance")
    limit = data.get("limit", 20)
    db_name = data.get("dbName", "*")
    source_ids = data.get("sourceIds", None)
    search_res, status_code = audiosearch_api(audio_path, 0, start_index, limit, db_name, source_ids)
    return jsonify(search_res), status_code

@app.route('/stream-embeddings', methods=['POST'])
def stream_embeddings_rest():
    # data = request.get_json()
    # print("Received data for streaming embeddings:", data)
    # if not data:
    #     return jsonify({"error": "No data provided"}), 400
    # db_name = data.get("dbName", "_default_db.pkl")
    # k = data.get("k", 10)
    # vec_results, status = stream_vec_embeddings(db_name,"text", k)
    # if status == 200:
    #     return jsonify(vec_results), 200
    vec_results = {"message": "Streaming embeddings feature is currently disabled."}
    return jsonify(vec_results), 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_num)

