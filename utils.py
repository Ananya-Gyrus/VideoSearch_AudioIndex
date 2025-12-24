import os
import json
import time
import uuid
import threading
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Blueprint, Flask, jsonify, current_app, request, abort, send_file, render_template
from werkzeug.utils import secure_filename
import faiss
import pickle
import subprocess

from scenedetect import open_video, SceneManager
from scenedetect import detectors
from scenedetect.frame_timecode import FrameTimecode
from decord import cpu, VideoReader
from decord._ffi.base import DECORDError
from languagebind_utils import get_languagebind_model, get_video_embedding, get_text_embedding, get_text_embedding_batch

from datetime import datetime
from dotenv import load_dotenv
from generate_key import *
from db_utils import get_db_manager
import whisper


app = Flask(__name__)
SCENE_DETECT_THRESHOLD = 27.0
FRAMES_PER_CLIP_FOR_EMBEDDING = 8
BATCH_SIZE = 32  
OUTPUT_DIR = 'work_dir/database' 
USER_ID_FILE = 'work_dir/client_hardware_info.txt'
LICENCE_KEY_FILE = "work_dir/licence_key.txt"

load_dotenv()

def get_index_files(db_name="_default_db"):
    if db_name:
        base_name = db_name.replace(".index", "")
    return {
        'video': os.path.join(OUTPUT_DIR, f'{base_name}_video.index'),
        'text': os.path.join(OUTPUT_DIR, f'{base_name}_text.index'),
        'audio': os.path.join(OUTPUT_DIR, f'{base_name}_audio.index'),
    }

STARTDATE = datetime.fromisoformat(os.getenv("STARTDATE"))
HOURS_ATTRS = json.loads(os.getenv("HOURS_ATTRS"))
DATE_ATTRS = json.loads(os.getenv("DATE_ATTRS"))
PASSWORD = os.getenv("PASSWORD")
OFFLINE_LICENSE_LIMIT_HOURS = 0
RECENT_DATE = STARTDATE

def load_index(index_path, embedding_dim=768):
    """Helper function to load an index from a file"""
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
        except Exception as e:
            print(f"Failed to load existing FAISS index from {index_path}: {e}")
            index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
    else:
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
    return index

def save_index(index_path, index):
    """Helper function to save an index to a file"""
    try:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        return True
    except Exception as e:
        print(f"Failed to save FAISS index to {index_path}: {e}")
        return False

def check_licence_validation():
    try:
        global EXPIRYDATE, RECENT_DATE, LICENCE_KEY_FILE, OFFLINE_LICENSE_LIMIT_HOURS
        # print("Checking licence key...", LICENCE_KEY_FILE)
        if os.path.exists(LICENCE_KEY_FILE):
            dec_data = decrypt_file(LICENCE_KEY_FILE, PASSWORD).decode()
            dec_data = dec_data.split("\n")
            EXPIRYDATE = datetime.fromisoformat(dec_data[1].strip())
            OFFLINE_LICENSE_LIMIT_HOURS = float(dec_data[2])
            RECENT_DATE = datetime.fromisoformat(dec_data[3])
            curr_time = datetime.now()
            if (RECENT_DATE.year == curr_time.year and RECENT_DATE.month < curr_time.month) or RECENT_DATE.year < curr_time.year:
                # print("resetting to 1000 hours")
                update_usage_hours(1000)
                OFFLINE_LICENSE_LIMIT_HOURS = 1000
                
            if curr_time > RECENT_DATE:
                set_recent_date(curr_time)
                RECENT_DATE = curr_time

            uuid_ =  subprocess.check_output(
                        ["sudo", "dmidecode", "-s", "system-uuid"], stderr=subprocess.DEVNULL
                    ).decode().strip()
            print(uuid_,dec_data)
            if dec_data[0] != uuid_:
                print("User ID mismatch. Please generate a new key.")
                return 0
            else:
                return 1
        else:
            print("No licence key file found. Please generate a new key.")
            return 0
        
    except Exception as e:
        print(f"Error checking licence validation")
        return 0

def get_remaining_credit():
    global OFFLINE_LICENSE_LIMIT_HOURS
    return OFFLINE_LICENSE_LIMIT_HOURS
    
def initialize_config(app):
    global WORKING_DIR,BATCH_SIZE, OUTPUT_DIR, USER_ID_FILE, LICENCE_KEY_FILE
    BATCH_SIZE = app.config.get('BATCH_SIZE', 32)
    WORKING_DIR = app.config.get('WORKING_DIR', 'work_dir') 
    os.makedirs(WORKING_DIR, exist_ok=True)
    OUTPUT_DIR =os.path.join(WORKING_DIR, "database")
    USER_ID_FILE = os.path.join(WORKING_DIR, 'client_hardware_info.txt')
    LICENCE_KEY_FILE = os.path.join(WORKING_DIR, 'licence_key.txt')

loaded_db = None

model = None
indices_and_metas = []


def create_licence_requirement():
    try:
        with open(USER_ID_FILE, 'w+') as f:
            uuid =  subprocess.check_output(
                    ["sudo", "dmidecode", "-s", "system-uuid"], stderr=subprocess.DEVNULL
                ).strip()
            encrypted_uuid = encrypt_data(uuid, PASSWORD)
            f.writelines(encrypted_uuid.decode())
        return {'success': True,
                "User Key": encrypted_uuid.decode(),
                "status": "Key Successfully Generated"}, 200
    except:
        return {"success": False,
                "status": "Failed to create key"}, 500


def encrypt_data_update(data, expiry_date, hourly_credits, recent_date, password):
    key = get_key_from_password(password)
    fernet = Fernet(key)
    data += b"\n" + expiry_date.encode()
    data += b"\n" + str(hourly_credits).encode()
    data += b"\n" + str(recent_date).encode()
    encrypted = fernet.encrypt(data)
    return encrypted

def update_usage_hours(hrs):
    global RECENT_DATE, LICENCE_KEY_FILE, PASSWORD
    try:
        data_dec = decrypt_file(LICENCE_KEY_FILE, PASSWORD).decode()
        lines = data_dec.split("\n")
        encrypted_data = encrypt_data_update(lines[0].encode(),lines[1],hrs, RECENT_DATE, PASSWORD)
        with open(LICENCE_KEY_FILE,'w') as f:
            f.write(encrypted_data.decode())
        # print("Updated usage hours successfully")
    except Exception as e:
        print(f"Error updating usage hours: {e}")


def get_recent_date():
    global RECENT_DATE
    return RECENT_DATE

def set_recent_date(dt):
    global LICENCE_KEY_FILE, PASSWORD, OFFLINE_LICENSE_LIMIT_HOURS
    try:
        data_dec = decrypt_file(LICENCE_KEY_FILE, PASSWORD).decode()
        lines = data_dec.split("\n")
        encrypted_data = encrypt_data_update(lines[0].encode(),lines[1], OFFLINE_LICENSE_LIMIT_HOURS, dt, PASSWORD)
        with open(LICENCE_KEY_FILE,'w') as f:
            f.write(encrypted_data.decode())
    except Exception as e:
        print(f"Error setting it: {e}")

def is_online():
    return False


index_bp = Blueprint('index', __name__)

indexing_status = {
    'in_progress': False,
    'current_video': '',
    'processed_videos': 0,
    'processed_audios': 0,
    'partially_processed': 0,
    'video_queue': 0,
    'scenes_processed': 0,
    'total_scenes': 0,
    'overall_scenes_processed': 0,
    'overall_total_scenes': 0,
    'start_time': 0,
    'errors': []
}


def get_indexed_videos():
    """Get a list of all indexed videos and their metadata from the database"""
    proc_videos = 0
    proc_audios = 0
    partially_proccessed = 0

    # Get database manager and get grouped metadata
    db_manager = get_db_manager()
    db_groups = db_manager.get_indexed_files_by_db_and_type()
    for db_name, db_data in db_groups.items():
        proc_videos += len(db_data.get('video', []))
        proc_audios += len(db_data.get('text', []))
        partially_proccessed += len(db_data.get('partial', []))
    global indexing_status
    indexing_status['processed_videos'] = proc_videos
    indexing_status['processed_audios'] = proc_audios
    indexing_status['partially_processed'] = partially_proccessed
    return db_groups


# Helper functions for indexing
def sanitize_filename(filename):
    import re
    sanitized = re.sub(r'[^0-9a-zA-Z_.-]', '_', filename)
    return sanitized

def find_scenes_from_images(image_folder: str, image_pattern: str, fps: float, threshold: float = 27.0):
    image_sequence_path = os.path.join(image_folder, image_pattern)

    video_stream = None 
    try:
        print(f"Attempting to open image sequence: {image_sequence_path} with FPS: {fps}")
        video_stream = open_video(path=image_sequence_path, framerate=fps)
        
        if video_stream.frame_size[0] == 0 or video_stream.frame_size[1] == 0:
            # raise RuntimeError(f"Could not open or read the initial images in the sequence: {image_sequence_path}. "
            #                    "Please check the path, pattern, and ensure images exist and are readable. "
            #                    "Also, ensure image numbering starts as expected by the pattern (e.g., 0 or 1 for %d.jpg).")
            print((f"Could not open or read the initial images in the sequence: {image_sequence_path}. "
                               "Please check the path, pattern, and ensure images exist and are readable. "
                               "Also, ensure image numbering starts as expected by the pattern (e.g., 0 or 1 for %d.jpg)."))

        print(f"Successfully opened image sequence. Detected frame size: {video_stream.frame_size}, Duration: {video_stream.duration}")

    except Exception as e:
        print(f"Error opening image sequence: {e}")
        print("Please ensure that:")
        print(f"1. The image folder '{image_folder}' exists and contains images.")
        print(f"2. The image_pattern '{image_pattern}' is correct (e.g., '%d.jpg' for 0.jpg, 1.jpg...).")
        print("3. OpenCV (used by PySceneDetect) can read the image format and sequence.")
        print("4. Image numbering starts from 0 or 1 if using a simple '%d.jpg' pattern.")
        return None

    scene_manager = SceneManager()


    scene_manager.add_detector(detectors.ContentDetector(threshold=threshold))

    try:

        print(f"\nDetecting scenes in {image_sequence_path} at {fps} FPS using threshold {threshold}...")
        
        scene_manager.detect_scenes(video=video_stream, show_progress=False)
        scene_list_tc = scene_manager.get_scene_list()


        if not scene_list_tc:
            # Check if any frames were processed
            if video_stream.frame_number == 0 and video_stream.duration.get_frames() > 0 :
                 print("Warning: No frames seem to have been processed from the image sequence, though duration was reported.")
                 print("This might indicate an issue with OpenCV reading past the first few frames or an incorrect sequence pattern/numbering.")
            elif video_stream.frame_number == 0 and video_stream.duration.get_frames() == 0:
                 print("Warning: The image sequence appears to be empty or could not be read by OpenCV.")
            return []


        return scene_list_tc

    except Exception as e:
        print(f"Error during scene detection: {e}")
        return None


def detect_scenes(video_path, source_id, threshold, is_video=True, video_fps=30, manual_scene_frames=None):
    try:
        folder_name = source_id
        if not is_video and manual_scene_frames is not None:
            if folder_name in manual_scene_frames and manual_scene_frames[folder_name]:  
                frame_numbers = manual_scene_frames[folder_name]
                
                if not frame_numbers or not isinstance(frame_numbers, list):
                    print(f"Invalid frame numbers for {folder_name}, falling back to automatic detection")
                else:
                    scene_list = []
                    image_files = [f for f in os.listdir(video_path) if f.lower().endswith('.jpg')]
                    total_frames = len(image_files) - 1
                    
                    if not total_frames:
                        # raise ValueError(f"No JPG files found in {video_path}")
                        print(f"No JPG files found in {video_path}")
                    
                    frame_numbers.sort()
                    frame_numbers = [int(i) for i in frame_numbers]
                    
                    start_frame = 0
                    
                    for frame in frame_numbers:
                        if frame >= total_frames:
                            continue
                        if frame > start_frame and frame < total_frames:
                            start_timecode = FrameTimecode(start_frame, video_fps)
                            end_timecode = FrameTimecode(frame, video_fps)
                            scene_list.append((start_timecode, end_timecode))
                            start_frame = frame
                    
                    if start_frame < total_frames:
                        start_timecode = FrameTimecode(start_frame, video_fps)
                        end_timecode = FrameTimecode(total_frames, video_fps)
                        scene_list.append((start_timecode, end_timecode))
                    
                    print(f"Created {len(scene_list)} scenes from manual frame list for {folder_name}")
                    return scene_list, video_fps
        
        if is_video:
            video = open_video(video_path)
            frame_rate = video.frame_rate
            total_frames = int(frame_rate * video.duration.get_seconds())
            scene_manager = SceneManager()
            scene_manager.add_detector(detectors.ContentDetector(threshold=threshold))
            
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            

            if not scene_list or len(scene_list) == 0:
                scene_list = []
                num_scenes = int(total_frames / (frame_rate*6))
                frame_numbers = np.linspace(0, total_frames, num_scenes)
                frame_numbers.sort()
                start_frame = 0
                
                for frame in frame_numbers:

                    if frame > start_frame and frame < total_frames:
                        start_timecode = FrameTimecode(int(start_frame), video_fps)
                        end_timecode = FrameTimecode(int(frame), video_fps)
                        scene_list.append((start_timecode, end_timecode))
                        start_frame = frame

                if start_frame < total_frames:
                    start_timecode = FrameTimecode(int(start_frame), video_fps)
                    end_timecode = FrameTimecode(int(total_frames), video_fps)
                    scene_list.append((start_timecode, end_timecode))
                
                
            return scene_list, frame_rate
        else:
            scene_list = find_scenes_from_images(video_path, "%d.jpg", video_fps, threshold)
            if not scene_list or len(scene_list) == 0:
                scene_list = []
                total_frames = len(os.listdir(video_path))
                num_scenes = int(total_frames / (video_fps*6))
                frame_numbers = np.linspace(0, total_frames, num_scenes)
                frame_numbers.sort()
                start_frame = 0
                
                for frame in frame_numbers:
                    if frame > start_frame and frame < total_frames:
                        start_timecode = FrameTimecode(int(start_frame), video_fps)
                        end_timecode = FrameTimecode(int(frame), video_fps)
                        scene_list.append((start_timecode, end_timecode))
                        start_frame = frame
                
                if start_frame < total_frames:
                    start_timecode = FrameTimecode(int(start_frame), video_fps)
                    end_timecode = FrameTimecode(int(total_frames), video_fps)
                    scene_list.append((start_timecode, end_timecode))
            return scene_list, video_fps
            
    except Exception as e:
        print(f"Error detecting scenes in {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], 0

prevProcessedVideo = None
vidReader = None

def sample_frames(video_path, source_id, start_sec, end_sec, num_frames, fps, is_video=True):
    global prevProcessedVideo, vidReader
    if is_video and prevProcessedVideo != source_id:
        # vidReader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        vidReader = VideoReader(video_path, ctx=cpu(0))
        prevProcessedVideo = source_id 
    
    duration_sec = end_sec - start_sec
    if duration_sec <= 0:
        return []

    if is_video:
        if not vidReader:
            print(f"Error loading video")
            return []

        total_frames = len(vidReader)
        if total_frames < 8:
            return []
        video_fps = vidReader.get_avg_fps()
        if video_fps <= 0:
            video_fps = fps

        start_frame = int(start_sec * video_fps)
        end_frame = min(int(end_sec * video_fps), total_frames - 1)
        
        if end_frame <= start_frame:
            return []
        if end_frame >= total_frames:
            return []
        frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
        
        frames = vidReader.get_batch(frame_indices).asnumpy()
        return frames
    else:
        if not os.path.isdir(video_path):
            print(f"Error: {video_path} is not a directory")
            return []
            
        image_files = [f for f in os.listdir(video_path) if f.lower().endswith('.jpg')]
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        
        if not image_files:
            print(f"Error: No JPG files found in {video_path}")
            return []
            
        start_frame = int(start_sec * fps)
        end_frame = min(int(end_sec * fps), len(image_files) - 1)
        
        if end_frame <= start_frame:
            return []
            
        frame_indices = np.linspace(start_frame, end_frame, num=num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            idx = int(idx)
            if idx >= len(image_files):
                continue
                
            img_path = os.path.join(video_path, image_files[idx])
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            frames.append(frame)
            
        return frames

def preprocess_frames_for_batch(frames):
    if not len(frames):
        return None

    image_size = 224 
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(), 
            normalize, 
        ]
    )

    transformed_frames = []
    try:
        for frame_bgr in frames:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            transformed_tensor = transform_pipeline(pil_image) # Output: [C, H, W]
            transformed_frames.append(transformed_tensor)

        if not transformed_frames:
            print("Error: No frames were transformed successfully.")
            return None

        clip_tensor = torch.stack(transformed_frames, dim=0)

        return clip_tensor

    except Exception as e:
        print(f"Error during frame preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_embedding_batch_faiss(clip_tensor_batch, clip_metadata_batch, model, device, index, db_name):
    if not clip_tensor_batch:
        return

    try:
        clip_tensor_batch = torch.cat(clip_tensor_batch, dim=0)

        video_embedding = get_video_embedding(clip_tensor_batch, model, device, FRAMES_PER_CLIP_FOR_EMBEDDING)
        
        embeddings_np = video_embedding.cpu().numpy()
        current_idx = index.ntotal

        faiss.normalize_L2(embeddings_np)
        ids = np.arange(current_idx, current_idx + len(embeddings_np), dtype='int64')
        index.add_with_ids(embeddings_np, ids)
        db_manager = get_db_manager()
        for i, metadata in enumerate(clip_metadata_batch):
            metadata['faiss_id'] = current_idx + i
        # Save metadata batch to PostgreSQL
        db_manager.insert_metadata_batch(clip_metadata_batch, db_name)

    except Exception as e:
        print(f"Error processing batch: {e}")

def index_audio_and_text(video_path, source_id, is_video, db_name, video_fps=30):
    global OUTPUT_DIR, BATCH_SIZE, OFFLINE_LICENSE_LIMIT_HOURS, indexing_status
    # Create debug directory structure
    debug_dir = os.path.join(OUTPUT_DIR, "..", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    video_name = sanitize_filename(os.path.splitext(os.path.basename(video_path))[0])
    model, tokenizer = get_model()

    def extract_audio_chunks(video_path, chunk_duration=10, is_video=True):
        """
        Extracts audio from a video or image sequence and splits it into chunks of chunk_duration seconds.
        Returns a list of file paths to the audio chunks and total duration.
        """
        audio_chunks = []
        temp_dir = tempfile.mkdtemp()
        try:
            # Step 1: Extract audio to a temporary file
            audio_file = os.path.join(temp_dir, "audio.wav")
            if is_video:
                # Extract audio from video file
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_file
                ]
            else:
                # For image sequence, try to find a matching audio file in the same folder
                # (Assume audio.wav or audio.mp3 exists in the folder)
                possible_audio = None
                for ext in ["wav", "mp3", "aac", "m4a"]:
                    candidate = os.path.join(video_path, f"audio.{ext}")
                    if os.path.exists(candidate):
                        possible_audio = candidate
                        break
                if possible_audio:
                    shutil.copy(possible_audio, audio_file)
                else:
                    # No audio found for image sequence
                    return [], 0
                cmd = None

            if is_video and cmd:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            if not os.path.exists(audio_file):
                return [], 0

            # Step 2: Get audio duration
            probe_cmd = [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of",
                "default=noprint_wrappers=1:nokey=1", audio_file
            ]
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            duration_str = result.stdout.decode().strip()
            if not duration_str:
                return [], 0
            duration = float(duration_str)
            num_chunks = math.ceil(duration / chunk_duration)

            # Step 3: Split audio into chunks
            for i in range(num_chunks):
                start = i * chunk_duration
                out_chunk = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
                split_cmd = [
                    "ffmpeg", "-y", "-i", audio_file,
                    "-ss", str(start), "-t", str(chunk_duration),
                    "-acodec", "copy", out_chunk
                ]
                subprocess.run(split_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                if os.path.exists(out_chunk):
                    audio_chunks.append(out_chunk)

            return audio_chunks, duration
        except Exception as e:
            print(f"Error extracting audio chunks: {e}")
            return [], 0
        # Note: temp_dir will be cleaned up by the caller if needed
    # Example usage inside index_audio_and_text:
    # Get total duration of the media
    chunk_duration = 30
    # total_duration = get_media_duration(video_path, is_video)
    audio_chunks, total_duration = extract_audio_chunks(video_path, chunk_duration, is_video=is_video)
    print(f"Extracted {len(audio_chunks)} audio chunks from {video_path} (total duration: {total_duration:.2f}s)")
    AUDIO_BATCH_SIZE = BATCH_SIZE * 4  # You can adjust this as needed

    index_files = get_index_files(db_name)
    embedding_dim = 768

    # Get database manager for metadata storage
    db_manager = get_db_manager()

    text_model = whisper.load_model("turbo", download_root="cache_dir/whisper")
    text_index = load_index(index_files['text'])
    # print(f"Loaded text index with {text_index.ntotal} entries")
    if text_index is None:
        text_index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process text chunks
    duration = 0
    text_clip_tensor_batch = []
    text_clip_metadata_batch = []
    text_buffer = ""
    buffer_start_chunk_index = 0
    buffer_start_time = 0
    aud_trans_debug = []
    duration_in_hours = chunk_duration / 3600.0
    indexing_status["total_scenes"] += len(audio_chunks)
    indexing_status["overall_total_scenes"] += len(audio_chunks)

    max_chunk_indexed = db_manager.get_max_chunk_indexed(source_id, db_name)

    for i, audio_chunk_path in enumerate(audio_chunks):
        # Check if chunk already exists in database
        
        if i <= max_chunk_indexed:
            # print(f"Text chunk {i} for source_id {source_id} already indexed, skipping...")
            indexing_status["scenes_processed"] += 1
            continue
        
        # Get audio embedding tensor
        audio_bytes = open(audio_chunk_path, 'rb').read()
        if not audio_bytes:
            print(f"Skipping empty audio chunk: {audio_chunk_path}")
            continue
        text = text_model.transcribe(audio_chunk_path)
        new_text =  text['text'] if isinstance(text, dict) and 'text' in text else text
        
        # total_sentences = text_buffer + " " + new_text
        # sentences = nltk.sent_tokenize(total_sentences)
        sentences = [seg["text"] for seg in text["segments"]]
        time_stamps = [(seg["start"], seg["end"]) for seg in text["segments"]]
        no_speech_probs = [seg["no_speech_prob"] for seg in text["segments"]]
        
        new_sentences = copy.deepcopy(sentences)
        # new_sentences = nltk.sent_tokenize(new_text)
        # new_sentences = [s.strip() for s in new_sentences if s.strip()]
        if not text_buffer:
            buffer_start_chunk_index = i
            buffer_start_time = i * chunk_duration + time_stamps[0][0] if sentences else 0
        
        if sentences:
            sentences[0] = text_buffer + sentences[0]
        uses_buffer = []
        sentences_to_process = sentences
        if sentences and i < len(audio_chunks) - 1:
            # Not the last chunk, so hold back the last sentence fragment
            if sentences[-1].endswith('?') or sentences[-1].endswith('!') or (sentences[-1].endswith('.') and not new_text.endswith('...')):
                sentences_to_process = sentences
                text_buffer = ""
            else:
                sentences_to_process = sentences[:-1]
                text_buffer = sentences[-1]
        else:
            # Last chunk, process everything and clear the buffer
            text_buffer = ""
            buffer_start_chunk_index = i
            buffer_start_time = i * chunk_duration + time_stamps[0][0] if sentences else 0
        for each_sent in sentences_to_process:
            if each_sent not in new_sentences:
                uses_buffer.append(True)
            else:
                uses_buffer.append(False)

        if not sentences:
            # print(f"Skipping chunk {i} as it has no sentences.")
            # current_hours = OFFLINE_LICENSE_LIMIT_HOURS - duration_in_hours
            # OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, current_hours)
            # update_usage_hours(OFFLINE_LICENSE_LIMIT_HOURS)
            indexing_status["scenes_processed"] += 1
            continue
                
        if len(sentences) <= 1 and i < len(audio_chunks) - 1 : 
            if new_text.endswith('...'):
                # print(f"Skipping chunk {i} as it has only one sentence and ending with ... , indicating incomplete text.")
                duration += time_stamps[-1][1] - time_stamps[-1][0] 
                # current_hours = OFFLINE_LICENSE_LIMIT_HOURS - duration_in_hours
                # OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, current_hours)
                # update_usage_hours(OFFLINE_LICENSE_LIMIT_HOURS)
                indexing_status["scenes_processed"] += 1
                continue
            if not (new_text.endswith('.') or new_text.endswith('!') or new_text.endswith('?')):
                # print(f"Skipping chunk {i} as it has only one sentence and not ending with .?! , indicating incomplete text.")
                duration += time_stamps[-1][1] - time_stamps[-1][0] 
                # current_hours = OFFLINE_LICENSE_LIMIT_HOURS - duration_in_hours
                # OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, current_hours)
                # update_usage_hours(OFFLINE_LICENSE_LIMIT_HOURS)
                indexing_status["scenes_processed"] += 1
                continue    

        for sent_num, sent in enumerate(sentences_to_process):
            if not sent.strip():
                continue
            text_clip_tensor_batch.append(sent)
            if not uses_buffer[sent_num]:
                buffer_start_chunk_index = i
            
                
            start_chunk = buffer_start_chunk_index if uses_buffer[sent_num] else i
            start_time = max(0, (start_chunk * chunk_duration + time_stamps[sent_num][0] if not uses_buffer[sent_num] else buffer_start_time) - 3)
            end_time = start_chunk * chunk_duration + time_stamps[sent_num][1] if not uses_buffer[sent_num] else min( i * chunk_duration + time_stamps[0][1], total_duration)

            curr_meta = {
                "source_id": str(source_id),
                "chunk_index_start": start_chunk,
                "chunk_index_end": i,
                "sentence_index": sent_num,
                "embedding_type": "text",
                "video_filename": os.path.basename(video_path) if is_video else video_path,
                "video_path_relative": os.path.relpath(video_path, os.path.dirname(OUTPUT_DIR)),
                "embedding_filename": f"{db_name}_{source_id}_chunk_{i:04d}_{sent_num:04d}.txt",
                "total_scenes": len(audio_chunks),
                "start_frame": start_time * video_fps,
                "end_frame": end_time * video_fps,
                "start_time_sec": start_time,  # Each chunk is 10 seconds, so start time is chunk index * 10
                "end_time_sec": end_time,  # Use actual duration for last chunk
                "text": sent,
                "no_speech_prob": no_speech_probs[sent_num]
            }
            text_clip_metadata_batch.append(curr_meta)
            aud_trans_debug.append(curr_meta)
            
        if text_buffer:
            buffer_start_chunk_index = i
            buffer_start_time = i * chunk_duration + time_stamps[-1][0] if sentences else 0
            duration += time_stamps[-1][1] - time_stamps[-1][0]
            # print(f"Duration for this chunk:{text_buffer}:   ", duration)
        else:
            duration = 0 
            
        # Batchwise indexing
        if len(text_clip_tensor_batch) >= AUDIO_BATCH_SIZE:
            text_embeddings = get_text_embedding_batch(text_clip_tensor_batch, model, tokenizer, device, AUDIO_BATCH_SIZE)
            text_embeddings_np = text_embeddings.cpu().numpy() if text_embeddings is not None else None
            # print("Text embeddings shape:", text_embeddings_np.shape)
            faiss.normalize_L2(text_embeddings_np)
            current_idx = text_index.ntotal
            ids = np.arange(current_idx, current_idx + len(text_embeddings_np), dtype='int64')
            text_index.add_with_ids(text_embeddings_np, ids)
            # Store metadata in database
            for j, metadata in enumerate(text_clip_metadata_batch):
                metadata['faiss_id'] = current_idx + j

            db_manager.insert_metadata_batch(text_clip_metadata_batch, db_name)

            # Save FAISS index
            if text_index is not None:
                if not save_index(index_files['text'], text_index):
                    indexing_status['errors'].append("Failed to save text index")
            
            text_clip_tensor_batch = []
            text_clip_metadata_batch = []
        
        # current_hours = OFFLINE_LICENSE_LIMIT_HOURS - duration_in_hours
        # OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, current_hours)
        # update_usage_hours(OFFLINE_LICENSE_LIMIT_HOURS)

        indexing_status["scenes_processed"] += 1

    if text_buffer.strip():
        final_sent = text_buffer.strip()
        text_clip_tensor_batch.append(final_sent)
        start_time = max(0, buffer_start_time - 3, (len(audio_chunks) -1) * chunk_duration - 3)
        end_time = total_duration
        curr_meta = {
            "source_id": str(source_id),
            "chunk_index_start": buffer_start_chunk_index,
            "chunk_index_end": len(audio_chunks) - 1,
            "sentence_index": 0,
            "embedding_type": "text",
            "video_filename": os.path.basename(video_path) if is_video else video_path,
            "video_path_relative": os.path.relpath(video_path, os.path.dirname(OUTPUT_DIR)),
            "embedding_filename": f"{db_name}_{source_id}_chunk_{i:04d}.txt",
            "total_scenes": len(audio_chunks),
            "start_time_sec": start_time,  # Each chunk is 10 seconds, so start time is chunk index * 10
            "end_time_sec": end_time,  # Use actual duration for last chunk
            "start_frame": start_time * video_fps,
            "end_frame": end_time * video_fps,
            "text": final_sent,
            "no_speech_prob": no_speech_probs[-1] if no_speech_probs else 0
        }
        text_clip_metadata_batch.append(curr_meta)
        aud_trans_debug.append(curr_meta)

    # Process any remaining audio embeddings in the batch
    if text_clip_tensor_batch:
        text_embeddings = get_text_embedding_batch(text_clip_tensor_batch, model, tokenizer, device, AUDIO_BATCH_SIZE)
        text_embeddings_np = text_embeddings.cpu().numpy() if text_embeddings is not None else None
        # print("Text embeddings shape:", text_embeddings_np.shape)
        faiss.normalize_L2(text_embeddings_np)
        current_idx = text_index.ntotal
        ids = np.arange(current_idx, current_idx + len(text_embeddings_np), dtype='int64')
        text_index.add_with_ids(text_embeddings_np, ids)
        
        # Store metadata in database
        for j, metadata in enumerate(text_clip_metadata_batch):
            metadata['faiss_id'] = current_idx + j

        db_manager.insert_metadata_batch(text_clip_metadata_batch, db_name)

    # Save FAISS index if we have any updates
    if text_index is not None:
        if not save_index(index_files['text'], text_index):
            indexing_status['errors'].append("Failed to save text index")
    # Save the debug audio transcript file
    debug_transcript_path = os.path.join(debug_dir, f"{video_name}_transcripts.json")
    with open(debug_transcript_path, 'w') as f:
        json.dump({f"{video_name}": aud_trans_debug}, f, indent=4)
    del text_model
    return

def run_indexing_process(video_files, sourceIds, video_fps_list, use_audio_list, is_video=True, scene_frames=None, db_name= "_default_db"):
    global indexing_status, OFFLINE_LICENSE_LIMIT_HOURS, prevResults, EXPIRYDATE, RECENT_DATE, vidReader, prevProcessedVideo
    online = is_online()

    # Reset all status counters at the start
    indexing_status['in_progress'] = True
    indexing_status['start_time'] = time.time()
    indexing_status['video_queue'] = len(video_files)
    indexing_status['errors'] = []
    indexing_status['overall_scenes_processed'] = 0
    indexing_status['overall_total_scenes'] = 0
    succesfully_indexed = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index = None
    index_files = get_index_files(db_name)
    
    # Load indices for video, audio, and text
    video_index = load_index(index_files['video'])
    
    # Use video index as the main one
    index = video_index
    # print(f"Loaded video index with {index.ntotal} entries")
    # Get database manager
    db_manager = get_db_manager()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, tokenizer = get_model()
        
        if model is None:
            
            indexing_status['errors'].append("Failed to load model")
            indexing_status['in_progress'] = False
            return
    except Exception as e:

        indexing_status['errors'].append(f"Failed to load model: {str(e)}")
        indexing_status['in_progress'] = False
        return

    current_clip_tensor_batch = []
    current_clip_metadata_batch = []
    new_usage_hours = 0.0
    embedding_dim = 768
    #res = faiss.StandardGpuResources() if torch.cuda.is_available() else None
    if index is None:
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
    video_idx = 0

    for source_id, video_path, video_fps, use_audio in zip(sourceIds, video_files, video_fps_list, use_audio_list):
        print(f"Processing video {video_idx + 1}/{len(video_files)}: {video_path}")
        video_filename = os.path.basename(video_path)
        indexing_status['current_video'] = video_filename
        indexing_status['scenes_processed'] = 0
        scenes, video_frame_rate = detect_scenes(video_path, source_id, SCENE_DETECT_THRESHOLD, is_video, video_fps, scene_frames)
        print("Extracted", len(scenes), "scenes from", video_filename)
        indexing_status['total_scenes'] = len(scenes)
        indexing_status["overall_total_scenes"] += len(scenes)
        if use_audio:
            index_audio_and_text(video_path, source_id, is_video, db_name, video_fps)
        # Get existing metadata for this video from DB
        existing_metadata = db_manager.get_metadata_by_source_id_and_type(source_id, "video", db_name)
        # print(existing_metadata)
        existing_embeddings_count = len(existing_metadata)
        # print("existing_embeddings_count", existing_embeddings_count)
        if len(scenes) == existing_embeddings_count and len(scenes) > 0:
            # print("All scenes already indexed for this video, skipping:", video_filename)
            # If already indexed, count these scenes as processed
            indexing_status['scenes_processed'] += len(scenes)
            indexing_status["overall_scenes_processed"] += len(scenes)
            indexing_status['video_queue'] -= 1
            succesfully_indexed += 1
            video_idx += 1
            continue
        else:
            embedding_filenames = [item.get('embedding_filename') for item in existing_metadata]
        if not scenes or video_frame_rate <= 0:
            indexing_status['errors'].append(f"Failed scene detection for {video_filename}")
            # Still count this video as processed, but no scenes
            # indexing_status['processed_videos'] += 1
            indexing_status['video_queue'] -= 1
            video_idx += 1
            # indexing_status['total_scenes'] += len(scenes)
            continue
        
        # indexing_status['total_scenes'] += len(scenes)
        video_basename_sanitized = sanitize_filename(video_filename)
        succesfully_indexed_clips = 0
        for i, (start_timecode, end_timecode) in enumerate(scenes):
            try:
                scene_idx = i + 1
                embedding_filename = f"{db_name}_{source_id}_sc{scene_idx:04d}_emb"
                if embedding_filename in embedding_filenames:
                    indexing_status['scenes_processed'] += 1
                    indexing_status["overall_scenes_processed"] += 1
                    # print("skipping already indexed scene:", embedding_filename)
                    continue
                start_sec = start_timecode.get_seconds()
                end_sec = end_timecode.get_seconds()
                start_frame = start_timecode.get_frames()
                end_frame = end_timecode.get_frames()
                duration_sec = end_sec - start_sec
                duration_in_hours = duration_sec / 3600.0
                if duration_sec > 0:
                    new_usage_hours += duration_in_hours
                    
                else:
                    print(f"Invalid scene duration for {video_filename} (scene {scene_idx}), skipping...")
                    continue
                try:
                    frames = sample_frames(video_path, source_id, start_sec, end_sec, FRAMES_PER_CLIP_FOR_EMBEDDING, video_frame_rate, is_video)
                except Exception as e:
                    if isinstance(e, DECORDError):
                        vidReader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
                        frames = sample_frames(video_path, source_id, start_sec, end_sec, FRAMES_PER_CLIP_FOR_EMBEDDING, video_frame_rate, is_video)
                    else:
                        indexing_status['errors'].append(f"Error processing scene {i+1} in {video_filename}: {str(e)}")
                        frames = []
                if len(frames) != FRAMES_PER_CLIP_FOR_EMBEDDING:
                    print(f"Warning: Expected {FRAMES_PER_CLIP_FOR_EMBEDDING} frames, but got {len(frames)} for {video_filename} (scene {scene_idx}), skipping...")
                    indexing_status['errors'].append(f"Expected {FRAMES_PER_CLIP_FOR_EMBEDDING} frames, but got {len(frames)} for {video_filename} (scene {scene_idx}), skipping...")
                    continue
                if not len(frames):
                    print(f'len of frames is {len(frames)}')
                    continue
                clip_tensor = preprocess_frames_for_batch(frames)
                if clip_tensor is None:
                    print(f'clip_tensor is none for {video_filename} (scene {scene_idx}), skipping...')
                    continue

                clip_metadata = {
                    "source_id": str(source_id),
                    "video_filename": secure_filename(video_filename),
                    "video_path_relative": os.path.relpath(video_path, os.path.dirname(OUTPUT_DIR)),
                    "total_scenes": len(scenes),
                    "scene_index": scene_idx,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_time_sec": round(start_sec, 3),
                    "end_time_sec": round(end_sec, 3),
                    "duration_sec": round(duration_sec, 3),
                    "embedding_filename": embedding_filename,
                    "embedding_type": "video",
                }
                current_clip_tensor_batch.append(clip_tensor)
                current_clip_metadata_batch.append(clip_metadata)

                todays_date = datetime.now()
                if todays_date > EXPIRYDATE or todays_date < STARTDATE or todays_date < get_recent_date():
                    indexing_status['errors'].append("Licence Expired")
                    indexing_status['in_progress'] = False
                    print("Licence Expired, please contact support")
                    return

                if (RECENT_DATE.year == todays_date.year and RECENT_DATE.month < todays_date.month) or RECENT_DATE.year < todays_date.year:
                    # print("resetting to 1000 hours")
                    update_usage_hours(1000)
                    OFFLINE_LICENSE_LIMIT_HOURS = 1000
                
                RECENT_DATE = todays_date
                set_recent_date(RECENT_DATE)

                succesfully_indexed_clips += 1
                indexing_status['scenes_processed'] += 1
                indexing_status["overall_scenes_processed"] += 1

                # Only increment scenes_processed when actually processed (after batch)
                if len(current_clip_tensor_batch) >= BATCH_SIZE:
                    # Update license hours
                    current_hours = OFFLINE_LICENSE_LIMIT_HOURS - new_usage_hours
                    OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, current_hours)
                    update_usage_hours(OFFLINE_LICENSE_LIMIT_HOURS)
                    if OFFLINE_LICENSE_LIMIT_HOURS <= 0:
                        indexing_status['errors'].append("Usage limit exceeded please renew your licence")
                        indexing_status['in_progress'] = False
                        print("Usage limit exceeded, please contact support")
                        return
                    process_embedding_batch_faiss(
                        current_clip_tensor_batch,
                        current_clip_metadata_batch,
                        model,
                        device,
                        index,
                        db_name
                    )

                    try:
                        index_files = get_index_files(db_name)
                        
                        # Save video embeddings
                        if not save_index(index_files['video'], index):
                            indexing_status['errors'].append("Failed to save video index")
                        
                    except Exception as e:
                        indexing_status['errors'].append(f"Failed to save FAISS index and metadata: {str(e)}")
                    
                    # Increment scenes_processed by the number of scenes in this batch
                    current_clip_tensor_batch = []
                    current_clip_metadata_batch = []
                    new_usage_hours = 0.0

            except Exception as e:
                indexing_status['errors'].append(f"Error processing scene {i+1} in {video_filename}: {str(e)}")
                # If a scene fails, do not increment scenes_processed
        
        indexing_status['processed_videos'] += 1
        indexing_status['video_queue'] -= 1
        if succesfully_indexed_clips > 0:
            succesfully_indexed += 1
        video_idx += 1
        vidReader = None   
        prevProcessedVideo = None 

    # Process any remaining batch
    if current_clip_tensor_batch:
        # Update license hours
        current_hours = OFFLINE_LICENSE_LIMIT_HOURS - new_usage_hours
        OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, current_hours)
        update_usage_hours(OFFLINE_LICENSE_LIMIT_HOURS)
        if OFFLINE_LICENSE_LIMIT_HOURS <= 0:
            indexing_status['errors'].append("Usage limit exceeded please renew your licence")
            indexing_status['in_progress'] = False
            print("Usage limit exceeded, please contact support")
            return
        process_embedding_batch_faiss(
            current_clip_tensor_batch,
            current_clip_metadata_batch,
            model,
            device,
            index,
            db_name
        )

        try:
            index_files = get_index_files(db_name)
        
            # Convert indices to CPU if needed
            if torch.cuda.is_available():
                video_index_cpu = faiss.index_gpu_to_cpu(index)
            else:
                video_index_cpu = index

            # Save video embeddings
            if not save_index(index_files['video'], video_index_cpu):
                indexing_status['errors'].append("Failed to save video index")
                
        except Exception as e:
            indexing_status['errors'].append(f"Failed to save indices: {str(e)}") 
    
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # indexing_status['processed_videos'] += len(video_files)
    indexing_status['in_progress'] = False
    # total_hours_indexed = get_total_hours()
    # print("Total indexed hours:", total_hours_indexed)
    global prevResults
    prevResults = None
    print(f"Indexing completed, Given: {len(video_files)} videos, Successfully Indexed: {succesfully_indexed}, Time Elapsed: {time.time() - indexing_status['start_time']} seconds")


def index_videos(filepaths, sourceIds, video_fps_list, use_audio_list, is_video, scene_frames, db_name):

    global OUTPUT_DIR, OFFLINE_LICENSE_LIMIT_HOURS, indexing_status
    if not check_licence_validation():
        return {'error': 'License expired or invalid'}, 403
    if OFFLINE_LICENSE_LIMIT_HOURS <= 0:
        return {'error': 'Hour credits expired'}, 403
    # if not db_name.endswith(".index"):
    #     db_name = db_name + ".index"
    
    if indexing_status['in_progress']:
        return {'error': 'Indexing already in progress'}, 409

    if not filepaths:
        return {'error': 'No filenames provided'}, 400
    
    if not sourceIds:
        return {'error': 'No sourceIds provided'}, 400
    
    if len(filepaths) != len(sourceIds):
        return {'error': 'Filepaths and SourceIds are of different length'}, 400

    video_paths = []
    
    if len(filepaths) > 1:
        for filepath in filepaths:
            if filepath.endswith('/'):
                filepath = filepath[:-1]
            # wrk_dir = f"/{WORKING_DIR}/"
            src_path = os.path.join(WORKING_DIR, filepath)
            filename = os.path.basename(src_path)
            secure_name = secure_filename(filename)
            
            if os.path.exists(src_path):
                try:
                    video_paths.append(src_path)
                except Exception as e:
                    return {'error': f'Error copying file {secure_name}: {str(e)}'}, 500
            else:
                return {'error': f'File not found: {secure_name}'}, 404
        threading.Thread(target=run_indexing_process, 
                         args=(video_paths, sourceIds, video_fps_list, use_audio_list, is_video , scene_frames, db_name)).start()
        time.sleep(3)
        return {'success': True, 
                        'message': f'Started Indexing {len(video_paths)} videos as a group',
                }, 200
    else:
        if filepaths[0].endswith('/'):
                filepaths[0] = filepaths[0][:-1]
        filename = os.path.basename(filepaths[0])

        file_path = filepaths[0]
        file_path = os.path.join(WORKING_DIR, file_path)

        if not os.path.isfile(file_path) and (is_video or not os.path.isdir(file_path)):
            return {'error': f"""File not found or directory invalid {file_path}"""}, 404
        threading.Thread(target=run_indexing_process, 
                         args=([file_path], sourceIds, video_fps_list, use_audio_list, is_video, scene_frames, db_name)).start()
        time.sleep(3)
        
        return {'success': True, 'message': f'Started Indexing {filename}'}, 200
    
def reconstruct_index(sourceId, database_name, index_type='both'):
    db_manager = get_db_manager()
    index_files = get_index_files(database_name)
    if index_type in ['video', 'both']:
        video_index = load_index(index_files['video'])
        if video_index is not None:
            # Get all faiss_ids for the source from metadata
            faiss_ids = db_manager.get_faiss_ids_by_source_id_and_type(sourceId, 'video', database_name)
            if faiss_ids:
                fake_vector = np.zeros((1, video_index.d), dtype='float32')
                faiss.normalize_L2(fake_vector)
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(video_index.d))
                for i in range(video_index.ntotal):
                    if i in faiss_ids:
                        new_index.add_with_ids(fake_vector, np.array([i], dtype='int64'))
                    else:
                        vector = video_index.index.reconstruct(i).reshape(1, -1)
                        new_index.add_with_ids(vector, np.array([i], dtype='int64'))

                if not save_index(index_files['video'], new_index):
                    print(f"Error saving updated video index after replacing vectors for source {sourceId}")
    if index_type in ['text', 'both']:
        text_index = load_index(index_files['text'])
        if text_index is not None:
            faiss_ids = db_manager.get_faiss_ids_by_source_id_and_type(sourceId, 'text', database_name)
            if faiss_ids:
                
                fake_vector = np.zeros((1, text_index.d), dtype='float32')
                faiss.normalize_L2(fake_vector)
                new_index = faiss.IndexIDMap(faiss.IndexFlatIP(text_index.d))

                for i in range(text_index.ntotal):
                    
                    if i in faiss_ids:
                        new_index.add_with_ids(fake_vector, np.array([i], dtype='int64'))
                    else:
                        vector = text_index.index.reconstruct(i).reshape(1, -1)
                        new_index.add_with_ids(vector, np.array([i], dtype='int64'))

                if not save_index(index_files['text'], new_index):
                    print(f"Error saving updated text index after replacing vectors for source {sourceId}")

def remove_video(sourceId, db_name, index_type='both'):
    """
    Remove a video and its associated metadata from the database and note that the FAISS index needs rebuilding.
    Args:
        sourceId: The unique identifier of the video to remove
        db_name: The database name to remove from
        index_type: The type of indices to update ('video', 'text', or 'both')
    """
    # if db_name and not db_name.endswith(".index"):
    #     db_name = db_name + ".index"
    
    if not check_licence_validation():
        return {'error': 'License expired or invalid'}, 403
    if not sourceId:
        return {'error': 'No sourceId provided'}, 400
    if indexing_status['in_progress']:
        return {'error': 'Cannot remove videos while indexing is in progress'}, 409

    db_manager = get_db_manager()
    
    # Get database name from filename and handle type-specific deletions
    database_name = db_name.replace('.index', '') if db_name else None
    removed_count = 0
    
    try:
        # set the vectors of the removed indices to fake vector that won't be returned in search results
        if db_name is None:
            dbs = db_manager.get_all_databases()
            for db in dbs:
                reconstruct_index(sourceId, db, index_type)
        else:
            reconstruct_index(sourceId, db_name, index_type)

        if index_type == 'both':
            # Remove all metadata for the source
            removed_count = db_manager.remove_metadata_by_source_id_and_type(sourceId, database_name)
        else:
            # Only remove metadata of specific type
            removed_count = db_manager.remove_metadata_by_source_id_and_type(sourceId, database_name, index_type)

        if removed_count == 0:
            return {'message': f'No clips found for video with Source ID {sourceId} of type {index_type}', 'removed': 0}, 200

        # Reset search results cache
        global prevResults
        prevResults = None

        return {
            'success': True,
            'message': f'Removed {sourceId} from {index_type} index',
            'removed_clips': removed_count,
            # 'note': f'FAISS {index_type} index may need rebuilding for optimal performance'
        }, 200
        
    except Exception as e:
        return {'error': f'Error removing video: {str(e)}'}, 500 
    

def get_status():
    if not check_licence_validation():
        return {'error': 'License expired or invalid'}, 403
    global OFFLINE_LICENSE_LIMIT_HOURS
    elapsed = 0
    if indexing_status['start_time'] > 0 and indexing_status['in_progress']:
        elapsed = int(time.time() - indexing_status['start_time'])
    indexed_video_list = get_indexed_videos()
    return {
        'remaining_credits': OFFLINE_LICENSE_LIMIT_HOURS,
        'in_progress': indexing_status['in_progress'],
        'current_video': indexing_status['current_video'],
        'processed_videos': indexing_status['processed_videos'],
        'processed_audios': indexing_status['processed_audios'],
        # 'partially_processed': indexing_status['partially_processed'],
        'video_queue': indexing_status['video_queue'],
        'cv_scenes_processed': indexing_status['scenes_processed'],
        'cv_total_scenes': indexing_status['total_scenes'], 
        # 'overall_scenes_processed': indexing_status['overall_scenes_processed'],
        # 'overall_total_scenes': indexing_status['overall_total_scenes'],
        'elapsed_time': elapsed,
        'errors': indexing_status['errors'],
        "indexed_data" : indexed_video_list
    }


model = None
tokenizer = None

def get_model():
    global model, tokenizer
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, tokenizer, device = get_languagebind_model(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
            tokenizer = None
    return model, tokenizer


def get_faiss_data(dbName, index_type): 
    db_manager = get_db_manager()
    index_files = get_index_files(dbName) 
    file_path = index_files[index_type]
    try:
        if os.path.exists(file_path):
            # Load FAISS index
            index = faiss.read_index(file_path)
            # Get metadata from PostgreSQL
            metadata = db_manager.get_metadata_by_database_dict(dbName,index_type)
            return [{'index': index, 'metadata': metadata}, os.path.basename(file_path)]
        else:
            print(f"FAISS index file not found: {file_path}")
            return [{'index': None, 'metadata': None}, None]
    except Exception as e:
        print(f"Error loading FAISS data from {file_path}: {e}")
        return [{'index': None, 'metadata': None}, None]

import copy
import tempfile
import shutil
import math

prevQuery = None
prevResults = None
prevDbName = None
prevSourceIds = None
prevIndexType = None

def search_api(query, threshold, startIndex, limit, dbName, sourceIds=None, index_type='video'):
    """
    Search across embeddings with support for different index types (video/audio/text)
    Args:
        index_type: One of 'video', 'audio', or 'text' to determine which index to search
    """
    global prevQuery, prevResults, prevDbName, prevSourceIds, prevIndexType
    start_time = time.time()

    if startIndex < 1:
        startIndex = 1
    if limit <= 0:
        limit = 20
    startIndex -= 1  # Convert to 0-based index
    
    if not os.listdir(os.path.join(WORKING_DIR, "database")) or not os.path.exists(os.path.join(WORKING_DIR, "database")):
        prevResults = None

    if sourceIds and isinstance(sourceIds, list):
        sourceIds = [str(sid) for sid in sourceIds]  # Ensure all are strings
    elif sourceIds is None or sourceIds == []:
        sourceIds = None  # Keep current logic
    
    if prevQuery != query:
        prevQuery = query
    elif prevResults is not None and prevQuery == query and (startIndex+limit) <= len(prevResults) and prevDbName == dbName and prevSourceIds == sourceIds and prevIndexType == index_type:
        results = prevResults

        # Apply sourceIds filtering to cached results if needed
        if sourceIds is not None:
            results = [result for result in results if result['metadata'].get('source_id') in sourceIds]

        results.sort(key=lambda x: x['score'], reverse=True)
        
        startIndex = max(0, startIndex)
        endIndex = min(startIndex + limit, len(results))
        results = results[startIndex:endIndex]
        for result in results:
            metadata = result['metadata']
            metadata["result_number"] = startIndex + results.index(result) + 1
            result = {
                "score": result['score'],
                "metadata": metadata
            }
        
        search_time = time.time() - start_time
        return {
            'query': query,
            'results': results,
            'total_results': len(results),
            'search_time': search_time
        }, 200
    
    if not check_licence_validation():
        return {'error': 'License expired or invalid'}, 403
    
    if not query:
        prevResults = None
        return {'error': 'Query cannot be empty'}, 400
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model()
    if model is None or tokenizer is None:
        return {'error': 'Failed to load model for search'}, 500

    db_manager = get_db_manager()
    all_db_names = db_manager.get_all_databases()
    if not all_db_names:
        return {'error': 'No databases found. Please index videos first.'}, 404
    
    query_embedding = get_text_embedding(query, model, tokenizer, device)
    if query_embedding is None:
        return {'error': 'Failed to generate query embedding'}, 500

    query_embedding_np = query_embedding.cpu().numpy()
    faiss.normalize_L2(query_embedding_np)

    results = []
    existing_scenes = []
    if index_type == 'video':
        db_names = all_db_names if dbName == "*" else [dbName]
        for db_name in db_names:
            data, dbFileName = get_faiss_data(db_name, index_type)
            # print(f"Searching in database: {db_name}, index file: {dbFileName}")
            index = data.get('index')
            metadata = data.get('metadata', {})
            # print(f"Index has {index.ntotal} entries and metadata has {len(metadata)} items")
            if index is None or not metadata:
                continue
            try:
                k = min(startIndex+limit, len(metadata)) + 100
                distances, indices = index.search(query_embedding_np, k)
                for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
                    # print(idx, score)
                    if idx < 0 :
                        continue  
                    if score > threshold:  
                        metadata_item = None
                        try:
                            metadata_item = metadata[idx]
                        except KeyError:
                            continue
                        if sourceIds is not None:
                            if metadata_item.get('source_id') not in sourceIds:
                                continue  # Skip this result if source_id not in allowed list

                        if metadata_item.get('embedding_filename', "") not in existing_scenes:
                            results.append({
                                "score": float(score),  
                                "metadata": metadata_item
                            })
                            existing_scenes.append(metadata_item['embedding_filename'])
            except Exception as e:
                print(f'Error searching with FAISS: {str(e)}')
    elif index_type == 'text':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        db_names = all_db_names if dbName == "*" else [dbName]
        for db_name in db_names:
            data, dbFileName = get_faiss_data(db_name, index_type)
            # print(f"Searching in database: {db_name}, index file: {dbFileName}")
            index = data.get('index')
            metadata = data.get('metadata', {})
            # print(f"Index has {index.ntotal} entries and metadata has {len(metadata)} items")
            if index is None or not metadata:
                continue
                
            try:
                k = min(startIndex+limit, len(metadata)) + 100
                distances, indices = index.search(query_embedding_np, k)
                for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
                    if idx < 0 :
                        continue  
                    if score > threshold:  
                        metadata_item = None
                        try:
                            metadata_item = metadata[idx]
                        except KeyError:
                            continue
                        
                        if metadata_item is None:
                            metadata_item = metadata[idx].copy()

                        # Filter by sourceIds if provided
                        if sourceIds is not None:
                            if metadata_item.get('source_id') not in sourceIds:
                                continue  # Skip this result if source_id not in allowed list
                        if metadata_item.get('embedding_filename', "") not in existing_scenes:
                            results.append({
                                "score": float(score),  
                                "metadata": metadata_item
                            })
                            existing_scenes.append(metadata_item['embedding_filename'])
            except Exception as e:
                print(f'Error searching with FAISS: {str(e)}')

    prevResults = copy.deepcopy(results)
    prevDbName = dbName
    prevSourceIds = sourceIds
    prevIndexType = index_type

    results.sort(key=lambda x: x['score'], reverse=True)
    
    startIndex = max(0, startIndex)
    endIndex = min(startIndex + limit, len(results))
    results = results[startIndex:endIndex]
    for result in results:
        metadata = result['metadata']
        metadata["result_number"] = startIndex + results.index(result) + 1
        result = {
            "score": result['score'],
            "metadata": metadata
        }
    
    search_time = time.time() - start_time
    # print(results)
    return {
        'query': query,
        'results': results,
        'total_results': len(results),
        'search_time': search_time
    }, 200

from languagebind_utils import get_languagebind_image_model, get_image_embedding, get_audio_embedding, get_languagebind_audio_model
import io

image_model = None
image_tokenizer = None

audio_model = None
audio_tokenizer = None

prevImageQuery = None
prevImageResults = None
prevImageDbName = None
prevImageSourceIds = None

prevAudioQuery = None
prevAudioResults = None
prevAudioDbName = None
prevAudioSourceIds = None

def imagesearch_api(image_path, threshold, startIndex, limit, dbName, sourceIds=None):
    global prevImageQuery, prevImageResults, prevImageDbName, prevImageSourceIds
    image_path = os.path.join(WORKING_DIR, image_path)
    start_time = time.time()
    filename = image_path.split("/")[-1]
    if startIndex < 1:
        startIndex = 1
    if limit <= 0:
        limit = 20
    startIndex -= 1  # Convert to 0-based index
    # if dbName != "*" and (not dbName.endswith(".pkl")):
    #     dbName = dbName + ".pkl"

    # Handle sourceIds filtering
    if sourceIds and isinstance(sourceIds, list):
        sourceIds = [str(sid) for sid in sourceIds]  # Ensure all are strings
    elif sourceIds is None or sourceIds == []:
        sourceIds = None  # Keep current logic

    if prevImageQuery != image_path:
        prevImageQuery = image_path
    elif prevImageQuery == image_path and prevImageResults is not None and (startIndex+limit) <= len(prevImageResults) and prevImageDbName == dbName and prevImageSourceIds == sourceIds:
        # print("Using cached results for query:", image_path)
        results = prevImageResults

        # Apply sourceIds filtering to cached results if needed
        if sourceIds is not None:
            results = [result for result in results if result['metadata'].get('source_id') in sourceIds]

        results.sort(key=lambda x: x['score'], reverse=True)
        startIndex = max(0, startIndex)
        endIndex = min(startIndex + limit, len(results))
        # print("Start Index:", startIndex, "End Index:", endIndex, "Total Results:", len(results), "limit:", limit)
        results = results[startIndex:endIndex]
        for result in results:
            metadata = result['metadata']
            metadata["result_number"] = startIndex + results.index(result) + 1
            result = {
                "score": result['score'],
                "metadata": metadata
            }
        search_time = time.time() - start_time
        return {
            'query': filename,
            'results': results,
            'total_results': len(results),
            'search_time': search_time
        }, 200
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
    model, tokenizer, _ = get_languagebind_image_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("database name", dbName)
    if model is None:
        return {'error': 'Failed to load model for search'}, 500
    if image_bytes:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
            try:
                query_embedding = get_image_embedding(img, model, device)
                text_embedd = get_text_embedding("blank and black frames", model, tokenizer, device)
                if query_embedding is None:
                    return {'error': 'Failed to generate query embedding'}, 500
            except Exception as e:
                return {'error': f'Error generating query embedding: {str(e)}'}, 500
            query_embedding_np = query_embedding.cpu().numpy()
            faiss.normalize_L2(query_embedding_np)
            text_embedd_np = text_embedd.cpu().numpy()
            faiss.normalize_L2(text_embedd_np)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            db_manager = get_db_manager()
            all_db_names = db_manager.get_all_databases()
            if not all_db_names:
                return {'error': 'No databases found. Please index videos first.'}, 404
            results = []
            existing_scenes = []
            db_names = all_db_names if dbName == "*" else [dbName]
            for db_name in db_names:
                data, dbFileName = get_faiss_data(db_name, "video")
                # print(f"Searching in database: {db_name}, index file: {dbFileName}")
                index = data.get('index')
                metadata = data.get('metadata', [])
                if index is None or not metadata:
                    continue
                k = min(startIndex+limit, len(metadata)) + 100
                distances, indices = index.search(query_embedding_np, k)
                b_distances, b_indices = index.search(text_embedd_np, k//3)
                res_no = 0
                for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
                    # print(idx, score)
                    if idx < 0 or score <= threshold:
                        continue
                    metadata_item = None
                    try:
                        metadata_item = metadata[idx]
                    except KeyError:
                        continue
                    
                    if metadata_item is None:
                        metadata_item = metadata[idx].copy()
                    
                    # Filter by sourceIds if provided
                    if sourceIds is not None:
                        if metadata_item.get('source_id') not in sourceIds:
                            continue  # Skip this result if source_id not in allowed list
                    br = 0
                    for j, (b_idx, b_score) in enumerate(zip(b_indices[0], b_distances[0])):
                        b_metadata_item = None
                        try:
                            b_metadata_item = metadata[b_idx]
                        except KeyError:
                            continue

                        if b_metadata_item == metadata_item:
                            br = 1
                            break
                    # if br == 1:
                    #     continue
                    if float(metadata_item["duration_sec"]) < 2:
                        continue
                    res_no += 1
                    metadata_item['result_number'] = res_no
                    if metadata_item['embedding_filename'] not in existing_scenes:
                        results.append({
                            "score": float(score),
                            "metadata": metadata_item
                        })
                        existing_scenes.append(metadata_item['embedding_filename'])
            
            results.sort(key=lambda x: x['score'], reverse=True)
            prevImageResults = copy.deepcopy(results)
            prevImageDbName = dbName
            prevImageSourceIds = sourceIds
            startIndex = max(0, startIndex)
            endIndex = min(startIndex + limit, len(results))
            results = results[startIndex:endIndex]
            for result in results:
                metadata = result['metadata']
                metadata["result_number"] = startIndex + results.index(result) + 1
                result = {
                    "score": result['score'],
                    "metadata": metadata
                }
            search_time = time.time() - start_time
            return {
                'query': filename,
                'results': results,
                'total_results': len(results),
                'search_time': search_time
            }, 200
        except Exception as e:
            return {'error': f'Error in image search: {str(e)}'}, 500

def audiosearch_api(audio_path, threshold, startIndex, limit, dbName, sourceIds=None):
    global prevAudioQuery, prevAudioResults, prevAudioDbName, prevAudioSourceIds
    audio_path = os.path.join(WORKING_DIR, audio_path)
    start_time = time.time()
    if startIndex < 1:
        startIndex = 1
    if limit <= 0:
        limit = 20
    startIndex -= 1  # Convert to 0-based index
    # if dbName != "*" and (not dbName.endswith(".pkl")):
    #     dbName = dbName + ".pkl"

    # Handle sourceIds filtering
    if sourceIds and isinstance(sourceIds, list):
        sourceIds = [str(sid) for sid in sourceIds]  # Ensure all are strings
    elif sourceIds is None or sourceIds == []:
        sourceIds = None  # Keep current logic

    if prevAudioQuery != audio_path:
        prevAudioQuery = audio_path
    elif prevAudioQuery == audio_path and prevAudioResults is not None and (startIndex+limit) <= len(prevAudioResults) and prevAudioDbName == dbName and prevAudioSourceIds == sourceIds:
        # print("Using cached results for query:", audio_path)
        results = prevAudioResults

        # Apply sourceIds filtering to cached results if needed
        if sourceIds is not None:
            results = [result for result in results if result['metadata'].get('source_id') in sourceIds]
        
        
        results.sort(key=lambda x: x['score'], reverse=True)
        startIndex = max(0, startIndex)
        endIndex = min(startIndex + limit, len(results))
        # print("Start Index:", startIndex, "End Index:", endIndex, "Total Results:", len(results), "limit:", limit)
        results = results[startIndex:endIndex]
        for result in results:
            metadata = result['metadata']
            metadata["result_number"] = startIndex + results.index(result) + 1
            result = {
                "score": result['score'],
                "metadata": metadata
            }
        search_time = time.time() - start_time
        return {
            'query': audio_path.split("/")[-1],
            'results': results,
            'total_results': len(results),
            'search_time': search_time
        }, 200
    with open(audio_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    # print(audio_bytes)
    model, tokenizer, _ = get_languagebind_audio_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if dbName != "*" and (not dbName.endswith(".pkl")):
    #     dbName = dbName + ".pkl"
    if model is None:
        return {'error': 'Failed to load model for search'}, 500
    if audio_bytes:
        try:
            try:
                query_embedding = get_audio_embedding(audio_bytes, model, device)
                text_embedd = get_text_embedding("blank and black frames", model, tokenizer, device)
                if query_embedding is None:
                    return {'error': 'Failed to generate query embedding'}, 500
            except Exception as e:
                return {'error': f'Error generating query embedding: {str(e)}'}, 500
            query_embedding_np = query_embedding.cpu().numpy()
            faiss.normalize_L2(query_embedding_np)
            text_embedd_np = text_embedd.cpu().numpy()
            faiss.normalize_L2(text_embedd_np)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            db_manager = get_db_manager()
            all_db_names = db_manager.get_all_databases()
            if not all_db_names:
                return {'error': 'No databases found. Please index videos first.'}, 404
            results = []
            existing_scenes = []
            db_names = all_db_names if dbName == "*" else [dbName]
            for db_name in db_names:
                data, dbFileName = get_faiss_data(db_name, "video")
                # print(f"Searching in database: {db_name}, index file: {dbFileName}")
                index = data.get('index')
                metadata = data.get('metadata', [])
                if index is None or not metadata:
                    continue
                k = min(startIndex+limit, len(metadata)) + 100
                distances, indices = index.search(query_embedding_np, k)
                # distances /= 40
                b_distances, b_indices = index.search(text_embedd_np, k//2)
                res_no = 0
                for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
                    # print(idx, score)
                    if idx < 0 or score <= threshold:
                        continue
                    metadata_item = None
                    try:
                        metadata_item = metadata[idx]
                    except KeyError:
                        continue

                    if metadata_item is None:
                        metadata_item = metadata[idx].copy()

                     # Filter by sourceIds if provided
                    if sourceIds is not None:
                        if metadata_item.get('source_id') not in sourceIds:
                            continue  # Skip this result if source_id not in allowed list

                    br = 0
                    for j, (b_idx, b_score) in enumerate(zip(b_indices[0], b_distances[0])):
                        b_metadata_item = None
                        try:
                            b_metadata_item = metadata[b_idx]
                        except KeyError:
                            continue
                        if b_metadata_item == metadata_item:
                            br = 1
                            break
                    # if br == 1:
                    #     continue
                    if float(metadata_item["duration_sec"]) < 2:
                        continue
                    res_no += 1
                    metadata_item['result_number'] = res_no
                    if metadata_item['embedding_filename'] not in existing_scenes:
                        results.append({
                            "score": float(score),
                            "metadata": metadata_item
                        })
                        existing_scenes.append(metadata_item['embedding_filename'])
            
            results.sort(key=lambda x: x['score'], reverse=True)
            prevAudioResults = copy.deepcopy(results)
            prevAudioDbName = dbName
            prevAudioSourceIds = sourceIds
            startIndex = max(0, startIndex)
            endIndex = min(startIndex + limit, len(results))
            results = results[startIndex:endIndex]
            for result in results:
                metadata = result['metadata']
                metadata["result_number"] = startIndex + results.index(result) + 1
                result = {
                    "score": result['score'],
                    "metadata": metadata
                }
            search_time = time.time() - start_time
            return {
                'query': audio_path.split("/")[-1],
                'results': results,
                'total_results': len(results),
                'search_time': search_time
            }, 200
        except Exception as e:
            return {'error': f'Error in audio search: {str(e)}'}, 500
        
from languagebind_utils import get_audio_embeddings_batch

def index_audio_only(audio_path, source_id, db_name, batch_size=8):
    """
    Index raw audio content into FAISS and metadata DB.
    Uses get_audio_embeddings_batch() for efficient batched inference.
    """
    import tempfile, subprocess, math, shutil, numpy as np, torch, json, os
    from pathlib import Path

    global OUTPUT_DIR, BATCH_SIZE, OFFLINE_LICENSE_LIMIT_HOURS, indexing_status

    # --- Initialize directories ---
    debug_dir = os.path.join(OUTPUT_DIR, "..", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]

    # --- Load model + FAISS ---
    model, tokenizer, device = get_languagebind_audio_model()
    index_files = get_index_files(db_name)
    embedding_dim = 768
    audio_index = load_index(index_files["audio"])
    if audio_index is None:
        audio_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))

    db_manager = get_db_manager()
    max_chunk_indexed = db_manager.get_max_chunk_indexed(source_id, db_name)

    # --- Helper: extract & split audio into chunks ---
    def extract_audio_chunks(audio_path, chunk_duration=10, overlap_seconds=5):
        """
        Split long audio into overlapping chunks using ffmpeg.
        Example: chunk_duration=10, overlap_seconds=5 =>
        0-10, 5-15, 10-20, 15-25, ...
        """
        import tempfile, subprocess, math, os

        temp_dir = tempfile.mkdtemp()
        try:
            # Get audio duration using ffprobe
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ]
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            duration_str = result.stdout.decode().strip()
            if not duration_str:
                return [], 0
            total_duration = float(duration_str)

            # Overlap logic
            step = chunk_duration - overlap_seconds
            if step <= 0:
                raise ValueError("overlap_seconds must be smaller than chunk_duration")

            audio_chunks = []
            i = 0
            start_time = 0.0

            # Generate overlapping windows
            while start_time < total_duration:
                end_time = min(start_time + chunk_duration, total_duration)

                out_chunk = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
                cmd = [
                    "ffmpeg", "-y",
                    "-i", audio_path,
                    "-ss", str(start_time),
                    "-t", str(chunk_duration),
                    "-ac", "1", "-ar", "16000",
                    "-f", "wav", "-acodec", "pcm_s16le",
                    out_chunk
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

                if os.path.exists(out_chunk):
                    audio_chunks.append(out_chunk)

                i += 1
                start_time += step  # <-- THIS is the overlap

            return audio_chunks, total_duration

        except Exception as e:
            print(f"Error extracting audio chunks: {e}")
            return [], 0


    # --- Extract audio chunks ---
    chunk_duration = 10
    audio_chunks, total_duration = extract_audio_chunks(audio_path, chunk_duration)
    print("DEBUG chunk count =", len(audio_chunks))

    print(f" Extracted {len(audio_chunks)} chunks from {audio_path} ({total_duration:.2f}s)")

    AUDIO_BATCH_SIZE = batch_size
    indexing_status["total_scenes"] += len(audio_chunks)
    indexing_status["overall_total_scenes"] += len(audio_chunks)
    duration_in_hours = chunk_duration / 3600.0

    # --- Batch accumulation ---
    current_audio_batch = []
    current_metadata_batch = []

    for i, chunk_path in enumerate(audio_chunks):
        if i <= max_chunk_indexed:
            indexing_status["scenes_processed"] += 1
            continue

        try:
            with open(chunk_path, "rb") as f:
                audio_bytes = f.read()
            if not audio_bytes:
                print(f"Skipping empty chunk {chunk_path}")
                continue

            current_audio_batch.append(audio_bytes)
            meta = {
                "source_id": str(source_id),
                "chunk_index": i,
                "embedding_type": "audio",
                "audio_filename": os.path.basename(audio_path),
                "audio_path_relative": os.path.relpath(audio_path, os.path.dirname(OUTPUT_DIR)),
                "embedding_filename": f"{db_name}_{source_id}_chunk_{i:04d}.aud",
                "total_chunks": len(audio_chunks),
                "start_time_sec": i * chunk_duration,
                "end_time_sec": min((i + 1) * chunk_duration, total_duration)
            }
            current_metadata_batch.append(meta)

            # --- Process batch ---
            if len(current_audio_batch) >= AUDIO_BATCH_SIZE:
                embeddings = get_audio_embeddings_batch(current_audio_batch, model, device, batch_size=AUDIO_BATCH_SIZE)
                if embeddings is None:
                    print("Skipping batch — failed embeddings")
                    current_audio_batch, current_metadata_batch = [], []
                    continue

                emb_np = embeddings.cpu().numpy().astype("float32")
                faiss.normalize_L2(emb_np)
                current_idx = audio_index.ntotal
                ids = np.arange(current_idx, current_idx + len(emb_np), dtype='int64')
                audio_index.add_with_ids(emb_np, ids)

                # Assign FAISS IDs + store metadata
                for j, meta in enumerate(current_metadata_batch):
                    meta["faiss_id"] = current_idx + j
                db_manager.insert_metadata_batch(current_metadata_batch, db_name)
                save_index(index_files["audio"], audio_index)

                # Reset for next batch
                current_audio_batch, current_metadata_batch = [], []

            indexing_status["scenes_processed"] += 1
            OFFLINE_LICENSE_LIMIT_HOURS = max(0.0, OFFLINE_LICENSE_LIMIT_HOURS - duration_in_hours)

        except Exception as e:
            print(f"Error indexing chunk {chunk_path}: {e}")
            continue

    # --- Process any remaining audios ---
    if current_audio_batch:
        embeddings = get_audio_embeddings_batch(current_audio_batch, model, device, batch_size=AUDIO_BATCH_SIZE)
        if embeddings is not None:
            emb_np = embeddings.cpu().numpy().astype("float32")
            faiss.normalize_L2(emb_np)
            current_idx = audio_index.ntotal
            ids = np.arange(current_idx, current_idx + len(emb_np), dtype='int64')
            audio_index.add_with_ids(emb_np, ids)
            for j, meta in enumerate(current_metadata_batch):
                meta["faiss_id"] = current_idx + j
            db_manager.insert_metadata_batch(current_metadata_batch, db_name)
            save_index(index_files["audio"], audio_index)

    # --- Save debug info ---
    debug_log = {
        "audio_file": audio_path,
        "total_chunks": len(audio_chunks),
        "db_name": db_name,
        "indexed_ids": [m.get("faiss_id") for m in current_metadata_batch if "faiss_id" in m]
    }
    debug_json = os.path.join(debug_dir, f"{audio_name}_audio_debug.json")
    json.dump(debug_log, open(debug_json, "w"), indent=4)

    del model
    print(f"Finished audio indexing for {audio_path}")
    return
