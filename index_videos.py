#!/usr/bin/env python3
"""
index_videos_csv.py

Standalone video indexing script derived from VideoSearchRestV2 repo.
- No Flask / API / license code
- Uses the same LanguageBind model utilities from the repo to compute video embeddings
- Stores metadata + embedding summary in a CSV file instead of a SQLite DB
- Prints each processed video's metadata and embedding summary to the terminal

Usage:
    python index_videos_csv.py --videos_dir ./videos --csv_path ./videos_index.csv --num_frames 8
"""

import os, sys, argparse, io, datetime
import numpy as np
import torch
import pandas as pd

# Ensure repo root is on path so we can import languagebind utilities
REPO_ROOT = os.path.join(os.path.dirname(__file__), 'VideoSearchRestV2-main', 'VideoSearchRestV2-main')
if not os.path.isdir(REPO_ROOT):
    REPO_ROOT = os.path.join(os.path.dirname(__file__), 'VideoSearchRestV2-main')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import project utilities
try:
    from languagebind_utils import get_languagebind_model, get_video_embedding
    from LanguageBind.vl_ret.rawvideo_util import RawVideoExtractor
except Exception as e:
    print("ERROR importing project utilities. Make sure this script sits next to the extracted repo folder.")
    print("Import error:", e)
    raise

def process_video_file(path, extractor, model, device, num_frames=8):
    meta = {'CosmosLaundromat.mp4': os.path.basename(path), '/home/ananya/Videos/CosmosLaundromat.mp4': os.path.abspath(path)}
    try:
        video_input = extractor.get_video_data(path)
        if isinstance(video_input, dict) and 'video' in video_input:
            video_frames = video_input['video']
        else:
            video_frames = video_input

        emb_tensor = get_video_embedding(video_frames, model, device, num_frames=num_frames)
        emb_np = emb_tensor.detach().cpu().numpy()
        meta['embedding_shape'] = str(emb_np.shape)
        meta['embedding_dtype'] = str(emb_np.dtype)

        import cv2
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            meta['fps'] = fps
            meta['frame_count'] = frames
            meta['duration_sec'] = frames / fps if fps > 0 else None
        cap.release()

        meta['indexed_at'] = datetime.datetime.utcnow().isoformat()

        return meta
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def index_videos_to_csv(videos_dir, csv_path, num_frames=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading LanguageBind model on:", device)
    model = get_languagebind_model(device)
    extractor = RawVideoExtractor(centercrop=False, size=224, framerate=-1)

    video_exts = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
    files = [os.path.join(root, f)
             for root, _, fs in os.walk(videos_dir)
             for f in fs if os.path.splitext(f)[1].lower() in video_exts]
    print(f"Found {len(files)} videos")

    rows = []
    for i, path in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing {path}")
        meta = process_video_file(path, extractor, model, device, num_frames)
        if not meta: continue
        rows.append(meta)
        print("→ Saved metadata for:", meta['filename'])
        print("   Embedding shape:", meta['embedding_shape'])

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"\n✅ CSV saved to: {csv_path}")

    print("\n✅ Indexing complete.")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--videos_dir', type=str, default='./videos')
    p.add_argument('--csv_path', type=str, default='./videos_index.csv')
    p.add_argument('--num_frames', type=int, default=8)
    args = p.parse_args()
    index_videos_to_csv(args.videos_dir, args.csv_path, args.num_frames)