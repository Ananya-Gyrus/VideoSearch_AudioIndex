# save as qwen_frames_infer_debug.py
import sys
import os
print("✅ Imports starting...", flush=True)

import cv2
import tempfile
import time
import torch
print("✅ Core libs loaded...", flush=True)

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from huggingface_hub import snapshot_download
    print("✅ Transformers + Qwen utils imported successfully!", flush=True)
except Exception as e:
    print("❌ Import failed:", e, flush=True)
    sys.exit(1)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEO_PATH = "/home/gyrus3pc/Desktop/SMOL Models/videos/tearsofsteel.mp4"

def extract_frames_1fps(video_path, out_dir):
    print("📸 Extracting frames at 1 FPS...", flush=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open video: {video_path}")

    sec = 0
    saved = []
    while True:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_dir, f"frame_{sec:06d}.jpg")
        cv2.imwrite(fname, frame)
        saved.append(f"file://{fname}")
        sec += 1
        if sec % 10 == 0:
            print(f"  ⏱️ Extracted {sec} frames...", flush=True)
    cap.release()
    print(f"✅ Total frames extracted: {len(saved)}", flush=True)
    return saved

def ensure_model_downloaded(model_id):
    print(f"⬇️ Checking model cache for '{model_id}'...", flush=True)
    try:
        snapshot_download(model_id, local_files_only=False, resume_download=True)
        print("✅ Model ready (cached or downloaded).", flush=True)
    except Exception as e:
        print("⚠️ Model download failed:", e, flush=True)

def run_on_frames(frame_paths, max_new_tokens=384):
    print("⚙️ Loading model...", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL)
    print(f"✅ Model loaded in {time.time() - t0:.2f}s", flush=True)

    content = [{"type": "image", "image": path} for path in frame_paths]
    content.append({
        "type": "text",
        "text": "Describe this sequence of frames (1 FPS). Give per-frame captions and a brief summary."
    })
    messages = [{"role": "user", "content": content}]

    print("🧠 Preparing input tensors...", flush=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)

    print("🚀 Running inference...", flush=True)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    print("✨ Decoding output...", flush=True)
    out_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    result = processor.batch_decode(out_ids_trimmed, skip_special_tokens=True)
    return result[0]

if __name__ == "__main__":
    print("🔍 Script started.", flush=True)
    start_all = time.time()

    ensure_model_downloaded(MODEL)
    print("🎬 Beginning frame extraction...", flush=True)

    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames_1fps(VIDEO_PATH, td)
        print(f"🧩 {len(frames)} frames ready for model.", flush=True)
        out = run_on_frames(frames)
        print("\n=== 🧾 MODEL OUTPUT ===\n", flush=True)
        print(out, flush=True)

    print(f"\n🏁 Done in {time.time() - start_all:.2f}s", flush=True)
