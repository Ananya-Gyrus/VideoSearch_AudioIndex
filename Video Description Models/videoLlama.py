import torch
import cv2
from PIL import Image
import numpy as np
import time
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

# ---------------------------------------------------
# 1. Download HF model & load VideoLLaVA
# ---------------------------------------------------
def load_model(model_name="LanguageBind/Video-LLaVA-7B"):
    print("🔄 Downloading and loading Video-LLaVA model from HuggingFace...")

    t0 = time.time()
    processor = VideoLlavaProcessor.from_pretrained(model_name)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    load_time = time.time() - t0
    print(f"✅ Model loaded successfully. (Time: {load_time:.2f} sec)\n")
    return processor, model

# ---------------------------------------------------
# 2. Extract frames at timestamps using OpenCV
# ---------------------------------------------------
def extract_frames(video_path, timestamps):
    print(f"🎞 Extracting frames from video: {video_path}")

    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    frames = []

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Warning: could not read frame at {t}s")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame).resize((1280, 720))
        frames.append(np.array(frame))

    cap.release()
    extract_time = time.time() - t0
    print(f"✅ Frames extracted. (Time: {extract_time:.2f} sec)\n")
    return frames

# ---------------------------------------------------
# 3. Run Video-LLaVA inference
# ---------------------------------------------------
def run_inference(model, processor, frames, transcript=None):
    prompt = (
        "You are a video understanding assistant.\n"
        f"Transcript: {transcript}\n\n"
        "Describe the video segment and extract OCR text and scene details."
    )

    print("🧠 Running Video-LLaVA inference...")
    t0 = time.time()

    output = model.chat(
        processor=processor,
        frames=frames,
        question=prompt
    )

    infer_time = time.time() - t0
    per_frame = infer_time / max(len(frames), 1)
    print(f"✅ Inference complete.")
    print(f"⏱ Total inference time: {infer_time:.2f} sec")
    print(f"⏱ Per frame time: {per_frame:.2f} sec/frame\n")
    return output, infer_time

# ---------------------------------------------------
# 4. MAIN end-to-end pipeline
# ---------------------------------------------------
if __name__ == "__main__":

    HF_MODEL = "LanguageBind/Video-LLaVA-7B"
    VIDEO_PATH = "sample_video.mp4"       # <<< replace with your video
    FRAME_TIMES = [1, 3, 5, 7]            # seconds
    TRANSCRIPT = "A player is running with the ball toward the goal."  # optional

    processor, model = load_model(HF_MODEL)
    frames = extract_frames(VIDEO_PATH, FRAME_TIMES)
    result, infer_time = run_inference(model, processor, frames, TRANSCRIPT)

    print("📌 FINAL OUTPUT:\n")
    print(result)
