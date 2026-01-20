# save as qwen_frames_infer.py
import os
import cv2
import tempfile
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEO_PATH = "/path/to/video.mp4"

def extract_frames_1fps(video_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    saved = []
    sec = 0
    while True:
        # set to frame corresponding to second 'sec'
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(out_dir, f"frame_{sec:06d}.jpg")
        cv2.imwrite(fname, frame)
        saved.append(f"file://{fname}")
        sec += 1  # next second -> 1 FPS
    cap.release()
    return saved

def run_on_frames(frame_paths, max_new_tokens=256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": "Describe this sequence of frames (1 FPS). Give per-frame caption/time and a short overall summary."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)  # when passing frame list, return_video_kwargs not necessary
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_texts

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        frames = extract_frames_1fps(VIDEO_PATH, td)
        print(f"Extracted {len(frames)} frames")
        out = run_on_frames(frames, max_new_tokens=384)
        print("Model output:\n", out[0])
