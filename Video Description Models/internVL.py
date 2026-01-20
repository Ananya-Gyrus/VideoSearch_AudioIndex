import time
import subprocess
import os
import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "OpenGVLab/InternVL2-2B"
VIDEO_PATH = "/home/gyrus3pc/Desktop/SMOL Models/videos/tearsofsteel.mp4"
FRAME_INTERVAL = 1  # extract 1 frame per second

# -------------------------------
# Load model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Model loaded.")

# -------------------------------
# Extract frames
# -------------------------------
def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_id = 0
    next_capture = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id >= next_capture:
            # Convert BGR → RGB → PIL
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            frames.append(pil_img)
            next_capture += int(fps * interval)

        frame_id += 1

    cap.release()
    return frames

# -------------------------------
# Inference on the video
# -------------------------------
start_total = time.time()

print("Extracting frames...")
frames = extract_frames(VIDEO_PATH, FRAME_INTERVAL)
print(f"Extracted {len(frames)} frames.")

results = []

for idx, img in enumerate(frames):
    print(f"Running InternVL on frame {idx+1}/{len(frames)}...")

    # Build multimodal input
    query = [
        {"role": "user", 
         "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": "Describe what is happening in this frame."}
         ]}
    ]

    # Format
    inputs = tokenizer.apply_chat_template(
        query,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    # Generate
    output_ids = model.generate(
        inputs,
        max_new_tokens=200
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    results.append(output_text)


end_total = time.time()

print("\n=====================================")
print(f"🏁 TOTAL TIME: {end_total - start_total:.2f} sec")
print("=====================================")

print("\nSample Output:")
print(results[0][:500])
