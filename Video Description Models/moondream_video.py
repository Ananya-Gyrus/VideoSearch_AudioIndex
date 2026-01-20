import os
import math
import torch
import tempfile
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "OpenGVLab/InternVL2-2B"
VIDEO_PATH = "/path/to/your/video.mp4"
CHUNK_SECONDS = 60        # 1-min chunks
FPS = 1.0                 # InternVL will sample 1 FPS frames

# ----------------------------
# LOAD MODEL ONCE
# ----------------------------
print("Loading InternVL2-2B...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device).eval()

print("Loaded InternVL.\n")

# ----------------------------
# EXTRACT CHUNK WITH FFMPEG
# ----------------------------
def extract_chunk(video, start, duration):
    out = tempfile.mktemp(suffix=".mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", video,
        "-t", str(duration),
        "-c", "copy",
        out
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out

# ----------------------------
# RUN INFER ON ONE CHUNK
# ----------------------------
def infer_chunk(video_chunk_path, fps=1.0):
    """
    InternVL takes videos like:
    {"type": "video", "video": <numpy array>}

    But we can let InternVL auto-handle video reading
    by passing the filepath directly.
    """

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_chunk_path, "fps": fps},
                {"type": "text", "text": "Give me a short summary of this video clip."}
            ]
        }
    ]

    inputs = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=False
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text


# ----------------------------
# MAIN LOGIC
# ----------------------------
if __name__ == "__main__":

    # Get video duration
    dur = float(subprocess.check_output(
        f"ffprobe -v error -show_entries format=duration "
        f"-of default=noprint_wrappers=1:nokey=1 '{VIDEO_PATH}'",
        shell=True
    ))

    total_chunks = math.ceil(dur / CHUNK_SECONDS)
    print(f"Total duration: {dur:.2f}s → {total_chunks} chunks")

    # Process sequential chunks
    for idx in range(total_chunks):
        start = idx * CHUNK_SECONDS
        print(f"\n=== Processing chunk {idx+1}/{total_chunks}, start={start}s ===")

        chunk_file = extract_chunk(VIDEO_PATH, start, CHUNK_SECONDS)

        try:
            out = infer_chunk(chunk_file, fps=FPS)
            print(out)
        except Exception as e:
            print("Chunk failed:", e)
        finally:
            os.remove(chunk_file)
