import os
import torch
import numpy as np
from LanguageBind.languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
import whisper

# Load once globally
WHISPER_MODEL = whisper.load_model("medium")


# LanguageBind model singleton
_model = None
_tokenizer = None

def get_languagebind_model(device=None):
    """
    Get the LanguageBind model as a singleton
    """
    global _model, _tokenizer
    
    if _model is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup clip types - Video is what we need for video search
        clip_type = {
            'video': 'LanguageBind_Video_FT',  # Full-tuned version for better performance
        }
        
        try:
            _model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
            _model = _model.to(device)
            _model.eval()
            
            # Load tokenizer for text processing
            pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
            _tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
            
            # print("LanguageBind model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            _model = None
            _tokenizer = None
    
    return _model, _tokenizer, device

def get_video_embedding(video_frames_4d, model, device, num_frames=8):
    """
    Generate a single embedding for the given video clip (sequence of frames).
    Averages the per-frame embeddings.
    
    Args:
        video_frames_4d: preprocessed video frames tensor with shape [T, C, H, W]
        model: LanguageBind model
        device: torch device
    
    Returns:
        single embedding tensor [1, embedding_dim] or None on error.
    """
    if model is None:
        return None
    
    try:
        inputs = {
            'video': to_device({'pixel_values': video_frames_4d}, device)
        }
        
        # Generate embedding
        with torch.no_grad():
            embeddings = model(inputs)
            # print(f"Embeddings: {embeddings.keys()}")
            # embeddings['video'] likely has shape [T, embedding_dim]
            per_frame_embeddings = embeddings['video'] 
            # print(f"Per-frame embeddings shape: {per_frame_embeddings.shape}")
            per_frame_embeddings = per_frame_embeddings.view(per_frame_embeddings.shape[0]//num_frames,num_frames,-1)
            
            # Average across the time dimension (T) to get a single clip embedding
            if per_frame_embeddings.dim() > 1 and per_frame_embeddings.shape[0] >= 1: # Check if we have multiple frames
                clip_embedding = per_frame_embeddings.mean(dim=1, keepdim=True) # Shape [1, embedding_dim]
                clip_embedding = clip_embedding.squeeze(1)
            elif per_frame_embeddings.dim() == 1: # If it's already a single vector
                 clip_embedding = per_frame_embeddings.unsqueeze(0) # Shape [1, embedding_dim]
            else: # Should be [1, embedding_dim] already if T=1
                 clip_embedding = per_frame_embeddings

        # Ensure output is [1, embedding_dim]
        if clip_embedding.dim() == 1:
            clip_embedding = clip_embedding.unsqueeze(0)
        elif clip_embedding.dim() != 2 or clip_embedding.shape[0] != 1:
            #  print(f"Warning: Unexpected clip embedding shape after pooling: {clip_embedding.shape}")
            pass
             # Fallback or error handling might be needed here

        return clip_embedding

    except Exception as e:
        print(f"Error generating video embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_text_embedding(text_query, model, tokenizer, device):
    """
    Generate an embedding for the given text query
    
    Args:
        text_query: text string
        model: LanguageBind model
        tokenizer: LanguageBind tokenizer
        device: torch device
    
    Returns:
        embedding tensor
    """
    if model is None or tokenizer is None:
        print("Error: LanguageBind model or tokenizer is None")
        return None
    
    try:
        # Tokenize the text
        inputs = tokenizer([text_query], max_length=77, padding='max_length', 
                          truncation=True, return_tensors='pt')
        inputs = to_device(inputs, device)
        
        # Generate embedding
        with torch.no_grad():
            embeddings = model({'language': inputs})
            text_embedding = embeddings['language']
        
        return text_embedding
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        import traceback
        traceback.print_exc()
        return None 

def get_text_embedding_batch(text_batch, model, tokenizer, device, batch_size=32):
    """
    Generate embeddings for a batch of text queries with batching support
    
    Args:
        text_batch: list of text strings
        model: LanguageBind model
        tokenizer: LanguageBind tokenizer
        device: torch device
        batch_size: maximum batch size for processing
    
    Returns:
        embedding tensor for all text queries
    """
    if model is None or tokenizer is None:
        print("Error: LanguageBind model or tokenizer is None")
        return None
    
    try:
        # Handle batch processing if text_batch is longer than batch_size
        if len(text_batch) > batch_size:
            all_embeddings = []
            for i in range(0, len(text_batch), batch_size):
                batch = text_batch[i:i + batch_size]
                text_inputs = tokenizer(batch, max_length=77, padding='max_length', 
                                     truncation=True, return_tensors='pt')
                text_inputs = to_device(text_inputs, device)
                
                with torch.no_grad():
                    embeddings = model({'language': text_inputs})
                    batch_embedding = embeddings['language']
                    all_embeddings.append(batch_embedding)
            
            # Concatenate all batch embeddings
            text_embedding = torch.cat(all_embeddings, dim=0)
            return text_embedding
        
        # Process normally if batch size is small enough
        text_inputs = tokenizer(text_batch, max_length=77, padding='max_length', 
                              truncation=True, return_tensors='pt')
        text_inputs = to_device(text_inputs, device)
        
        with torch.no_grad():
            embeddings = model({'language': text_inputs})
            text_embedding = embeddings['language']
        
        return text_embedding
        
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

import torchvision.transforms as transforms
from PIL import Image
import torchaudio
import io
torchaudio.set_audio_backend("soundfile")


def torchaudio_loader_from_bytes(audio_bytes):
    audio_stream = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(audio_stream)
    return waveform, sample_rate

image_model = None
image_tokenizer = None
def get_languagebind_image_model(device=None):
    """
    Get the LanguageBind model as a singleton
    """
    global image_model, image_tokenizer
    
    if image_model is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup clip types - Video is what we need for video search
        clip_type = {
            'image': 'LanguageBind_Image',  # Full-tuned version for better performance
        }
        
        try:
            image_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
            image_model = image_model.to(device)
            image_model.eval()
            
            # Load tokenizer for text processing
            pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
            image_tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
            
            # print("LanguageBind model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            image_model = None
            image_tokenizer = None
    
    return image_model, image_tokenizer, device


audio_model = None
audio_tokenizer = None
def get_languagebind_audio_model(device=None):
    """
    Get the LanguageBind model as a singleton
    """
    global audio_model, audio_tokenizer
    
    if audio_model is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup clip types - Video is what we need for video search
        clip_type = {
            'audio': 'LanguageBind_Audio_FT',  # Full-tuned version for better performance
        }
        # print("Loading LanguageBind audio model...")
        try:
            audio_model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
            audio_model = audio_model.to(device)
            audio_model.eval()
            # print(audio_model)
            # Load tokenizer for text processing
            pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
            audio_tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
            
            # print("LanguageBind model and tokenizer loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            audio_model = None
            audio_tokenizer = None
    return audio_model, audio_tokenizer, device

    
def preprocess_image(frame: Image.Image): # Added type hint for clarity
    """
    Preprocess a single PIL Image frame for LanguageBind model input.
    Resizes the frame to 224x224 and formats it as a [B, C, T, H, W] tensor.
    B=1, T=1 for a single input frame.
    """
    if frame is None: # Check if the frame itself is None
        print("Error: Input frame is None.")
        return None

    # Define transform with resize and normalization
    image_size = 224  # Required by the model
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # Converts PIL image [0,255] to tensor [0,1] of shape [C, H, W]
            normalize,  # Operates on tensor
        ]
    )

    try:
        transformed_tensor = transform_pipeline(frame)  # Output: [C, H, W]
    except Exception as e:
        print(f"Error during image transformation: {e}")
        return None

    # The check 'if not transformed_tensor:' was problematic.
    # If transform_pipeline succeeds, transformed_tensor will be a valid tensor.
    # If it fails, it should raise an exception caught above.

    # Reshape [C, H, W] to [B, C, T, H, W]
    # For a single frame, B=1 and T=1.
    # Add Time dimension (T=1)
    clip_tensor_t = transformed_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    # Add Batch dimension (B=1)
    # final_clip_tensor = clip_tensor_t.unsqueeze(0)   # Shape: [1, 1, C, H, W]

    return clip_tensor_t

def get_image_embedding(image, model, device):
    if model is None :
        print("Error: LanguageBind model ")
        return None
    video_frames_4d = preprocess_image(image)
    # print(video_frames_4d.shape)
    try:
        inputs = {
            'image': to_device({'pixel_values': video_frames_4d}, device)
        }
        
        # Generate embedding
        with torch.no_grad():
            embeddings = model(inputs)
            # embeddings['video'] likely has shape [T, embedding_dim]
            per_frame_embeddings = embeddings['image'] 
            
            # Average across the time dimension (T) to get a single clip embedding
            if per_frame_embeddings.dim() > 1 and per_frame_embeddings.shape[0] > 1: # Check if we have multiple frames
                clip_embedding = per_frame_embeddings.mean(dim=0, keepdim=True) # Shape [1, embedding_dim]
            elif per_frame_embeddings.dim() == 1: # If it's already a single vector
                 clip_embedding = per_frame_embeddings.unsqueeze(0) # Shape [1, embedding_dim]
            else: # Should be [1, embedding_dim] already if T=1
                 clip_embedding = per_frame_embeddings
        # print(clip_embedding.shape)
        # Ensure output is [1, embedding_dim]
        if clip_embedding.dim() == 1:
            clip_embedding = clip_embedding.unsqueeze(0)
        elif clip_embedding.dim() != 2 or clip_embedding.shape[0] != 1:
             print(f"Warning: Unexpected clip embedding shape after pooling: {clip_embedding.shape}")
             # Fallback or error handling might be needed here
             pass

        return clip_embedding

    except Exception as e:
        print(f"Error generating image embedding: {e}")
        import traceback
        traceback.print_exc()
        return None

def waveform2melspec(audio_data):
    audio_mean=-4.2677393
    audio_std=4.5689974
    target_length=1036
    mel = get_mel(audio_data)
    if mel.shape[0] > target_length:
        # split to three parts
        chunk_frames = target_length
        total_frames = mel.shape[0]
        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
        # print('total_frames-chunk_frames:', total_frames-chunk_frames,
        #       'len(audio_data):', len(audio_data),
        #       'chunk_frames:', chunk_frames,
        #       'total_frames:', total_frames)
        if len(ranges[1]) == 0:  # if the audio is too short, we just use the first chunk
            ranges[1] = [0]
        if len(ranges[2]) == 0:  # if the audio is too short, we just use the first chunk
            ranges[2] = [0]
        # randomly choose index for each part
        idx_front = np.random.choice(ranges[0])
        idx_middle = np.random.choice(ranges[1])
        idx_back = np.random.choice(ranges[2])
        # idx_front = ranges[0][0]  # fixed
        # idx_middle = ranges[1][0]
        # idx_back = ranges[2][0]
        # select mel
        mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
        mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
        mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
        # print(total_frames, idx_front, idx_front + chunk_frames, idx_middle, idx_middle + chunk_frames, idx_back, idx_back + chunk_frames)
        # stack
        mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
    elif mel.shape[0] < target_length:  # padding if too short
        n_repeat = int(target_length / mel.shape[0]) + 1
        # print(self.target_length, mel.shape[0], n_repeat)
        mel = mel.repeat(n_repeat, 1)[:target_length, :]
        mel_fusion = torch.stack([mel, mel, mel], dim=0)
    else:  # if equal
        mel_fusion = torch.stack([mel, mel, mel], dim=0)
    # print("came till here")
    mel_fusion = mel_fusion.transpose(1, 2)  # [3, target_length, mel_bins] -> [3, mel_bins, target_length]

    # self.mean.append(mel_fusion.mean())
    # self.std.append(mel_fusion.std())
    mel_fusion = (mel_fusion - audio_mean) / (audio_std * 2)
    # print(mel_fusion.shape)
    return mel_fusion

def get_mel(audio_data):
    # mel shape: (n_mels, T)
    sample_rate = 16000
    num_mel_bins = 112  
    DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # Default frame shift in milliseconds
    audio_data -= audio_data.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        audio_data,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    return mel  # (T, n_mels)

def preprocess_audio(audio):
    sample_rate = 16000
    
    if audio is None:
        print("Error: Input audio is None.")
        return None
    
    waveform_and_sr = torchaudio_loader_from_bytes(audio)
    audio_data, origin_sr = waveform_and_sr
    if sample_rate != origin_sr:
        # print(audio_data.shape, origin_sr)
        audio_data = torchaudio.functional.resample(audio_data, orig_freq=origin_sr, new_freq=sample_rate)
    waveform_melspec = waveform2melspec(audio_data)
    waveform_melspec = waveform_melspec.unsqueeze(0)  # Add batch dimension [1, 3, mel_bins, target_length]
    return waveform_melspec

def get_audio_embedding(audio, model, device):
    # print(audio)
    if model is None :
        print("Error: LanguageBind model is not loaded")
        return None
    # print("#"* 20, audio.shape)
    audio_frames_4d = preprocess_audio(audio)
    # print("#"* 20, audio_frames_4d.shape)
    try:
        inputs = {
            'audio': to_device({'pixel_values': audio_frames_4d}, device)
        }
        
        # Generate embedding
        with torch.no_grad():
            embeddings = model(inputs)
            # embeddings['video'] likely has shape [T, embedding_dim]
            per_frame_embeddings = embeddings['audio'] 
            # print(per_frame_embeddings.shape)
            # Average across the time dimension (T) to get a single clip embedding
            if per_frame_embeddings.dim() > 1 and per_frame_embeddings.shape[0] > 1: # Check if we have multiple frames
                clip_embedding = per_frame_embeddings.mean(dim=0, keepdim=True) # Shape [1, embedding_dim]
            elif per_frame_embeddings.dim() == 1: # If it's already a single vector
                 clip_embedding = per_frame_embeddings.unsqueeze(0) # Shape [1, embedding_dim]
            else: # Should be [1, embedding_dim] already if T=1
                 clip_embedding = per_frame_embeddings
        # print(clip_embedding.shape)
        # Ensure output is [1, embedding_dim]
        if clip_embedding.dim() == 1:
            clip_embedding = clip_embedding.unsqueeze(0)
        elif clip_embedding.dim() != 2 or clip_embedding.shape[0] != 1:
             print(f"Warning: Unexpected clip embedding shape after pooling: {clip_embedding.shape}")
             # Fallback or error handling might be needed here

        return clip_embedding

    except Exception as e:
        print(f"Error generating audio embedding: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def transcribe_tamil_to_english(wav_path):
    """
    Uses Whisper Large-v3 to transcribe Tamil audio and translate to English.
    """
    result = WHISPER_MODEL.transcribe(wav_path, task="translate")
    english_text = result["text"].strip()
    
    if not english_text:
        english_text = ""

    return english_text


def get_audio_embeddings_batch(audio_list, model, device, batch_size=8):
    """
    Generate embeddings for a batch of audios using the LanguageBind model.
    Handles batching, device alignment, and pooling.
    """

    if model is None:
        print("Error: LanguageBind model is not loaded")
        return None

    all_embeddings = []

    for i in range(0, len(audio_list), batch_size):
        batch_audios = audio_list[i:i + batch_size]
        batch_melspecs = []

        # --- Step 1: Preprocess each audio into mel spectrograms ---
        for audio in batch_audios:
            if audio is None:
                print("Warning: Skipping None audio input.")
                continue
            mel = preprocess_audio(audio)
            if mel is not None:
                batch_melspecs.append(mel)

        if not batch_melspecs:
            continue

        # --- Step 2: Stack into tensor [B, 3, mel_bins, target_length] ---
        batch_tensor = torch.cat(batch_melspecs, dim=0)

        # --- Step 3: Move to correct device and dtype ---
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        batch_tensor = batch_tensor.to(device=model_device, dtype=model_dtype)



        try:
            inputs = {'audio': {'pixel_values': batch_tensor}}

            # --- Step 4: Forward pass ---
            with torch.no_grad():
                embeddings = model(inputs)
                per_frame_embeddings = embeddings['audio']

                # --- Step 5: Pool if multiple frames ---
                if per_frame_embeddings.dim() == 3:
                    clip_embeddings = per_frame_embeddings.mean(dim=1)
                elif per_frame_embeddings.dim() == 2:
                    clip_embeddings = per_frame_embeddings
                else:
                    print(f"Warning: Unexpected embedding shape {per_frame_embeddings.shape}")
                    continue

            # --- Step 6: Store results ---
            all_embeddings.append(clip_embeddings.cpu())

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping batch — failed embeddings")

    # --- Step 7: Combine all embeddings ---
    if all_embeddings:
        final_embeds = torch.cat(all_embeddings, dim=0)
        print(final_embeds.shape)
        return final_embeds

    else:
        return None

