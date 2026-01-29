import os
from huggingface_hub import snapshot_download

# Define paths
base_model_path = r"D:\ComfyUI\0.10\ComfyUI\models\checkpoints\HeartMuLa-oss-3B"
codec_path = os.path.join(base_model_path, "HeartCodec-oss")

# Ensure base folder exists (it should, based on previous `dir`)
if not os.path.exists(base_model_path):
    print(f"Error: Base model folder not found at {base_model_path}")
    exit(1)

print(f"Downloading HeartCodec-oss to: {codec_path}")
try:
    snapshot_download(repo_id="filliptm/HeartCodec-oss", local_dir=codec_path)
    print("Download complete.")
    
    # Verify
    if os.path.exists(os.path.join(codec_path, "model.safetensors")):
        print("Success: model.safetensors found in codec folder.")
    else:
        print("Warning: model.safetensors missing from codec download?")
        
except Exception as e:
    print(f"Download failed: {e}")
