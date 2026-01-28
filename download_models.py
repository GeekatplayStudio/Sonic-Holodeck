import os
import sys

def download_models():
    print("Downloading MusicGen Stereo Large and Dependencies...")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, snapshot_download

    models_path = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_path, exist_ok=True)

    # Note: AudioCraft usually handles downloading internally to a cache dir (often ~/.cache/huggingface).
    # This script pre-fetches them to ensure they are available offline if cache is used.
    
    repo_id = "facebook/musicgen-stereo-large"
    print(f"Downloading {repo_id}...")
    
    files = ["config.json", "pytorch_model.bin", "generation_config.json", "spiece.model", "tokenizer.json"]
    
    for f in files:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=f)
            print(f"Downloaded {f} to {path}")
        except Exception as e:
            print(f"Skipping {f} or error: {e}")

    # Download TangoFlux (High Fidelity)
    print("\nDownloading TangoFlux model (declare-lab/TangoFlux)...")
    try:
        snapshot_download(repo_id="declare-lab/TangoFlux")
        print("Downloaded TangoFlux successfully.")
    except Exception as e:
        print(f"Error downloading TangoFlux: {e}")

    print("\nModels specific to Sonic Holodeck setup are pre-cached.")
    print("When you run the node for the first time, AudioCraft will locate these in the cache.")

if __name__ == "__main__":
    download_models()
