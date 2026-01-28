#!/bin/bash
echo "Installing Geekatplay Studio's Sonic-Holodeck..."

if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../venv
fi

source ../venv/bin/activate

echo "Installing core requirements..."
pip install -r ../requirements.txt --prefer-binary

echo "Installing AudioCraft (no deps)..."
pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps

# echo "Installing runtime dependencies..."
# pip install soundfile librosa protobuf pesq pystoi torchmetrics torchdiffeq

echo ""
echo "==================================================="
echo "INSTALLATION COMPLETE"
echo "To start, run ComfyUI with this venv active."
echo "==================================================="
