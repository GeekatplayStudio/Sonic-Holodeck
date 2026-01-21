# ComfyUI Sonic-Holodeck 🎧

**Created by Geekatplay Studio**

A futuristic "DJ Deck" interface for ComfyUI that controls local SOTA music generation (MusicGen Stereo Large) with real-time broadcast-quality 3D audio visualization.

## Features

*   **Sonic-Holodeck Node**: A custom node that looks like a cyberpunk integrated circuit/DJ deck.
*   **3D Visualizer**: Real-time Three.js visualization (Neon Reactor + Retro-wave Grid) powered by specific audio frequency bands (Bass).
*   **Auto-Mastering**: Built-in audio processing chain (Compression, Limiting, Normalization to -14 LUFS equivalent) to make raw AI audio sound professional.
*   **MusicGen Stereo Large**: Integration with the high-fidelity stereo model.

## Installation

### Prerequisites
*   ComfyUI installed.
*   Python 3.10+.
*   FFmpeg installed on your system (required for audio processing).

### Automatic Installation

1.  Open the terminal in this folder.
2.  Run the installer:
    *   **Windows**: Double-click `installer.bat`
    *   **Mac/Linux**: Run `bash install.sh`
3.  Restart ComfyUI.

### Manual Installation

```bash
pip install -r requirements.txt
```

## Usage

1.  Add the **SonicHoloSynth** node (right click -> SonicHolodeck -> Holo-Synth).
2.  Connect the output `AUDIO` to a `SaveAudio` or `PreviewAudio` node.
3.  The 3D Visualizer will appear over the node.
4.  Use the **floating knobs** on the visualizer to control BPM, Duration, CFG (Creativity), and Temperature (Chaos).
5.  **Click and Drag** the central reactor ring to "scratch" or seemingly interact (visual effect).

## Models

The node uses `facebook/musicgen-stereo-large`. It will be downloaded automatically on the first run. 
To pre-download, run:
```bash
python download_models.py
```
