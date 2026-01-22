# ComfyUI Sonic-Holodeck 🎧

**Created by Geekatplay Studio**

A futuristic "DJ Deck" interface for ComfyUI that controls local SOTA music generation (MusicGen Stereo Large) and vocal synthesis (Bark), complete with mixing tools and professional visualizers.

## Features

*   **Sonic-Holodeck (HoloSynth)**: Generates high-fidelity stereo music using Facebook's MusicGen Large Stereo model. Includes auto-mastering (compression/limiting) for pro sound.
*   **Sonic DJ Mixer 🎛️**: A style-aware prompt builder. Supports 40+ genres including Cyberpunk, K-Pop, Orchestral, Swing, Metal, and more. Tuning parameters (CFG/Temperature) are automatically optimized for the chosen genre.
*   **Sonic Singer 🎤**: Vocal synthesis using Suno's **Bark** model. Generates vocals from text lyrics with various voice presets.
*   **Sonic Track Layer 🎚️**: A professional mixer node to combine backing tracks with vocals. Features auto-resampling, volume control, and soft-clipping.
*   **Visualizers**:
    *   **Sonic Spectrogram**: Generates frequency heatmaps (Magma, Inferno, Viridis, etc.) for audio analysis.
    *   **Sonic Waveform**: Generates amplitude plots with customizable colors.
*   **Sonic Saver 💾**: Saves audio with an embedded HTML5 player directly on the node for instant playback in ComfyUI.

## Installation

### Automatic Installation (Recommended)

1.  Open the terminal in `ComfyUI/custom_nodes/Sonic-Holodeck`.
2.  Run the installer:
    *   **Windows**: Double-click `installer.bat`
    *   **Mac/Linux**: Run `bash install.sh`
3.  **Restart ComfyUI**.

### Manual Installation

If you prefer installing manually or are troubleshooting:

```bash
# 1. Install Dependencies
pip install -r requirements.txt --prefer-binary

# 2. Install AudioCraft (without deps to protect your Torch version)
pip install git+https://github.com/facebookresearch/audiocraft.git --no-deps
```

> **Note for Windows Users:** This node includes automatic patching for `xformers` and `pesq` issues common on Windows/Python 3.10+. It is designed to work out-of-the-box on standard ComfyUI portable builds.

## Workflow Usage

1.  **Load the Workflow**: Drag and drop `workflow/Sonic-Holodeck-Workflow_V2.json` into ComfyUI.
2.  **Select Style**: Choose from 40+ Music Styles in the **Sonic Mixer**.
3.  **Write Lyrics**: Enter lyrics for the **Sonic Singer** (use `[verse]` / `[chorus]` tags).
4.  **Generate**: Queue the prompt.
    *   The **HoloSynth** will generate the beat.
    *   The **Singer** will generate the vocals.
    *   The **Track Layer** will mix them together.
    *   The **Saver** will save the file and display a player.
    *   **Visualizers** will generate images of your track.

## Models

*   **MusicGen**: `facebook/musicgen-stereo-large` (Auto-downloaded to HuggingFace cache).
*   **Bark**: `suno/bark` (Auto-downloaded).

To pre-download models manually, run:
```bash
python download_models.py
```
