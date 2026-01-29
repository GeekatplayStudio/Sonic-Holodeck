# ComfyUI Sonic-Holodeck 

**Created by Geekatplay Studio**

A futuristic "DJ Deck" interface for ComfyUI that acts as a complete AI Audio Workstation. It consolidates local SOTA music generation (MusicGen, TangoFlux, HeartMuLa) and vocal synthesis (Bark) into a unified, professional mixing console.

##  Recent Updates
*   **HeartMuLa Fixes:** Optimized memory usage and fixed download scripts for HeartMuLa/HeartCodec (January 2026).
*   **TangoFlux Integration:** Added support for the high-speed, high-fidelity **TangoFlux** model.
*   **Global Model Cache:** Implemented smart memory management. The system now automatically unloads heavy models (MusicGen vs Flux) when switching workflows, determining which model to keep in memory to prevent VRAM overflow.
*   **Workflow Validation:** Relaxed validation rules on `SonicFluxSynth` and `SonicTrackLayer` so you can use lower step counts (down to 1) and optional vocal inputs.

---

##  Core Architecture & Modules

The Sonic Holodeck is modular by design, consisting of 7 core components that handle the entire audio production chain:

*   **Module A: Sonic-Holodeck (HoloSynth)**
    *   *Engine:* Facebook **MusicGen Stereo Large**.
    *   *Role:* The classic backing track generator. Creates rhythmic, musical foundations based on text prompts.
*   **Module B: Sonic Flux Synth  (New)**
    *   *Engine:* **TangoFlux** (Flow Matching).
    *   *Role:* High-Fidelity audio generator. Faster and higher quality for atmospheric/cinematic textures.
*   **Module C: Sonic DJ Mixer **
    *   *Role:* The "Brain". A style-aware prompt engineering node.
    *   *Features:* Supports 40+ genres. Automatically writes prompts optimized for the selected model.
*   **Module D: Sonic Singer **
    *   *Engine:* Suno **Bark**.
    *   *Role:* Vocal synthesizer.
*   **Module E: Sonic Track Layer **
    *   *Role:* Mixing Console.
    *   *Features:* Smart-mixes vocals and backing tracks. Handles auto-resampling and volume matching.
*   **Module F: Sonic Microphone V3 (Updated)**
    *   *Role:* Live Audio Input.
    *   *Features:* **"Press & Hold"** recording interface with native countdown timer and auto-stop. Includes Microphone source selection dropdown. 
    *   *Outputs:* Returns Audio, precise Float duration, and rounded Integer duration (for easy connection to MusicGen steps).

---

##  Workflow Guide (Standardized)

The project includes standardized workflows in the `workflows/` folder.

### 1. Sonic MusicGen Advanced (Standard)
*   **File:** `workflows/Sonic_MusicGen_Advanced.json`
*   **Goal:** Create a full song with Bark vocals and MusicGen backing.
*   **Best For:** Pop, Electronic, Rhythmic music.

### 2. Sonic TangoFlux High-Fidelity
*   **File:** `workflows/Sonic_TangoFlux_Workflow.json`
*   **Goal:** Generate 44.1kHz high-fidelity audio using the new TangoFlux model.
*   **Settings:** Default settings optimized for speed (Steps: 25, CFG: 4.5).
*   **Note:** Often referred to as "Stable Audio" quality due to its flow-matching architecture.

### 3. Sonic HeartMuLa Songwriter
*   **File:** `workflows/Sonic_HeartMuLa_Workflow.json`
*   **Goal:** End-to-end music+vocal generation using HeartMuLa 3B.
*   **Best For:** Cohesive songs where vocals match the beat perfectly.

### 4. Sonic MusicGen Hum-to-Music (Interactive)
*   **File:** `workflows/Sonic_MusicGen_HumToMusic.json`
*   **Goal:** Hum a melody into your microphone and have an instrument play it.

### 5. Sonic Utilities - Mixer Console
*   **File:** `workflows/Sonic_Utilities_MixerConsole.json`
*   **Goal:** A prompt-engineering dashboard to test styles without generating audio.

---

##  Installation

### Prerequisites
*   **ComfyUI** (Latest version recommended)
*   **Python 3.10+**
*   **NVIDIA GPU**

### Step 1: Core Installation
1.  Open your terminal in: `ComfyUI/custom_nodes/`
2.  Clone this repository:
    ```bash
    git clone https://github.com/GeekatplayStudio/Sonic-Holodeck
    ```
3.  Run the automated installer script:
    *   **Windows:** Run `installer.bat`

### Step 2: Restart
**Restart ComfyUI** completely to load the new nodes.

---

## Optional Installs (Recommended)
These optional dependencies improve codec support, performance, and overall stability. Install what fits your system and workflow.

### 1) ffmpeg (codec support, conversions)
Used by audio toolchains and often required for robust format handling.
Sources: https://ffmpeg.org/download.html | https://brew.sh | https://chocolatey.org/packages/ffmpeg

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

**Windows (Chocolatey):**
```powershell
choco install ffmpeg
```

### 2) libsndfile (better WAV/FLAC read/write)
Improves reliability for soundfile and other audio IO libraries.
Sources: https://libsndfile.github.io/libsndfile/ | https://brew.sh

**macOS (Homebrew):**
```bash
brew install libsndfile
```

**Ubuntu/Debian:**
```bash
sudo apt-get install -y libsndfile1
```

### 3) torchaudio (resampling + audio transforms)
Ensure it matches your installed torch version and CUDA runtime.
Sources: https://pypi.org/project/torchaudio/ | https://pytorch.org/get-started/locally/

```bash
pip install torchaudio
```

### 4) pydub (simple audio utilities)
Useful for format conversion and quick audio operations.
Sources: https://pypi.org/project/pydub/ | https://github.com/jiaaro/pydub

```bash
pip install pydub
```

### 5) xformers (optional speed/memory optimizations)
If available for your platform and Python/Torch version, can improve performance.
Sources: https://pypi.org/project/xformers/ | https://github.com/facebookresearch/xformers

```bash
pip install xformers
```

### 6) bitsandbytes (memory-efficient loading on CUDA)
Optional quantization support for compatible NVIDIA GPUs.
Sources: https://pypi.org/project/bitsandbytes/ | https://github.com/bitsandbytes-foundation/bitsandbytes

```bash
pip install bitsandbytes
```

> Tip: If installing `torchaudio`, `xformers`, or `bitsandbytes`, prefer matching wheels for your exact Torch + CUDA version. When in doubt, install Torch first, then add these packages.

---

##  Models & Checkpoints

| Model | File | Location | Auto-Download? |
| :--- | :--- | :--- | :--- |
| **MusicGen** | `facebook/musicgen-stereo-large` | HuggingFace Cache |  Yes |
| **TangoFlux** | `declare-lab/TangoFlux` | HuggingFace Cache |  Yes |
| **Bark** | `suno/bark` | HuggingFace Cache |  Yes |
| **HeartMuLa** | `HeartMuLa-3B` | `ComfyUI/models/checkpoints` |  Manual |

To pre-download standard models via script, run:
```bash
python download_models.py
```

---

##  Efficient Memory Management
The system now uses a **Global Model Cache**.
*   It ensures only **one** heavy audio model (MusicGen or TangoFlux) is in generated memory at a time.
*   Switching workflows will automatically unload the previous model to free VRAM.

