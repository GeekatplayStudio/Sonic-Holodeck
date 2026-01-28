# ComfyUI Sonic-Holodeck 🎧

**Created by Geekatplay Studio**

A futuristic "DJ Deck" interface for ComfyUI that acts as a complete AI Audio Workstation. It consolidates local SOTA music generation (MusicGen, HeartMuLa, Stable Audio) and vocal synthesis (Bark) into a unified, professional mixing console.

---

## 🏗️ Core Architecture & Modules

The Sonic Holodeck is modular by design, consisting of 7 core components that handle the entire audio production chain:

*   **Module A: Sonic-Holodeck (HoloSynth)**
    *   *Engine:* Facebook **MusicGen Stereo Large**.
    *   *Role:* The primary backing track generator. Creates rhythmic, musical foundations based on text prompts. Includes built-in auto-mastering (compression/limiting).
*   **Module B: Sonic DJ Mixer 🎛️**
    *   *Role:* The "Brain". A style-aware prompt engineering node.
    *   *Features:* Supports 40+ genres (Cyberpunk, K-Pop, Orchestral). Automatically adjusts CFG, Temperature, and Prompt Tags based on the selected **Target Model** (MusicGen vs HeartMuLa vs Stable Audio).
*   **Module C: Sonic Singer 🎤**
    *   *Engine:* Suno **Bark**.
    *   *Role:* Vocal synthesizer. Converts text lyrics into singing voice audio.
*   **Module D: Sonic Track Layer 🎚️**
    *   *Role:* The Mixing Console.
    *   *Features:* Smart-mixes vocals and backing tracks. Handles auto-resampling (converting different sample rates to match), volume balancing, and soft-clip limiting.
*   **Module E: Visualizers**
    *   *Role:* Audio analysis.
    *   *Tools:* **Sonic Spectrogram** (Frequency Heatmaps) and **Sonic Waveform** (Amplitude graphs).
*   **Module F: The "Holo-Fidelity" Engine**
    *   *Engine:* **Stable Audio Open**.
    *   *Role:* High-resolution texture generation (44.1kHz). Best for background scoring and sound design.
*   **Module G: The "Sonic Input" Deck**
    *   *Tool:* **Vrch Audio Recorder**.
    *   *Role:* Microphone integration. Allows "Hum-to-Music" workflows where your voice melody guides the AI generation.

---

## 📥 Installation

### Prerequisites
*   **ComfyUI** (Latest version recommended)
*   **Python 3.10+**
*   **NVIDIA GPU** (6GB+ VRAM recommended)

### Step 1: Core Installation
1.  Open your terminal in: `ComfyUI/custom_nodes/`
2.  Clone this repository:
    ```bash
    git clone https://github.com/GeekatplayStudio/Sonic-Holodeck
    ```
3.  Run the automated installer script:
    *   **Windows:** Run `Sonic-Holodeck\scripts\install_windows.bat`
    *   **Mac/Linux:** Run `bash Sonic-Holodeck/scripts/install_linux.sh`

### Step 2: Advanced Music AI Installation (Optional but Recommended)
To enable **HeartMuLa** (Vocals), **Stable Audio**, and **Microphone** support, you must install the diverse node pack:
1.  **Windows:** Run `Sonic-Holodeck\scripts\install_music_ai.bat`
   *(This script automatically clones and installs requirements for formatting strict audio environments)*

### Step 3: Restart
**Restart ComfyUI** completely to load the new nodes.

---

## 🎛️ Workflow Guide

The project includes pre-built JSON workflows in the `workflows/` folder.

### 1. Sonic Mix Console (Standard)
*   **File:** `workflows/sonic_holodeck_v2.json`
*   **Goal:** Create a full song with Bark vocals and MusicGen backing.
*   **How to Use:**
    1.  Select a **Style** in the `Sonic Mixer` (e.g., "Synthwave").
    2.  Set **Target Model** to `MusicGen`.
    3.  Enter lyrics in `Sonic Singer` (use `[verse]` / `[chorus]` tags).
    4.  Queue Prompt. The **Track Layer** will auto-mix the results.

### 2. HeartMuLa Songwriter (Advanced Vocals)
*   **File:** `workflows/music/heartmula_workflow.json`
*   **Goal:** Generate a cohesive song where music and vocals are generated together.
*   **How to Use:**
    1.  Ensure you have `HeartMuLa-3B.safetensors` in your checkpoints.
    2.  Set `Sonic Mixer` **Target Model** to `HeartMuLa`.
    3.  *Note:* The Mixer will automatically inject structural tags (`[Intro], [Verse], [Chorus]`) into the prompt based on your selected vibe.

### 3. Stable Audio High-Fidelity
*   **File:** `workflows/music/stable_audio_workflow.json`
*   **Goal:** Generate 44.1kHz background textures or sound effects.
*   **How to Use:**
    1.  Set `Sonic Mixer` **Target Model** to `StableAudio`.
    2.  Connect the mixer output to the `Stable Audio Sampler` node.
    3.  **Critical:** Keep duration **under 47 seconds** for this model.
    4.  The Mixer will automatically append "high fidelity, stereo, 44.1kHz" tags.

### 4. "Hum-to-Music" (Interactive)
*   **File:** `workflows/music/hum_to_music_workflow.json`
*   **Goal:** Hum a melody and have an instrument play it.
*   **How to Use:**
    1.  Add the `Vrch Audio Recorder` node.
    2.  Click **Record** and hum your tune.
    3.  Connect the `AUDIO` output to the `MusicGen` conditioning input.
    4.  Set the prompt to the instrument you want (e.g., "Electric Guitar").

---

## 🧩 Node Documentation

### **Sonic Mixer**
The central controller.
*   **Inputs:**
    *   `style`: Select from 40+ genres.
    *   `target_model`: Logic switch for MusicGen/HeartMuLa/StableAudio.
    *   `lyrics`: Text input for song lyrics.
    *   `vcoder_fx`: Vocal effects presets (Robotic, Ethereal, etc).
*   **Outputs:**
    *   `constructed_prompt`: The final engineered text prompt.
    *   `bpm`, `duration`, `cfg`, `temperature`: Optimized parameters for the chosen style.

### **Sonic Track Layer**
The mixing engine.
*   **Inputs:**
    *   `backing_audio`: The music track.
    *   `vocal_audio`: The vocal track.
    *   `alignment`: How to align them (`start`, `center`, `loop_backing`).
*   **Logic:** Automatically resamples the vocal track (often 24kHz) to match the backing track (32kHz or 44.1kHz) before mixing.

### **Sonic HoloSynth**
*   **Inputs:** `model_name` (default: facebook/musicgen-stereo-large), `prompt`, `bpm`, `duration`.
*   **Outputs:** `AUDIO` (Raw waveform).

---

## 📦 Models & Checkpoints

This project utilizes specific models. Some are auto-downloaded, others need manual placement.

| Model | File | Location | Auto-Download? |
| :--- | :--- | :--- | :--- |
| **MusicGen** | `facebook/musicgen-stereo-large` | HuggingFace Cache | ✅ Yes |
| **Bark** | `suno/bark` | HuggingFace Cache | ✅ Yes |
| **HeartMuLa** | `HeartMuLa-3B.safetensors` | `ComfyUI/models/checkpoints` | ❌ No |
| **Stable Audio** | `stable-audio-open-1.0.safetensors` | `ComfyUI/models/checkpoints` | ❌ No |

To pre-download standard models via script, run:
```bash
python scripts/download_models.py
```

---

## 🛠️ Troubleshooting

*   **"xformers" or "pesq" errors:**
    *   The project includes a built-in patcher in `nodes.py` to mock these libraries on Windows if they fail to load. If issues persist, ensure you are using the embedded Python environment.
*   **Out of Memory (OOM):**
    *   MusicGen Stereo Large requires ~4-6GB VRAM.
    *   Stable Audio Open is VRAM heavy; try reducing duration to 10s if crashing.
*   **Missing Nodes:**
    *   Ensure you ran `scripts/install_music_ai.bat` if you see red nodes related to AudioCraft or HeartMuLa.
