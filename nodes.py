import torch
import torchaudio
import os

# Suppress HuggingFace cache warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import folder_paths
import math
import numpy as np
import sys
import types
import importlib.machinery
import matplotlib.pyplot as plt
import io
import librosa
import soundfile as sf
from PIL import Image

def create_mock_module(name):
    mock = types.ModuleType(name)
    # Create a dummy ModuleSpec to satisfy importlib.util.find_spec checks
    mock.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mock

# --- PATCH: Handling missing xformers for AudioCraft ---
# AudioCraft unconditionally imports xformers in some versions.
# If it's missing (common on Windows Py3.13), we mock it to prevent crash.
try:
    import xformers
    import xformers.ops
except ImportError:
    print("Sonic-Holodeck: 'xformers' not found. Mocking it to allow AudioCraft import.")
    xformers_mock = create_mock_module("xformers")
    xformers_ops_mock = create_mock_module("xformers.ops")
    xformers_mock.ops = xformers_ops_mock
    sys.modules["xformers"] = xformers_mock
    sys.modules["xformers.ops"] = xformers_ops_mock

# Mock pesq if missing (fails to build on some Windows envs)
try:
    import pesq
except ImportError:
    print("Sonic-Holodeck: 'pesq' not found. Mocking it to allow AudioCraft import.")
    pesq_mock = create_mock_module("pesq")
    sys.modules["pesq"] = pesq_mock

# Mock torchtext if missing (deprecated/removed in newer Torch)
try:
    import torchtext
except ImportError:
    print("Sonic-Holodeck: 'torchtext' not found. Mocking it to allow AudioCraft import.")
    torchtext_mock = create_mock_module("torchtext")
    sys.modules["torchtext"] = torchtext_mock

# --- ADVANCED PATCH: Disable xformers in AudioCraft ---
# AudioCraft allows verifying xformers, which crashes if our mock is incomplete.
# We patch the Transformer init to FORCE disable memory_efficient attention.
try:
    import audiocraft.modules.transformer
    
    # 1. Patch the verification function to do nothing
    audiocraft.modules.transformer._verify_xformers_memory_efficient_compat = lambda: None
    
    # 2. Patch StreamingMultiheadAttention to force memory_efficient=False
    _original_attn_init = audiocraft.modules.transformer.StreamingMultiheadAttention.__init__
    
    def _patched_attn_init(self, *args, **kwargs):
        if kwargs.get('memory_efficient', False):
            print("Sonic-Holodeck: Forcing memory_efficient=False in StreamingMultiheadAttention")
            kwargs['memory_efficient'] = False
        return _original_attn_init(self, *args, **kwargs)
        
    audiocraft.modules.transformer.StreamingMultiheadAttention.__init__ = _patched_attn_init
    print("Sonic-Holodeck: Patched AudioCraft to disable xformers dependency.")
    
except ImportError:
    # This might happen if requirements are not yet installed; 
    # The node will fail gracefully later anyway.
    pass

# --------------------------------------------------------

class SonicHoloSynth:
    """
    The 'Holo-Synth' Generator & Controller.
    """
    
    def __init__(self):
        self.model = None
        self.processor = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["facebook/musicgen-stereo-large", "facebook/musicgen-large", "facebook/musicgen-stereo-medium", "facebook/musicgen-medium", "facebook/musicgen-small"], {"default": "facebook/musicgen-stereo-large"}),
                "prompt": ("STRING", {"multiline": True, "default": "Cyberpunk city rain, neon lights, synthwave, slow tempo"}),
                "bpm": ("INT", {"default": 120, "min": 60, "max": 200, "step": 1}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 300, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_k": ("INT", {"default": 250, "min": 10, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mastering": (["Auto-Master (Loud)", "Soft Clip", "None"], {"default": "None"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "bpm_out")
    FUNCTION = "generate_music"
    CATEGORY = "Geekatplay Studio"

    def apply_auto_mastering(self, wav, sample_rate, mode="Auto-Master (Loud)"):
        """
        Applies a mastering chain:
        1. Soft-knee Compressor
        2. Make-up Gain
        3. Limiter / Soft Clipper
        4. Normalization to approx -14dB integrated loudness (simplified RMS matching)
        """
        # Ensure wav is [channels, length]
        if wav.dim() == 3:
            wav = wav.squeeze(0)
            
        if mode == "None":
            return wav.unsqueeze(0)
            
        if mode == "Soft Clip":
             # Simple compression without aggressive gain
             return torch.tanh(wav).unsqueeze(0)
        
        # --- Auto-Master (Loud) ---
        
        # 1. Simple Compression (Simulated using tanh for soft clipping/saturation)
        # Real compression requires stateful processing or complex envelope following. 
        # A soft clipper is a good approximation for "loudening" in a simple script.
        
        # Drive input slightly 
        pre_gain = 1.3 # Reduced from 1.5 to reduce distortion
        compressed = torch.tanh(wav * pre_gain)
        
        # 2. Normalize to Target Level
        # Target RMS for -14dBFS is approx 10^(-14/20) = 0.2
        target_rms = 0.2
        
        current_rms = torch.sqrt(torch.mean(compressed**2))
        
        if current_rms > 0:
            gain_adjust = target_rms / current_rms
            # Cap gain adjustment to avoid noise floor explosion
            gain_adjust = min(gain_adjust, 3.0) 
            mastered = compressed * gain_adjust
        else:
            mastered = compressed

        # 3. Final Safety Limiter (Hard clip at -0.1dB)
        ceiling = 0.99
        mastered = torch.clamp(mastered, -ceiling, ceiling)
        
        # Reshape back to [1, C, L] for ComfyUI if needed, but Comfy usually takes [C, L] or [1, C, L]
        return mastered.unsqueeze(0)

    def generate_music(self, model_name, prompt, bpm, duration, cfg, temperature, top_k, top_p, mastering, seed):
        try:
            from audiocraft.models import MusicGen
        except ImportError as e:
            raise ImportError(f"Failed to import audiocraft. Error: {e}")

        # Basic model caching by name
        # If user switches model in dropdown, we need to reload. 
        # For simplicity, we just store the last loaded model name.
        if self.model is None or getattr(self, 'current_model_name', '') != model_name:
            # Unload old model if needed (though Python GC handles this eventually)
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            
            print(f"Loading MusicGen model: {model_name}...")
            self.model = MusicGen.get_pretrained(model_name)
            self.current_model_name = model_name

        torch.manual_seed(seed)
        
        self.model.set_generation_params(
            duration=duration, 
            top_k=top_k, 
            top_p=top_p, 
            temperature=temperature, 
            cfg_coef=cfg
        )

        print(f"Generating audio: '{prompt}' (BPM: {bpm})...")
        # Wav is [batch, channels, length]
        wav = self.model.generate([prompt], progress=True)

        # Auto-Mastering
        print(f"Applying Mastering: {mastering}...")
        mastered_wav = self.apply_auto_mastering(wav.cpu(), self.model.sample_rate, mode=mastering)

        audio_output = {
            "waveform": mastered_wav, 
            "sample_rate": self.model.sample_rate
        }

        return (audio_output, float(bpm))

class SonicSinger:
    """
    Experimental Vocal Generator using Bark.
    """
    def __init__(self):
        self.preload_models()

    def preload_models(self):
        # We assume bark is installed
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": "[verse]\nIn the neon rain, I lost my soul.\n[chorus]\nCybernetic heart, losing control."}),
                "voice_preset": (["v2/en_speaker_6", "v2/en_speaker_9", "v2/en_speaker_0", "announcer"],),
                "mode": (["text", "melody"],), 
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("vocal_audio",)
    FUNCTION = "sing"
    CATEGORY = "Geekatplay Studio"

    def sing(self, lyrics, voice_preset, mode):
        # Graceful handling of empty lyrics (Silence)
        if not lyrics or not lyrics.strip():
            # Return 1 second of silence
            print("SonicSinger: No lyrics provided, generating silence.")
            # Standard Bark sample rate is 24000
            silence = torch.zeros((1, 1, 24000), dtype=torch.float32)
            return ({"waveform": silence, "sample_rate": 24000},)

        # Redirect Bark/Suno downloads to ComfyUI/models/suno
        import os
        # Bark uses XDG_CACHE_HOME/suno/bark_v0
        # Check if XDG_CACHE_HOME is already set to something custom, if not, force it to models_dir
        # But to ensure it goes to keys we want, we just overwrite it for this session usually.
        # Ideally we want: ComfyUI/models/suno
        # IF XDG_CACHE_HOME = ComfyUI/models, then result is ComfyUI/models/suno/bark_v0
        os.environ["XDG_CACHE_HOME"] = folder_paths.models_dir
        
        print(f"Generating vocals with Bark: {voice_preset}")
        
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
        except ImportError:
            raise ImportError("Please install Bark: pip install git+https://github.com/suno-ai/bark.git")
            
        # Helper to ensure models are loaded
        preload_models()

        # Generate audio from text
        # Bark is pure text-to-speech effectively, but with [tags] can sing somewhat.
        audio_array = generate_audio(lyrics, history_prompt=voice_preset)

        # Bark output is numpy array float32
        # Convert to torch tensor [channels, samples]
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Add channel dimension if needed (Bark is mono usually)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        return ({"waveform": audio_tensor.unsqueeze(0), "sample_rate": SAMPLE_RATE},)

class SonicMixer:
    """
    Advanced 'DJ Mixer' Node that constructs the prompt and parameters.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        style_list = [
            # Electronic & Dance
            "Cyberpunk", "Synthwave", "Techno", "House", "Deep House", "Tech House", 
            "Trance", "Psytrance", "Dubstep", "Drum & Bass", "Liquid DnB", 
            "Ambient", "Chillout", "Lo-Fi", "Industrial", "Eurobeat", "Experimental",
            "Hardstyle", "Gabber", "EDM", "IDM", "Vaporwave", "Electro Swing",
            
            # Pop & Modern
            "Pop", "Euro Pop", "K-Pop", "J-Pop", "Synthpop", "Britpop", "Dream Pop",
            "Indie Pop", "Disco",
            
            # Rock & Metal
            "Rock", "Metal", "Heavy Metal", "Black Metal", "Death Metal", 
            "Punk", "Indie Rock", "Grunge", "Alternative Rock", "Psychedelic Rock",
            
            # Urban & Rhythm
            "Hip Hop", "Trap", "R&B", "Funk", "Soul", "Neo-Soul", "Reggaeton",
            
            # Jazz & Vintage
            "Jazz", "Smooth Jazz", "Bebop", "Swing", "Big Band", "Dixieland", 
            "Ragtime", "Blues", "Gospel", "Doo-Wop",
            
            # Classical & Orchestral
            "Cinematic", "Orchestral", "Classical", "Baroque", "Rococo", 
            "Romantic", "Chamber Music", "Opera", "Piano Solo", "Gregorian Chant",
            
            # World & Traditional
            "Reggae", "Ska", "Dub", "Latin", "Salsa", "Bossa Nova", "Samba", 
            "Tango", "Flamenco", "Afrobeat", "Celtic", "Polka", "Folk", "Country", "Bluegrass"
        ]
        style_list.sort() # Alphabetical is usually nicer for long lists
        
        return {
            "required": {
                "target_model": (["MusicGen", "HeartMuLa", "StableAudio"], {"default": "MusicGen"}),
                "lyrics": ("STRING", {"multiline": True, "default": "In the neon rain...", "placeholder": "Enter Lyrics here"}),
                "style": (style_list, {"default": "Cyberpunk"}),
                "mode": (["Song", "Instrumental"], {"default": "Song"}),
                "instruments": ("STRING", {"default": "Synthesizer, Drum Machine, Bass", "multiline": False}),
                "vocoder_fx": (["None", "Robotic", "Ethereal", "Distorted"],),
                "bpm": ("INT", {"default": 120, "min": 40, "max": 240}),
                "duration": ("INT", {"default": 15, "min": 5, "max": 300}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("constructed_prompt", "bpm", "duration", "cfg", "temperature", "lyrics_out")
    FUNCTION = "mix_track"
    CATEGORY = "Geekatplay Studio"

    def mix_track(self, target_model, lyrics, style, mode, instruments, vocoder_fx, bpm, duration):
        # Define Style Groups for Logic
        strong_styles = ["Techno", "Cyberpunk", "Dubstep", "Drum & Bass", "Industrial", "Hardstyle", "Gabber", "Metal", "Heavy Metal", "Rock", "Punk"]
        organic_styles = ["Ambient", "Lo-Fi", "Chillout", "Jazz", "Bossa Nova", "Folk", "Acoustic", "Swing", "Reggae"]
        orchestral_styles = ["Orchestral", "Cinematic", "Classical", "Baroque", "Rococo", "Opera"]
        pop_styles = ["Pop", "K-Pop", "J-Pop", "Euro Pop", "Disco", "Hip Hop", "Trap", "R&B", "Funk", "Soul"]

        # Construct a rich prompt
        prompt_parts = []
        
        # -- MODEL SPECIFIC LOGIC: PRE-PEND --
        if target_model == "HeartMuLa":
            # Add structural hints based on "Vibe"
            if style in strong_styles:
                prompt_parts.append("[Intro] High energy, Intense, Loud [Chorus]")
            elif style in pop_styles:
                prompt_parts.append("[Verse] Upbeat, Catchy [Chorus]")
            else:
                prompt_parts.append("[Verse] Melodic, Flowing [Chorus]")
        
        # Style base
        prompt_parts.append(f"{style} track")
        
        # Instruments
        if instruments:
            prompt_parts.append(f"featuring {instruments}")
        
        # Vocals/Lyrics intention
        # Only add vocal keywords if mode is Song
        if mode == "Song" and lyrics.strip():
            prompt_parts.append("with vocals, singing")
            if vocoder_fx != "None":
                prompt_parts.append(f"{vocoder_fx} vocal effects")
        else:
            # Force instrumental prompt
            prompt_parts.append("instrumental, no vocals")
            # Clear lyrics output to silence Singer
            lyrics = ""
        
        # BPM context
        prompt_parts.append(f"{bpm} bpm")
        
        # High fidelity keywords
        prompt_parts.append("high fidelity, stereo, masterpiece")

        # -- MODEL SPECIFIC LOGIC: APPEND --
        if target_model == "StableAudio":
            prompt_parts.append("44.1kHz, high fidelity, stereo")
        
        final_prompt = ", ".join(prompt_parts)
        
        # Set parameters based on style
        cfg = 3.0
        temperature = 1.0

        if style in strong_styles:
            cfg = 4.5 
        elif style in organic_styles:
             temperature = 1.15
        elif style in orchestral_styles:
             cfg = 3.5
        elif style in pop_styles:
            cfg = 3.2
             
        # Pass lyrics through for vocal nodes (empty if instrumental)
        return (final_prompt, bpm, duration, cfg, temperature, lyrics)

class SonicTrackLayer:
    """
    Mixes two audio tracks (e.g., Backing Track + Vocals).
    Handles resampling and length matching.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backing_audio": ("AUDIO",),
                "vocal_audio": ("AUDIO",),
                "backing_volume": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1}),
                "vocal_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "alignment": (["start", "center", "loop_backing", "loop_vocals"], {"default": "start"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "mix_tracks"
    CATEGORY = "Geekatplay Studio"

    def mix_tracks(self, backing_audio, vocal_audio, backing_volume, vocal_volume, alignment):
        # Unpack
        # ComfyUI audio format: {"waveform": [batch, channels, time], "sample_rate": int}
        b_wav = backing_audio["waveform"]
        b_sr = backing_audio["sample_rate"]
        
        v_wav = vocal_audio["waveform"]
        v_sr = vocal_audio["sample_rate"] # Likely 24k for Bark vs 32k/44k for MusicGen

        # 1. Resample Vocals to match Backing if needed
        if v_sr != b_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=v_sr, new_freq=b_sr)
            v_wav = resampler(v_wav)
            # v_sr is now b_sr

        # 2. Match Channels
        # Assume stereo target. If mono, duplicate.
        if b_wav.shape[1] == 1:
            b_wav = b_wav.repeat(1, 2, 1)
        if v_wav.shape[1] == 1:
            v_wav = v_wav.repeat(1, 2, 1)

        # 3. Handle Lengths
        b_len = b_wav.shape[-1]
        v_len = v_wav.shape[-1]
        
        target_len = max(b_len, v_len)
        
        # Prepare buffers
        final_b = torch.zeros((1, 2, target_len), device=b_wav.device)
        final_v = torch.zeros((1, 2, target_len), device=v_wav.device)
        
        # Place Backing
        if alignment == "loop_backing" and b_len < target_len:
            # Simple tile
            repeats = math.ceil(target_len / b_len)
            tiled = b_wav.repeat(1, 1, repeats)
            final_b = tiled[..., :target_len]
        else:
            final_b[..., :b_len] = b_wav

        # Place Vocals
        # Usually vocals start later or are shorter
        if alignment == "center":
            start = (target_len - v_len) // 2
            final_v[..., start:start+v_len] = v_wav
        else:
            final_v[..., :v_len] = v_wav

        # 4. Mix
        mixed = (final_b * backing_volume) + (final_v * vocal_volume)
        
        # 5. Soft Clip / Limit
        mixed = torch.tanh(mixed)
        
        return ({"waveform": mixed, "sample_rate": b_sr},)
        prompt_parts.append("high fidelity, stereo, masterpiece")
        
        final_prompt = ", ".join(prompt_parts)
        
        # Set parameters based on style
        cfg = 3.0
        temperature = 1.0
        
        if style in ["Techno", "Cyberpunk"]:
            cfg = 4.0 # More strict
        elif style in ["Ambient", "Lo-Fi"]:
             temperature = 1.2 # More variety
             
        return (final_prompt, bpm, duration, cfg, temperature)

class SonicSaver:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"audio": ("AUDIO", ),
                     "filename_prefix": ("STRING", {"default": "music/Sonic"}),
                     "format": (["wav", "mp3", "flac"], {"default": "wav"}),
                    }
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio"

    def save_audio(self, audio, filename_prefix="music/Sonic", format="wav"):
        if "waveform" not in audio:
            print("SonicSaver: Audio input missing 'waveform'.")
            return ()
            
        waveform_tensor = audio["waveform"]
        if waveform_tensor.is_cuda:
            waveform_tensor = waveform_tensor.cpu()
            
        batch_size = waveform_tensor.shape[0]
        channels = waveform_tensor.shape[1] if waveform_tensor.dim() > 1 else 1
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, channels, batch_size)
        results = list()
        
        # Handle sample_rate safely (int or Tensor)
        sample_rate = audio.get("sample_rate", 44100)
        if isinstance(sample_rate, torch.Tensor):
            sample_rate = int(sample_rate.item())
        if not isinstance(sample_rate, int):
            try:
                 sample_rate = int(sample_rate)
            except:
                 sample_rate = 44100
        
        for (batch_number, waveform) in enumerate(waveform_tensor):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{format}"
            
            # Ensure target directory exists
            target_dir = os.path.dirname(os.path.join(full_output_folder, file))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
                
            try:
                path = os.path.join(full_output_folder, file)
                
                # Check for MP3 specifically - Soundfile support for MP3 varies by install
                if format == "mp3":
                     # Try torchaudio first for MP3 (uses ffmpeg)
                     try:
                         torchaudio.save(path, waveform, sample_rate, format="mp3")
                     except Exception as exc:
                         print(f"SonicSaver: Torchaudio MP3 save failed ({exc}), trying soundfile...")
                         # Fallback to soundfile (requires newer libsndfile)
                         if waveform.dim() == 2:
                             wav_data = waveform.transpose(0, 1).numpy() 
                         else:
                             wav_data = waveform.numpy()
                         sf.write(path, wav_data, sample_rate, format="mp3")
                else:
                    # WAV and FLAC -> use SoundFile (Robust)
                    # waveform is [Channels, Length] -> transpose to [Length, Channels]
                    if waveform.dim() == 2:
                         wav_data = waveform.transpose(0, 1).numpy() 
                    else:
                         wav_data = waveform.numpy()

                    sf.write(path, wav_data, sample_rate, format=format)
            except Exception as e:
                print(f"SonicSaver Save Error: {e}")

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1

        return { "ui": { "audio": results } }

class SonicSpectrogram:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "n_fft": ("INT", {"default": 2048, "min": 64, "max": 8192}),
                "hop_length": ("INT", {"default": 512, "min": 32, "max": 2048}),
                "cmap": (["magma", "inferno", "plasma", "viridis", "cividis", "jet"], {"default": "magma"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_spectrogram"
    CATEGORY = "Geekatplay Studio"

    def generate_spectrogram(self, audio, n_fft, hop_length, cmap):
        waveform = audio["waveform"] # [batch, channels, samples]
        sample_rate = audio["sample_rate"]

        # Take the first item in batch and first channel for visualization
        y = waveform[0, 0].cpu().numpy()

        # Compute spectrogram using librosa directly for convenience
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.imshow(S_db, aspect='auto', origin='lower', cmap=cmap)
        plt.axis('off')
        plt.tight_layout(pad=0)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()

        # Convert to tensor
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return (img_tensor,)

class SonicWaveform:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "resample_factor": ("INT", {"default": 1, "min": 1, "max": 100}),
                "color": (["cyan", "magenta", "lime", "yellow", "white", "orange"], {"default": "cyan"}),
                "bg_color": (["black", "white", "none"], {"default": "black"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_waveform"
    CATEGORY = "Geekatplay Studio"

    def generate_waveform(self, audio, resample_factor, color, bg_color):
        waveform = audio["waveform"][0, 0].cpu().numpy()
        
        # Simple downsampling for display speed on large files
        if resample_factor > 1:
            waveform = waveform[::resample_factor]

        plt.figure(figsize=(10, 4))
        
        # Handle background
        if bg_color == "none":
            # Transparent not easily supported by simple RGB tensor conversion without alpha channel handling in ComfyUI preview sometimes
            # We'll use black for safe tensor RGB
            plt.style.use('dark_background')
            fig = plt.gcf()
            fig.patch.set_facecolor('black')
        elif bg_color == "black":
            plt.style.use('dark_background')
        else: # white
            plt.style.use('default')

        plt.plot(waveform, color=color, linewidth=0.8, alpha=0.8)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.margins(0)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()

        img = Image.open(buf).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return (img_tensor,)

class SonicFluxSynth:
    """
    High-Fidelity Audio Generator using TangoFlux (Flow Matching).
    """
    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic atmosphere, high fidelity, 4k audio, deep bass, synthesizer"}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "steps": ("INT", {"default": 25, "min": 10, "max": 100, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Geekatplay Studio"

    def generate(self, prompt, duration, steps, guidance_scale, seed):
        try:
            from tangoflux import TangoFluxInference
        except ImportError:
            raise ImportError("SonicFluxSynth: 'tangoflux' library not found. Please run 'pip install tangoflux diffusers transformers accelerate'.")

        if self.model is None:
            print("Loading TangoFlux (declare-lab/TangoFlux)...")
            # Loads from cache or downloads
            self.model = TangoFluxInference(name='declare-lab/TangoFlux')

        torch.manual_seed(seed)
        
        print(f"Generating Flux audio: '{prompt}' (Steps: {steps}, CFG: {guidance_scale})...")
        try:
            audio = self.model.generate(prompt, steps=steps, duration=duration, guidance_scale=guidance_scale)
        except Exception as e:
             # Fallback for API changes or issues
             print(f"Error in TangoFlux generation: {e}")
             raise e

        # TangoFlux returns usually [Channels, Length] tensor.
        # It operates at 44.1kHz.
        
        # Ensure format [1, Channels, Length] for ComfyUI audio dict
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0) # [1, 1, L]
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0) # [1, C, L]
            
        return ({"waveform": audio.cpu(), "sample_rate": 44100},)

NODE_CLASS_MAPPINGS = {
    "SonicHoloSynth": SonicHoloSynth,
    "SonicFluxSynth": SonicFluxSynth,
    "SonicMixer": SonicMixer,
    "SonicSinger": SonicSinger,
    "SonicTrackLayer": SonicTrackLayer,
    "SonicSaver": SonicSaver,
    "SonicSpectrogram": SonicSpectrogram,
    "SonicWaveform": SonicWaveform
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SonicHoloSynth": "Sonic-Holodeck 🎧",
    "SonicFluxSynth": "Sonic Flux (High Quality) 🌟",
    "SonicMixer": "Sonic DJ Mixer 🎛️",

    "SonicSinger": "Sonic Singer (Bark) 🎤",
    "SonicTrackLayer": "Sonic Trace Layer (Mixer) 🎚️",
    "SonicSaver": "Sonic Saver (Music) 💾"
}

