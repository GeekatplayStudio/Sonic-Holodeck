import torch
import torchaudio
import os
import folder_paths
import math
import numpy as np

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
                "prompt": ("STRING", {"multiline": True, "default": "Cyberpunk city rain, neon lights, synthwave, slow tempo"}),
                "bpm": ("INT", {"default": 120, "min": 60, "max": 200, "step": 1}),
                "duration": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "bpm_out")
    FUNCTION = "generate_music"
    CATEGORY = "Geekatplay Studio"

    def apply_auto_mastering(self, wav, sample_rate):
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
            
        # 1. Simple Compression (Simulated using tanh for soft clipping/saturation)
        # Real compression requires stateful processing or complex envelope following. 
        # A soft clipper is a good approximation for "loudening" in a simple script.
        
        # Drive input slightly 
        pre_gain = 1.5
        compressed = torch.tanh(wav * pre_gain)
        
        # 2. Normalize to Target Level
        # Target RMS for -14dBFS is approx 10^(-14/20) = 0.2
        target_rms = 0.2
        
        current_rms = torch.sqrt(torch.mean(compressed**2))
        
        if current_rms > 0:
            gain_adjust = target_rms / current_rms
            mastered = compressed * gain_adjust
        else:
            mastered = compressed

        # 3. Final Safety Limiter (Hard clip at -0.1dB)
        ceiling = 0.99
        mastered = torch.clamp(mastered, -ceiling, ceiling)
        
        # Reshape back to [1, C, L] for ComfyUI if needed, but Comfy usually takes [C, L] or [1, C, L]
        return mastered.unsqueeze(0)

    def generate_music(self, prompt, bpm, duration, cfg, temperature, seed):
        try:
            from audiocraft.models import MusicGen
        except ImportError:
            raise ImportError("Please install audiocraft.")

        if self.model is None:
            print("Loading MusicGen Stereo Large model...")
            self.model = MusicGen.get_pretrained('facebook/musicgen-stereo-large')

        torch.manual_seed(seed)
        
        self.model.set_generation_params(
            duration=duration, 
            top_k=250, 
            top_p=0.0, 
            temperature=temperature, 
            cfg_coef=cfg
        )

        print(f"Generating audio: '{prompt}' (BPM: {bpm})...")
        # Wav is [batch, channels, length]
        wav = self.model.generate([prompt], progress=True)

        # Auto-Mastering
        print("Applying Auto-Mastering...")
        mastered_wav = self.apply_auto_mastering(wav.cpu(), self.model.sample_rate)

        audio_output = {
            "waveform": mastered_wav, 
            "sample_rate": self.model.sample_rate
        }

        return (audio_output, float(bpm))

# Registration
NODE_CLASS_MAPPINGS = {
    "SonicHoloSynth": SonicHoloSynth
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SonicHoloSynth": "Sonic-Holodeck 🎧",
    "SonicMixer": "Sonic DJ Mixer 🎛️",
    "SonicSinger": "Sonic Singer (Bark) 🎤"
}

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
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": "In the neon rain...", "placeholder": "Enter Lyrics here"}),
                "style": (["Cyberpunk", "Synthwave", "Techno", "Lo-Fi", "Cinematic", "Orchestral", "Rock", "Hip Hop", "Ambient"],),
                "instruments": ("STRING", {"default": "Synthesizer, Drum Machine, Bass", "multiline": False}),
                "vocoder_fx": (["None", "Robotic", "Ethereal", "Distorted"],),
                "bpm": ("INT", {"default": 120, "min": 60, "max": 200}),
                "duration": ("INT", {"default": 15, "min": 5, "max": 30}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("constructed_prompt", "bpm", "duration", "cfg", "temperature")
    FUNCTION = "mix_track"
    CATEGORY = "Geekatplay Studio"

    def mix_track(self, lyrics, style, instruments, vocoder_fx, bpm, duration):
        # Construct a rich prompt for MusicGen
        prompt_parts = []
        
        # Style base
        prompt_parts.append(f"{style} track")
        
        # Instruments
        if instruments:
            prompt_parts.append(f"featuring {instruments}")
            
        # Vocals/Lyrics intention
        # Note: MusicGen is instrumental-primary, but adding vocal descriptors helps structure
        if lyrics.strip():
            prompt_parts.append("with vocals, singing")
            if vocoder_fx != "None":
                prompt_parts.append(f"{vocoder_fx} vocal effects")
        
        # BPM context
        prompt_parts.append(f"{bpm} bpm")
        
        # High fidelity keywords
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

NODE_CLASS_MAPPINGS = {
    "SonicHoloSynth": SonicHoloSynth,
    "SonicMixer": SonicMixer
}
