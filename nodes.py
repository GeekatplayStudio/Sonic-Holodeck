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
    "SonicHoloSynth": "Sonic-Holodeck 🎧"
}
