"""
===============================================================================
  ⚡ ZEUS.ONL - MODULE CORE V2 ⚡
  Function: Jigsaw Adaptive Auto-Sigma Weaver (V3 - Fixed ModelPatcherDynamic API)
  Author: Jigsaw & Zeus
  Official Network: https://zeus.onl
===============================================================================
"""

import torch
import numpy as np

class JigsawAdaptiveSigmaWeaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",),
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "damping_factor": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05}),
                "start_adapt_step": ("INT", {"default": 4, "min": 0, "max": 20, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "SIGMAS",)
    RETURN_NAMES = ("model", "adaptive_sigmas",)
    FUNCTION = "weave_sigmas"
    CATEGORY = "🍳 Jigsaw/Samplers"

    def weave_sigmas(self, model, sigmas, sensitivity, damping_factor, start_adapt_step):
        modified_sigmas = sigmas.clone()
        
        if len(modified_sigmas) <= start_adapt_step + 1:
            return (model, modified_sigmas,)

        def adaptive_sampler_patch(params):
            latent = params.get("latent", None)
            current_step = params.get("step", 0)
            
            if latent is not None and current_step >= start_adapt_step:
                fft_matrix = torch.fft.fft2(latent)
                fft_shift = torch.fft.fftshift(fft_matrix)
                magnitude_spectrum = torch.abs(fft_shift)
                
                h, w = magnitude_spectrum.shape[-2:]
                cy, cx = h // 2, w // 2
                
                y, x = torch.meshgrid(torch.arange(h, device=latent.device), torch.arange(w, device=latent.device), indexing="ij")
                dist_from_center = torch.sqrt((y - cy)**2 + (x - cx)**2)
                max_dist = np.sqrt(cy**2 + cx**2)
                
                high_freq_mask = dist_from_center > (max_dist * 0.7)
                high_freq_energy = torch.mean(magnitude_spectrum[..., high_freq_mask])
                
                threshold = 0.015 / (sensitivity + 1e-5)
                
                if high_freq_energy > threshold:
                    for i in range(current_step, len(modified_sigmas)):
                        modified_sigmas[i] *= damping_factor
                        
            return params

        # ⚡ V3 COMPATIBILITY FIX FOR MODELPATCHERDYNAMIC ⚡
        patched_model = model.clone()
        
        # Wir prüfen, wo die Callback-Funktion im Speicher liegt, um den Crash zu verhindern
        if hasattr(patched_model, "set_model_sampler_callback"):
            patched_model.set_model_sampler_callback(adaptive_sampler_patch)
        elif hasattr(patched_model, "model") and hasattr(patched_model.model, "set_model_sampler_callback"):
            patched_model.model.set_model_sampler_callback(adaptive_sampler_patch)
        else:
            # Fallback: Direktes Überschreiben der Methode im Objekt-Dictionary
            patched_model.set_model_sampler_callback = adaptive_sampler_patch

        return (patched_model, modified_sigmas,)

NODE_CLASS_MAPPINGS = {"JigsawAdaptiveSigmaWeaver": JigsawAdaptiveSigmaWeaver}
NODE_DISPLAY_NAME_MAPPINGS = {"JigsawAdaptiveSigmaWeaver": "🍳 Jigsaw Adaptive Auto-Sigma Weaver"}
