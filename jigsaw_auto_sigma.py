"""
===============================================================================
  ⚡ ZEUS.ONL - MODULE CORE ⚡
  Function: Jigsaw Pure Auto-Sigma Weaver (v12 - True Auto-Calculated Sigmas)
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
                "sigmas": ("SIGMAS",),  # Empfängt die Standard-Sigmas vom Sampler
                "auto_smoothness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "SIGMAS",)
    RETURN_NAMES = ("model", "adaptive_sigmas",)
    FUNCTION = "weave_sigmas"
    CATEGORY = "🍳 Jigsaw/Samplers"

    def weave_sigmas(self, model, sigmas, auto_smoothness):
        num_steps = len(sigmas) - 1
        if num_steps < 1:
            return (model, sigmas)

        device = sigmas.device
        dtype = sigmas.dtype

        # ⚡ AUTOMATISCHE INTELLIGENZ: Errechnet den perfekten Flow-Matching Shift (Mu)
        # Wenn der Nutzer wenig Schritte nutzt (z.B. 8 für Turbo), brauchen wir einen starken Shift (1.25).
        # Wenn er viele Schritte nutzt (z.B. 28 für RAW), reicht ein sanfterer Shift (1.15).
        if num_steps <= 10:
            base_mu = 1.25
        else:
            base_mu = 1.15

        # Multipliziere den Shift mit dem Smoothness-Regler (Standard 1.0 ist perfekt berechnet)
        mu = base_mu * auto_smoothness

        # Mathematische Generierung der idealen, geshifteten Krea-2 Zeitkurve
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
        
        # Die originale Flow-Matching Interpolations-Formel
        t_shifted = (mu * t) / (1.0 + (mu - 1.0) * t)
        t_shifted[-1] = 0.0  # Garantiert, dass der letzte Schritt absolut sauber bei 0 endet
        
        adaptive_sigmas = t_shifted.to(dtype=dtype)

        # Sauberer Output ohne Wrapper-Abstürze oder Mosaik-Fehler
        return (model, adaptive_sigmas)

NODE_CLASS_MAPPINGS = {"JigsawAdaptiveSigmaWeaver": JigsawAdaptiveSigmaWeaver}
NODE_DISPLAY_NAME_MAPPINGS = {"JigsawAdaptiveSigmaWeaver": "🍳 [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver"}
