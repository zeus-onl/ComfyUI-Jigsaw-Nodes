"""
===============================================================================
  ⚡ ZEUS.ONL - VAEMERGE CORE ⚡
  Function: Universal VAE Merging Engine (v2 - Layer-Agnostic Safe Core)
  Author: Jigsaw & Zeus
  Official Network: https://zeus.onl
===============================================================================
"""

import torch
from torch import nn
import comfy.sd
from tqdm import tqdm

class UniversalVAEMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_a": ("VAE",),
                "vae_b": ("VAE",),
                "merge_mode": (["weighted_sum", "weighted_subtraction", "tensor_sum", "add_difference"],),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "vae_c": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "merge_vae"
    CATEGORY = "🍳 Jigsaw/VAE"

    def merge_vae(self, vae_a, vae_b, merge_mode, alpha, beta, brightness, contrast, vae_c=None):
        # 1. State Dicts unzensiert extrahieren
        sd_a = vae_a.first_stage_model.state_dict()
        sd_b = vae_b.first_stage_model.state_dict()
        sd_c = vae_c.first_stage_model.state_dict() if vae_c is not None else None

        # 2. Schnittmenge der Keys berechnen (Läuft mit JEDEM Modell-Typ!)
        ckpt_keys = sd_a.keys() & sd_b.keys()
        if sd_c is not None:
            ckpt_keys = ckpt_keys & sd_c.keys()

        print(f"[ZEUS.ONL VAE-MERGE] Fusioniere {len(ckpt_keys)} Keys im Layer-Agnostic Modus...")
        merged_sd = {}

        # 3. Mathematische Fusion im CPU-Raum abwickeln
        for key in tqdm(ckpt_keys):
            a = sd_a[key].detach().cpu().float()
            b = sd_b[key].detach().cpu().float()
            c = sd_c[key].detach().cpu().float() if sd_c is not None else None

            if merge_mode == "weighted_sum":
                merged_sd[key] = a * (1.0 - alpha) + b * alpha
            elif merge_mode == "weighted_subtraction":
                merged_sd[key] = a - (b * alpha)
            elif merge_mode == "tensor_sum":
                merged_sd[key] = a + b * alpha
            elif merge_mode == "add_difference":
                if c is not None:
                    merged_sd[key] = a + (b - c) * alpha
                else:
                    merged_sd[key] = a

            # In das Original-Format zurückgießen
            merged_sd[key] = merged_sd[key].to(dtype=sd_a[key].dtype, device=sd_a[key].device)

        # 4. Helligkeit & Kontrast-Injektion (Kugelsicher absichern)
        bias_key = "decoder.conv_out.bias"
        weight_key = "decoder.conv_out.weight"
        
        # Fallback falls WAN die Layer anders benannt hat (z.B. final_layer)
        for k in merged_sd.keys():
            if "conv_out.bias" in k or "final.bias" in k:
                bias_key = k
            if "conv_out.weight" in k or "final.weight" in k:
                weight_key = k

        if bias_key in merged_sd:
            merged_sd[bias_key] = nn.Parameter(merged_sd[bias_key] + brightness)
        if weight_key in merged_sd:
            merged_sd[weight_key] = nn.Parameter(merged_sd[weight_key] + contrast / 40.0)

        comfy_vae = comfy.sd.VAE(merged_sd)
        return (comfy_vae,)

NODE_CLASS_MAPPINGS = {"UniversalVAEMerge": UniversalVAEMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalVAEMerge": "📐 ⚡ [ZEUS.ONL] Universal VAE Merge v2"}
