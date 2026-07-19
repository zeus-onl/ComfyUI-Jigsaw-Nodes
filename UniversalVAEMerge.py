"""
===============================================================================
  ⚡ ZEUS.ONL - VAEMERGE CORE ⚡
  Function: Universal VAE Merge (v6 - Fully Switchable Block-Chirurgie Core)
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
                # 🪐 WEICHE 1: Bestimmt, wie viele VAEs geladen/verarbeitet werden
                "number_of_vaes": (["1", "2", "3"], {"default": "2"}),
                "merge_mode": (["weighted_sum", "weighted_subtraction", "tensor_sum", "add_difference", "ties_add_difference"], {"default": "weighted_sum"}),
                "alpha": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "beta": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 0.00, "min": -1.0, "max": 1.0, "step": 0.01}),
                # 🪐 WEICHE 2: Schaltet das granulare Block-Merge-System ein/aus
                "use_blocks": ("BOOLEAN", {"default": False}),
                "block_conv_out": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_norm_out": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_0": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_1": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_2": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_3": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_mid": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_conv_in": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "block_quant_conv": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "vae_b": ("VAE",),
                "vae_c": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "merge_vae"
    CATEGORY = "🍳 Jigsaw/VAE"

    def _get_block_weight(self, key, block_weights):
        # Ermittelt das spezifische Gewicht basierend auf dem Key-Namen (Für SDXL/ZIT kompatible VAEs)
        if "conv_out" in key: return block_weights["block_conv_out"]
        if "norm_out" in key: return block_weights["block_norm_out"]
        if "mid" in key: return block_weights["block_mid"]
        if "conv_in" in key: return block_weights["block_conv_in"]
        if "quant_conv" in key: return block_weights["block_quant_conv"]
        if "down.0" in key or "up.3" in key: return block_weights["block_0"]
        if "down.1" in key or "up.2" in key: return block_weights["block_1"]
        if "down.2" in key or "up.1" in key: return block_weights["block_2"]
        if "down.3" in key or "up.0" in key: return block_weights["block_3"]
        return None

    def merge_vae(self, vae_a, number_of_vaes, merge_mode, alpha, beta, brightness, contrast, use_blocks,
                  block_conv_out, block_norm_out, block_0, block_1, block_2, block_3, block_mid, block_conv_in, block_quant_conv,
                  vae_b=None, vae_c=None):
        
        sd_a = vae_a.first_stage_model.state_dict()
        
        # Dictionary für die Block-Gewichte zusammenstellen
        b_weights = {
            "block_conv_out": block_conv_out, "block_norm_out": block_norm_out,
            "block_0": block_0, "block_1": block_1, "block_2": block_2, "block_3": block_3,
            "block_mid": block_mid, "block_conv_in": block_conv_in, "block_quant_conv": block_quant_conv
        }

        # ===============================================================================
        #  ⚡ MODUS 1: REINES SINGLE-VAE TUNING (1 VAE)
        # ===============================================================================
        if number_of_vaes == "1":
            print("[ZEUS.ONL VAE-MERGE] Modus 1 aktiv: Passe nur Helligkeit/Kontrast für VAE_A an...")
            merged_sd = {k: v.clone() for k, v in sd_a.items()}
            
        # ===============================================================================
        #  ⚡ MODUS 2 & 3: MULTI-VAE FUSION (2 ODER 3 VAEs)
        # ===============================================================================
        else:
            sd_b = vae_b.first_stage_model.state_dict() if vae_b is not None else None
            sd_c = vae_c.first_stage_model.state_dict() if vae_c is not None else None

            if number_of_vaes == "2" and sd_b is None:
                raise ValueError("💥 [ZEUS.ONL ERROR] Du hast '2 VAEs' ausgewählt, aber am Port vae_b steckt kein Kabel!")
            if number_of_vaes == "3" and (sd_b is None or sd_c is None):
                raise ValueError("💥 [ZEUS.ONL ERROR] Du hast '3 VAEs' ausgewählt, aber an vae_b oder vae_c fehlt ein Kabel!")

            # Schnittmenge berechnen
            ckpt_keys = sd_a.keys() & sd_b.keys()
            if number_of_vaes == "3":
                ckpt_keys = ckpt_keys & sd_c.keys()

            print(f"[ZEUS.ONL VAE-MERGE] Fusioniere {len(ckpt_keys)} Keys...")
            merged_sd = {}

            for key in tqdm(ckpt_keys):
                a = sd_a[key].detach().cpu().float()
                b = sd_b[key].detach().cpu().float()
                c = sd_c[key].detach().cpu().float() if sd_c is not None else None

                # 🪐 BLOCK-CHIRURGIE LOGIK INTERN 🪐
                current_alpha = alpha
                current_beta = beta
                
                if use_blocks:
                    w = self._get_block_weight(key, b_weights)
                    if w is not None:
                        current_alpha = w
                        current_beta = w

                # Mathematische Fusionsmodi ausführen
                if number_of_vaes == "2":
                    if merge_mode == "weighted_sum":
                        merged_sd[key] = a * (1.0 - current_alpha) + b * current_alpha
                    elif merge_mode == "weighted_subtraction":
                        merged_sd[key] = a - (b * current_alpha)
                    elif merge_mode == "tensor_sum":
                        merged_sd[key] = a + b * current_alpha
                    elif merge_mode in ["add_difference", "ties_add_difference"]:
                        merged_sd[key] = a  # Benötigt 3 VAEs, Fallback auf A
                
                elif number_of_vaes == "3":
                    if merge_mode == "weighted_sum":
                        merged_sd[key] = a * (1.0 - current_alpha - current_beta) + b * current_alpha + c * current_beta
                    elif merge_mode == "weighted_subtraction":
                        merged_sd[key] = a - (b * current_alpha) - (c * current_beta)
                    elif merge_mode == "tensor_sum":
                        merged_sd[key] = a + b * current_alpha + c * current_beta
                    elif merge_mode in ["add_difference", "ties_add_difference"]:
                        merged_sd[key] = a + (b - c) * current_alpha

                merged_sd[key] = merged_sd[key].to(dtype=sd_a[key].dtype, device=sd_a[key].device)

        # ===============================================================================
        #  🎨 GLOBALER KONTRAST & BRIGHTNESS INJEKTOR (Kugelsicher für alle VAEs)
        # ===============================================================================
        bias_key = None
        weight_key = None
        for k in merged_sd.keys():
            if "conv_out.bias" in k or "final.bias" in k or "final_layer.bias" in k:
                bias_key = k
            if "conv_out.weight" in k or "final.weight" in k or "final_layer.weight" in k:
                weight_key = k

        if bias_key is not None:
            merged_sd[bias_key] = nn.Parameter(merged_sd[bias_key] + brightness)
        if weight_key is not None:
            merged_sd[weight_key] = nn.Parameter(merged_sd[weight_key] + contrast / 40.0)

        comfy_vae = comfy.sd.VAE(merged_sd)
        return (comfy_vae,)

NODE_CLASS_MAPPINGS = {"UniversalVAEMerge": UniversalVAEMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"UniversalVAEMerge": "🔮 ⚡ [ZEUS.ONL] Universal VAE Merge v6"}
