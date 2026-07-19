"""
===============================================================================
  ⚡ ZEUS.ONL - MODULE CORE ⚡
  Function: Jigsaw Pure Turbo Model Engine (v56 - Real Flow Trajectory Core)
  Author: Jigsaw & Zeus
  Official Network: https://zeus.onl
===============================================================================
"""

import torch

class JigsawTCDVectorAccelerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "turbo_power": ("FLOAT", {"default": 1.25, "min": 0.5, "max": 2.5, "step": 0.05}),
                "gamma_curve": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "dynamic_turbo"
    CATEGORY = "🍳 Jigsaw/Samplers"

    def dynamic_turbo(self, model, turbo_power, gamma_curve):
        patched_model = model.clone()

        def turbo_lora_gradient_wrapper(model_function, params):
            if "input" in params and isinstance(params["input"], torch.Tensor):
                if not params["input"].is_contiguous():
                    params["input"] = params["input"].contiguous()
                
                # Datentyp-Absicherung für Krea2 RAW
                if "timestep" in params and isinstance(params["timestep"], torch.Tensor):
                    params["timestep"] = params["timestep"].to(device=params["input"].device, dtype=params["input"].dtype)

            # 🪐 1. DER ORIGINALE INFERENZ-SCHRITT
            out = model_function(params["input"], params["timestep"], **params.get("c", {}))

            # 🪐 2. DYNAMISCHES DIERS-FLOW-SCALING MIT CLIPPING (Die Turbo-LoRA-Simulation)
            if isinstance(out, torch.Tensor) and "timestep" in params and isinstance(params["timestep"], torch.Tensor):
                t = params["timestep"].view(-1, 1, 1, 1)
                
                # Exponentielle Kurve: Frühe Schritte kriegen brachialen Schub. 
                # Späte Schritte werden sanft ausgefadet, um Artefakte zu verhindern.
                # Wir haben den Standard-Power-Wert auf 1.25 gesenkt und die Kurve leicht angepasst.
                dynamic_scale = 1.0 + (turbo_power - 1.0) * torch.pow(t / 1000.0 if t.max() > 1.0 else t, gamma_curve)
                
                # Vektor skalieren!
                out = out * dynamic_scale
                
                # 🪐 3. CLIPPING (Der Brandlöscher): Wir zwingen die Werte in einen sicheren Bereich,
                # um das Überbrennen (Clipping wie in KoO_Krea2_TXT2IMG_NoiseLab_00003_.jpg) zu verhindern.
                out = torch.clamp(out, min=-12.0, max=12.0)

            return out
        
        patched_model.model_options["model_function_wrapper"] = turbo_lora_gradient_wrapper

        return (patched_model,)

NODE_CLASS_MAPPINGS = {"JigsawTCDVectorAccelerator": JigsawTCDVectorAccelerator}
NODE_DISPLAY_NAME_MAPPINGS = {"JigsawTCDVectorAccelerator": "⚡ [ZEUS.ONL] Jigsaw TCD Vector Accelerator v56"}