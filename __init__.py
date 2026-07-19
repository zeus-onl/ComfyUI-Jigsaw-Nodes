"""
===============================================================================
  ⚡ ZEUS.ONL - CORE INFRASTRUCTURE ⚡
  Module: Jigsaw Core Nodes (Weaver, Accelerator, Harmonizer & Universal VAE Merge)
  Status: SECURE / UNZENSIERT
  Owner: Zeus
  Official Network: https://zeus.onl
===============================================================================
"""

import sys
import os
from .jigsaw_auto_sigma import JigsawAdaptiveSigmaWeaver
from .jigsaw_tcd_accelerator import JigsawTCDVectorAccelerator
from .jigsaw_lora_harmonizer import JigsawLoraConflictHarmonizer
from .UniversalVAEMerge import UniversalVAEMerge

# ===============================================================================
#  ⚡ MODERN & EYE-CATCHING TERMINAL LOGGING
# ===============================================================================
def print_zeus_branding():
    terminal_width = 80
    cyan = "\033[96m"
    yellow = "\033[93m"
    bold = "\033[1m"
    red = "\033[91m"
    green = "\033[92m"
    reset = "\033[0m"
    
    border = f"{cyan}" + "=" * terminal_width + f"{reset}"
    
    print(border)
    print(f"  {bold}{cyan}[CORE-PROVIDER]{reset}  >>>  {bold}{yellow}https://zeus.onl{reset}  <<<")
    print(f"  {bold}{cyan}[PIPELINE]{reset}       Initializing Custom Inferenz Architecture...")
    print(f"  {bold}{green}[SUCCESS]{reset}        🍳 ⚡ [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver injected.")
    print(f"  {bold}{green}[SUCCESS]{reset}        🏎️ ⚡ [ZEUS.ONL] Jigsaw TCD Vector Accelerator injected.")
    print(f"  {bold}{green}[SUCCESS]{reset}        📐 ⚡ [ZEUS.ONL] Jigsaw LoRA Conflict Harmonizer injected.")
    print(f"  {bold}{green}[SUCCESS]{reset}        🔮 ⚡ [ZEUS.ONL] Universal VAE Merge injected.")
    print(f"  {bold}{yellow}[SECURITY]{reset}       NSFW/Alignment Filter Overrides: {bold}{green}ACTIVE{reset}")
    print(f"  {bold}{cyan}[SYSTEM-READY]{reset}   9.3MP Druckerpresse locked and loaded. Welcome back, Zeus.")
    print(border)

try:
    print_zeus_branding()
except Exception:
    print("===============================================================================")
    print("  [ZEUS.ONL] Jigsaw Core Nodes successfully injected.")
    print("===============================================================================")

# ===============================================================================
#  COMFYUI REGISTRY MAPPINGS
# ===============================================================================
NODE_CLASS_MAPPINGS = {
    "JigsawAdaptiveSigmaWeaver": JigsawAdaptiveSigmaWeaver,
    "JigsawTCDVectorAccelerator": JigsawTCDVectorAccelerator,
    "JigsawLoraConflictHarmonizer": JigsawLoraConflictHarmonizer,
    "UniversalVAEMerge": UniversalVAEMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawAdaptiveSigmaWeaver": "⚡ [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver",
    "JigsawTCDVectorAccelerator": "⚡ [ZEUS.ONL] Jigsaw TCD Vector Accelerator",
    "JigsawLoraConflictHarmonizer": "⚡ [ZEUS.ONL] Jigsaw LoRA Conflict Harmonizer",
    "UniversalVAEMerge": "🔮 ⚡ [ZEUS.ONL] Universal VAE Merge v2"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

