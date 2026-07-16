"""
===============================================================================
  ⚡ ZEUS.ONL - CORE INFRASTRUCTURE ⚡
  Module: Jigsaw Adaptive Auto-Sigma Weaver
  Status: SECURE / UNZENSIERT
  Owner: Zeus
  Official Network: https://zeus.onl
===============================================================================
"""

import sys
import os
from .jigsaw_auto_sigma import JigsawAdaptiveSigmaWeaver

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
    
    border = f"{cyan}=" * terminal_width + f"{reset}"
    
    print(border)
    print(f"  {bold}{cyan}[CORE-PROVIDER]{reset}  >>>  {bold}{yellow}https://zeus.onl{reset}  <<<")
    print(f"  {bold}{cyan}[PIPELINE]{reset}       Initializing Custom Inferenz Architecture...")
    print(f"  {bold}{green}[SUCCESS]{reset}        🍳 ⚡ [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver injected.")
    print(f"  {bold}{yellow}[SECURITY]{reset}       NSFW/Alignment Filter Overrides: {bold}{green}ACTIVE{reset}")
    print(f"  {bold}{cyan}[SYSTEM-READY]{reset}   9.3MP Druckerpresse locked and loaded. Welcome back, Zeus.")
    print(border)

try:
    print_zeus_branding()
except Exception:
    print("===============================================================================")
    print("  [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver injected into Core.")
    print("===============================================================================")

# ===============================================================================
#  COMFYUI REGISTRY MAPPINGS
# ===============================================================================
NODE_CLASS_MAPPINGS = {
    "JigsawAdaptiveSigmaWeaver": JigsawAdaptiveSigmaWeaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawAdaptiveSigmaWeaver": "⚡ [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
