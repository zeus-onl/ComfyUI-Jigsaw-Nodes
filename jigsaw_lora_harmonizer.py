"""
===============================================================================
  ⚡ ZEUS.ONL - MODULE CORE ⚡
  Function: Jigsaw LoRA Conflict Harmonizer (v3 - Cached Plan Edition)
  Author: Jigsaw & Zeus
  Official Network: https://zeus.onl
===============================================================================
"""

import os
import json
import hashlib
import copy

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harmonizer_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# RAM-Cache: überlebt nur die laufende ComfyUI-Session (Prozess-Neustart löscht ihn)
_SESSION_CACHE = {}


def _extract_lora_signature(patched_model):
    """
    Baut eine deterministische, sortierte Signatur aus allen aktuell geladenen
    LoRA-Patches. Wir nutzen NUR Metadaten (Key + Adapter-Typ-Name + Strength),
    keine Tensor-Inhalte -> bleibt winzig und schnell hashbar.
    """
    sig_parts = []
    for key, entries in sorted(patched_model.patches.items()):
        for entry in entries:
            strength = entry[0] if len(entry) > 0 else None
            adapter_type = type(entry[1]).__name__ if len(entry) > 1 else "unknown"
            sig_parts.append(f"{key}|{adapter_type}|{strength}")
    return "\n".join(sig_parts)


def _fingerprint(patched_model, mode):
    sig = _extract_lora_signature(patched_model)
    payload = f"{mode}::{sig}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_disk_cache(fingerprint):
    path = os.path.join(CACHE_DIR, f"{fingerprint}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_disk_cache(fingerprint, plan):
    path = os.path.join(CACHE_DIR, f"{fingerprint}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f)
    except Exception as e:
        print(f"[LoraConflictHarmonizer] Cache konnte nicht geschrieben werden: {e}")


class JigsawLoraConflictHarmonizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["scale_by_count", "priority_first", "priority_last"],),
                "cache_mode": (["ram_and_disk", "ram_only", "disk_only", "off"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "harmonize"
    CATEGORY = "🍳 Jigsaw/LoRA"

    def harmonize(self, model, mode, cache_mode):
        patched = model.clone()
        patches = copy.deepcopy(patched.patches)

        fingerprint = _fingerprint(patched, mode)
        plan = None

        # --- Cache-Lookup ---
        if cache_mode in ("ram_and_disk", "ram_only"):
            plan = _SESSION_CACHE.get(fingerprint)
            if plan is not None:
                print(f"[Harmonizer] RAM-Cache-Treffer ({fingerprint[:8]}...)")

        if plan is None and cache_mode in ("ram_and_disk", "disk_only"):
            plan = _load_disk_cache(fingerprint)
            if plan is not None:
                print(f"[Harmonizer] Disk-Cache-Treffer ({fingerprint[:8]}...)")

        # --- Neu berechnen, falls kein Treffer ---
        if plan is None:
            plan = self._compute_plan(patches, mode)
            print(f"[Harmonizer] Neu berechnet: {len(plan)} Konflikt-Keys ({fingerprint[:8]}...)")

            if cache_mode in ("ram_and_disk", "ram_only"):
                _SESSION_CACHE[fingerprint] = plan
            if cache_mode in ("ram_and_disk", "disk_only"):
                _save_disk_cache(fingerprint, plan)

        # --- Plan anwenden ---
        for key, instruction in plan.items():
            if key not in patches:
                continue  # Key existiert in dieser Model-Instanz nicht (sollte bei Fingerprint-Match nicht passieren)
            entries = patches[key]

            if instruction["action"] == "scale":
                new_entries = []
                for i, entry in enumerate(entries):
                    factor = instruction["factors"][i]
                    new_entries.append((entry[0] * factor,) + entry[1:])
                patches[key] = new_entries

            elif instruction["action"] == "keep_index":
                idx = instruction["index"]
                patches[key] = [entries[idx]]

        patched.patches = patches
        return (patched,)

    def _compute_plan(self, patches, mode):
        """
        Reine Planungs-Logik: erzeugt NUR Metadaten (welcher Key, welche Aktion,
        welche Faktoren/Indizes) - keine Tensoren, keine Adapter-spezifische
        Delta-Berechnung. Das hält die Logik unabhängig vom Adapter-Typ
        (LoRAAdapter, LoKrAdapter, GLoRAAdapter, ...) und damit robust.
        """
        plan = {}
        for key, entries in patches.items():
            if len(entries) <= 1:
                continue

            if mode == "scale_by_count":
                n = len(entries)
                plan[key] = {"action": "scale", "factors": [1.0 / n] * n}

            elif mode == "priority_first":
                plan[key] = {"action": "keep_index", "index": 0}

            elif mode == "priority_last":
                plan[key] = {"action": "keep_index", "index": len(entries) - 1}

        return plan


NODE_CLASS_MAPPINGS = {"JigsawLoraConflictHarmonizer": JigsawLoraConflictHarmonizer}
NODE_DISPLAY_NAME_MAPPINGS = {"JigsawLoraConflictHarmonizer": "🍳 [ZEUS.ONL] Jigsaw LoRA Conflict Harmonizer v3"}