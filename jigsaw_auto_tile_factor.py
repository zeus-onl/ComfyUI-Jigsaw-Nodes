"""
===============================================================================
  ⚡ ZEUS.ONL - CORE INFRASTRUCTURE ⚡
  Module: Jigsaw Auto Tile Factor
  Owner: Zeus
  Official Network: https://zeus.onl
===============================================================================

Berechnet width_factor / height_factor fuer TTP_Tile_image_size automatisch.

WICHTIG v2 FIX: Die reale Formel von TTP_Tile_image_size (Tile_imageSize in
comfyui_ttp_toolset) ist:

    tile_dim = int(canvas_dim / (factor * (1 - overlap_rate)))
    tile_dim = (tile_dim // 8) * 8   # auf Vielfaches von 8 abgerundet

Und TTP_Image_Tile_Batch berechnet daraus selbststaendig die Tile-Anzahl
und den Overlap:

    num_tiles = ceil(canvas_dim / tile_dim)
    overlap   = (num_tiles * tile_dim - canvas_dim) // (num_tiles - 1)

PROBLEM: Wenn canvas_dim exakt durch tile_dim teilbar ist, kommt overlap=0
raus. TTP_Image_Assy blendet dann GAR NICHT (blend_tiles() wird nur bei
overlap>0 aufgerufen) - es gibt einen harten Schnitt statt eines weichen
Uebergangs. Da jedes Tile unabhaengig durch SeedVR2 laeuft (leicht andere
Belichtung/Rauschen pro Tile), wird dieser harte Schnitt als sichtbare Naht
sichtbar (genau das beobachtete Linienmuster).

FIX: Dieser Node simuliert jetzt BEIDE echten TTP-Formeln (Tile-Groesse UND
Tile-Anzahl/Overlap) und erhoeht den Faktor automatisch so lange, bis ein
garantierter overlap > 0 fuer beide Achsen rauskommt - damit blend_tiles()
IMMER greift und keine harten Naehte mehr entstehen.

Verwendung:
    image (Resize Image v2 / ImageResizeKJv2 Output, gleiche Quelle wie
           TTP_Tile_image_size.image)
        -> Jigsaw Auto Tile Factor
            -> width_factor, height_factor, overlap_rate (Pass-Through)
                -> alle drei in TTP_Tile_image_size einspeisen
"""

import math

MAX_TILE_SIZE_DEFAULT = 896
MAX_FACTOR_SEARCH = 32  # Obergrenze, falls trotz Erhoehung nix passt


class _AnyType(str):
    """Wildcard-Typ fuer ComfyUI Input-Sockets."""

    def __ne__(self, other):
        return False

    def __eq__(self, other):
        return True


ANY_TYPE = _AnyType("*")


def _ttp_real_tile_size(canvas_dim, factor, overlap_rate):
    """Repliziert Tile_imageSize.image_width_height() 1:1."""
    tile_dim = int(canvas_dim / (factor * (1 - overlap_rate)))
    if tile_dim % 8 != 0:
        tile_dim = (tile_dim // 8) * 8
    return max(8, tile_dim)


def _ttp_real_overlap(canvas_dim, tile_dim):
    """Repliziert TTP_Image_Tile_Batch.calculate_step() 1:1."""
    if canvas_dim <= tile_dim:
        return 1, 0  # num_tiles, overlap
    num_tiles = (canvas_dim + tile_dim - 1) // tile_dim
    if num_tiles <= 1:
        return num_tiles, 0
    overlap = (num_tiles * tile_dim - canvas_dim) // (num_tiles - 1)
    return num_tiles, overlap


def _find_safe_factor(canvas_dim, overlap_rate, min_factor, max_tile_size):
    """
    Sucht den kleinsten Faktor ab min_factor, bei dem die ECHTE TTP-Formel
    einen Tile ergibt, der (a) unter max_tile_size bleibt UND (b) beim
    Batch-Tiling einen Overlap > 0 erzeugt (garantiertes Blending, keine
    harten Naehte).
    """
    factor = max(1, int(min_factor))
    for _ in range(MAX_FACTOR_SEARCH):
        tile_dim = _ttp_real_tile_size(canvas_dim, factor, overlap_rate)
        num_tiles, overlap = _ttp_real_overlap(canvas_dim, tile_dim)
        if tile_dim <= max_tile_size and (num_tiles <= 1 or overlap > 0):
            return factor, tile_dim, overlap
        factor += 1
    # Fallback: letzten Stand zurueckgeben, auch wenn nicht perfekt
    return factor, tile_dim, overlap


class JigsawAutoTileFactor:
    """
    Berechnet width_factor und height_factor dynamisch - und zwar so, dass
    die ECHTE TTP-Toolset-Formel garantiert einen Overlap > 0 produziert
    (kein Zero-Overlap-Fall mehr, keine harten Bild-Naehte).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": (ANY_TYPE, {"default": 2.0}),
                "overlap_rate": (
                    "FLOAT",
                    {"default": 0.20, "min": 0.0, "max": 0.9, "step": 0.01},
                ),
                "image_already_scaled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "True: das 'image' hier ist bereits auf die "
                            "Zielgroesse (Original x upscale_factor) resized. "
                            "False: 'image' ist noch das unskalierte "
                            "Original."
                        ),
                    },
                ),
                "max_tile_size": (
                    "INT",
                    {
                        "default": MAX_TILE_SIZE_DEFAULT,
                        "min": 64,
                        "max": 4096,
                        "step": 8,
                    },
                ),
                "min_factor": ("INT", {"default": 2, "min": 1, "max": 32}),
                "safety_margin": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": (
                            "Zusaetzlicher Puffer, z.B. 0.05 fuer 5% extra "
                            "Sicherheitsabstand unter max_tile_size."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "width_factor",
        "height_factor",
        "overlap_rate",
        "resulting_tile_w",
        "resulting_tile_h",
    )
    FUNCTION = "compute"
    CATEGORY = "ZEUS.ONL/Tiling"

    def compute(
        self,
        image,
        upscale_factor,
        overlap_rate,
        image_already_scaled,
        max_tile_size,
        min_factor,
        safety_margin,
    ):
        _, h, w, _ = image.shape
        upscale_factor = float(upscale_factor)

        # Falls image NICHT schon skaliert ist, rechnen wir die Canvas-
        # Zielgroesse hoch, BEVOR wir die TTP-Formel simulieren - denn TTP
        # selbst bekommt ja auch die finale (skalierte) Canvas-Groesse.
        canvas_w = w if image_already_scaled else w * upscale_factor
        canvas_h = h if image_already_scaled else h * upscale_factor

        effective_max = max_tile_size * (1.0 - safety_margin)

        width_factor, tile_w, overlap_w = _find_safe_factor(
            canvas_w, overlap_rate, min_factor, effective_max
        )
        height_factor, tile_h, overlap_h = _find_safe_factor(
            canvas_h, overlap_rate, min_factor, effective_max
        )

        print(
            f"  ⚡ [ZEUS.ONL] Auto Tile Factor: canvas={canvas_w:.0f}x{canvas_h:.0f} "
            f"overlap_rate={overlap_rate} -> width_factor={width_factor} "
            f"(tile_w={tile_w}, overlap_w={overlap_w}), "
            f"height_factor={height_factor} (tile_h={tile_h}, overlap_h={overlap_h}) "
            f"limit={effective_max:.1f}"
        )
        if overlap_w <= 0 or overlap_h <= 0:
            print(
                "  ⚠️ [ZEUS.ONL] WARNUNG: Kein Zero-Overlap-freier Faktor "
                f"innerhalb von {MAX_FACTOR_SEARCH} Versuchen gefunden - "
                "harte Naht an mindestens einer Achse weiterhin moeglich."
            )

        return (width_factor, height_factor, overlap_rate, tile_w, tile_h)


NODE_CLASS_MAPPINGS = {
    "JigsawAutoTileFactor": JigsawAutoTileFactor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JigsawAutoTileFactor": "⚡ [ZEUS.ONL] Jigsaw Auto Tile Factor",
}
