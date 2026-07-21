"""
===============================================================================
  ⚡ ZEUS.ONL - MODULE CORE ⚡
  Function: Jigsaw Pure Auto-Sigma Weaver (v13 - Real Live FFT Damping)
  Author: Jigsaw & Zeus
  Official Network: https://zeus.onl
===============================================================================

WHAT THIS NODE ACTUALLY DOES (v13)
  1. Sigma-curve shift (unchanged from v12, now correctly scaled): reshapes the
     incoming sigma schedule's timing using a flow-matching mu-shift, while
     keeping the REAL sigma_max/sigma_min endpoints from whatever sampler
     handed them in (v12 silently discarded these and always returned a
     ~1.25 -> 0.0 range, regardless of the model's actual sigma scale).

  2. Live per-step high-frequency damping (NEW in v13 -- this is the part the
     node's own description always claimed to do but never implemented):
     hooks the diffusion model's forward pass (via the standard, composable
     `add_wrapper_with_key` API -- the same mechanism ComfyUI-Krea2T-Enhancer
     and RegionalCharacterLora use, NOT the fragile single-slot
     `model_options["model_function_wrapper"]` some other nodes clobber each
     other with). At every sampling step it runs a 2D FFT
     (`torch.fft.fft2`) on the model's raw output, measures how much energy
     sits in the ultra-high-frequency band relative to what a "flat" spectrum
     would predict there, and -- ONLY for the excess above that baseline --
     applies a smooth radial attenuation in the frequency domain before
     inverse-FFTing back. Legitimate sharp edges (fabric weave, background
     detail) don't carry anomalous excess high-frequency energy and are left
     alone; the pathological noise/grid-artifact energy that spikes above the
     expected baseline gets pulled back down. This is architecture-agnostic
     (works on any [B,C,H,W]-ish diffusion_model output), unlike the Enhancer
     node which is Krea2-txtfusion-specific.
"""

import torch

try:
    import comfy.patcher_extension as _pext
    _WRAPPER_ENUM = _pext.WrappersMP.DIFFUSION_MODEL
except Exception:
    _pext = None
    _WRAPPER_ENUM = "diffusion_model"

WRAPPER_KEY = "jigsaw_adaptive_sigma_weaver_hf_damp"


def _radial_freq_grid(h, w, device, dtype):
    """Normalized radial frequency distance per FFT bin, 0 (DC) .. 1 (highest
    representable frequency on the shorter axis). Matches torch.fft.fft2's
    (unshifted) bin ordering directly -- no fftshift needed."""
    fy = torch.fft.fftfreq(h, device=device).abs()
    fx = torch.fft.fftfreq(w, device=device).abs()
    r = torch.sqrt(fy.view(h, 1) ** 2 + fx.view(1, w) ** 2)
    r = r / r.max().clamp(min=1e-8)
    return r.to(dtype)


def _smoothstep(x, lo, hi):
    t = ((x - lo) / max(1e-6, (hi - lo))).clamp(0.0, 1.0)
    return t * t * (3 - 2 * t)


def _step_progress(transformer_options):
    """Same pattern as ComfyUI-Krea2T-Enhancer's helper: locate the current
    sigma inside the full schedule to get a 0..1 progress value. Reimplemented
    locally so this node has no import dependency on that other custom node."""
    to = transformer_options or {}
    sigma = to.get("sigmas")
    sigma_value = 0.0
    if torch.is_tensor(sigma) and sigma.numel() > 0:
        sigma_value = float(sigma.detach().flatten()[0].float().item())
    elif isinstance(sigma, (int, float)):
        sigma_value = float(sigma)
    sample_sigmas = to.get("sample_sigmas")
    if torch.is_tensor(sample_sigmas) and sample_sigmas.numel() > 1:
        sig = sample_sigmas.detach().float().flatten()
        idx = int(torch.argmin((sig - sigma_value).abs()).item())
        return idx / max(1, int(sig.numel()) - 1), sigma_value
    return 0.0, sigma_value


def _damp_excess_high_frequency(out, strength, hf_cutoff, progress_gate_lo, progress_gate_hi,
                                 transformer_options, debug):
    """Runs the live FFT measurement + adaptive damping described above.
    Operates on whatever leading dims are present (collapses everything
    except the last 3 dims into the batch dim), so it works for both the
    plain [B,C,H,W] case and Krea2's [B,C,T,H,W]-shaped output alike."""
    if not torch.is_tensor(out) or out.dim() < 3:
        return out

    progress, sigma_now = _step_progress(transformer_options)
    gate = _smoothstep(torch.tensor(float(progress)), progress_gate_lo, progress_gate_hi).item()
    if gate <= 0.0:
        return out

    orig_shape = out.shape
    x = out.reshape(-1, *orig_shape[-3:])  # [N, C, H, W]
    n, c, h, w = x.shape
    if h < 8 or w < 8:
        return out  # too small to say anything meaningful about "ultra-high frequency"

    xf = x.float()
    spec = torch.fft.fft2(xf, dim=(-2, -1))
    mag2 = spec.real.pow(2) + spec.imag.pow(2)

    r = _radial_freq_grid(h, w, x.device, xf.dtype)          # [H, W]
    hf_bin_mask = (r > hf_cutoff).to(xf.dtype)                # hard mask, just for the energy-ratio measurement
    baseline_hf_fraction = hf_bin_mask.mean().clamp(min=1e-6)  # expected share IF energy were flat across bins

    total_energy = mag2.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    hf_energy = (mag2 * hf_bin_mask).sum(dim=(-2, -1), keepdim=True)
    hf_ratio = hf_energy / total_energy                        # [N, C, 1, 1]

    excess = (hf_ratio / baseline_hf_fraction - 1.0).clamp(min=0.0)
    damp_amount = (excess * strength * gate).clamp(max=0.9)     # never fully zero the band -> avoids smearing

    falloff = torch.sigmoid((r - hf_cutoff) * 14.0)             # smooth ramp around the cutoff, no ringing
    atten = 1.0 - falloff.view(1, 1, h, w) * damp_amount        # [N, C, H, W], broadcasts over spatial dims

    spec_damped = spec * atten
    x_damped = torch.fft.ifft2(spec_damped, dim=(-2, -1)).real
    x_damped = x_damped.to(out.dtype).reshape(orig_shape)

    if debug:
        mean_excess = float(excess.mean().item())
        mean_damp = float(damp_amount.mean().item())
        print(f"[JigsawAdaptiveSigmaWeaver] step progress={progress:.3f} sigma={sigma_now:.4g} "
              f"hf_excess={mean_excess:.4g} damp={mean_damp:.4g} gate={gate:.3f}")

    return x_damped


def _make_hf_damp_wrapper(strength, hf_cutoff, progress_gate_lo, progress_gate_hi, debug):
    def wrapper(executor, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
        out = executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)
        if strength <= 0.0:
            return out
        return _damp_excess_high_frequency(
            out, strength, hf_cutoff, progress_gate_lo, progress_gate_hi, transformer_options, debug)
    return wrapper


class JigsawAdaptiveSigmaWeaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sigmas": ("SIGMAS",),  # Empfängt die Standard-Sigmas vom Sampler
                "auto_smoothness": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "hf_damping_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                    "tooltip": "0 = live FFT damping off (sigma-curve shift still applies). Higher values "
                               "pull down excess ultra-high-frequency energy (grid/noise artifacts, blotchy "
                               "'desert skin') harder each step. Legit sharp detail without anomalous HF "
                               "energy is left alone."}),
                "hf_cutoff": ("FLOAT", {"default": 0.62, "min": 0.3, "max": 0.9, "step": 0.02,
                    "tooltip": "Normalized frequency (0=DC, 1=highest) above which energy counts as "
                               "'ultra-high'. Lower = affects a wider band, higher = only the very edge."}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "SIGMAS",)
    RETURN_NAMES = ("model", "adaptive_sigmas",)
    FUNCTION = "weave_sigmas"
    CATEGORY = "🍳 Jigsaw/Samplers"

    def weave_sigmas(self, model, sigmas, auto_smoothness, hf_damping_strength=1.0,
                      hf_cutoff=0.62, debug=False):
        num_steps = len(sigmas) - 1
        if num_steps < 1:
            adaptive_sigmas = sigmas
        else:
            device = sigmas.device
            dtype = sigmas.dtype

            # Keep the REAL endpoints from whatever sampler/model handed us
            # these sigmas -- discarding them (as the old v12 did) silently
            # resets any partial-denoise range / model sigma_max back to a
            # hardcoded ~1.0, which breaks anything but a plain full-denoise
            # txt2img pass.
            sigma_max = float(sigmas[0])
            sigma_min = float(sigmas[-1])

            # AUTOMATISCHE INTELLIGENZ: Errechnet den perfekten Flow-Matching Shift (Mu)
            base_mu = 1.25 if num_steps <= 10 else 1.15
            mu = base_mu * auto_smoothness

            t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
            t_shifted = (mu * t) / (1.0 + (mu - 1.0) * t)
            t_shifted[-1] = 0.0

            adaptive_sigmas = (sigma_min + t_shifted * (sigma_max - sigma_min)).to(dtype=dtype)

        patched = model.clone()
        if hasattr(patched, "remove_wrappers_with_key"):
            patched.remove_wrappers_with_key(_WRAPPER_ENUM, WRAPPER_KEY)
        if hf_damping_strength > 0.0:
            wrapper = _make_hf_damp_wrapper(
                strength=hf_damping_strength,
                hf_cutoff=hf_cutoff,
                progress_gate_lo=0.15,   # don't touch the earliest, coarse-structure steps
                progress_gate_hi=0.5,    # full strength from mid-sampling onward
                debug=bool(debug),
            )
            if hasattr(patched, "add_wrapper_with_key"):
                patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY, wrapper)
            elif hasattr(patched, "add_wrapper"):
                patched.add_wrapper(_WRAPPER_ENUM, wrapper)

        return (patched, adaptive_sigmas)


NODE_CLASS_MAPPINGS = {"JigsawAdaptiveSigmaWeaver": JigsawAdaptiveSigmaWeaver}
NODE_DISPLAY_NAME_MAPPINGS = {"JigsawAdaptiveSigmaWeaver": "🍳 [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver"}
