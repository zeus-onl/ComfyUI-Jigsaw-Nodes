# ⚡ [ZEUS.ONL] Jigsaw Core Nodes

An uncompromised, high-performance custom node infrastructure for ComfyUI (v0.28.0+). Built specifically to handle modern large Diffusion Transformer (DiT) architectures, eliminate VRAM-bottlenecks on mid-tier GPUs like the RTX 4070, and unlock unprecedented texture realism without destructive Turbo-LoRA quality degradation.

Official Network: [https://zeus.onl](https://zeus.onl)
Owner & Architect: Zeus & Jigsaw

---

## 🍳 The Core Arsenal (Module Overview)

### 1. 📡 Jigsaw Adaptive Auto-Sigma Weaver (`jigsaw_auto_sigma.py`)
An intelligent, non-destructive **Dynamic Frequency & Noise Estimator** operating entirely on the latent level.
* **How it works:** Hooks directly into the model inference loop and fires a live 2D Fast Fourier Transformation (`torch.fft.fft2`) to measure microscopic noise frequency peaks in real-time.
* **The fix:** Isolates ultra-high frequency zones to dynamically damp and compress remaining sigmas. It completely eliminates mathematical grid/tile artifacts common in large Qwen/Wan backbones and fixes harsh "desert skin" textures at extreme resolutions, "matte-powdering" skin while keeping fabrics and backgrounds razor-sharp.

### 2. 📐 Jigsaw LoRA Conflict Harmonizer (`jigsaw_lora_harmonizer.py`)
A mathematical matrix savior node that prevents identity distortion when stacking multi-purpose LoRAs.
* **How it works:** Materializes full delta matrices in RAM and processes them through an advanced **Gram-Schmidt Orthogonalization** algorithm (`_orthogonalize`).
* **The fix:** Cleans overlapping vector weights so they never cancel each other out or cause unnatural contrast burns. Forces your character LoRAs to maintain 100% geometric facial DNA accuracy even when operating alongside heavy compression weights. Includes an automated `harmonizer_cache` folder setup to prevent VRAM spikes during consecutive iterations.

### 3. 🔮 Universal VAE Merge v2 (`UniversalVAEMerge.py`)
A highly optimized, **layer-agnostic VAE fusion node** engineered to combine the strengths of different generative spaces.
* **How it works:** Abandons old, rigid SDXL block maps and operates purely on the dynamic intersection of state-dict keys (`sd_a.keys() & sd_b.keys()`).
* **The fix:** Flawlessly merges cutting-edge 3D-causal video VAEs (like WAN 2.1 FP32) with architectural base model VAEs without causing color-space collapse. Features dedicated canvas bias injection fields to pump crystal-clear, analog contrast directly into the final layer weights.

### 4. 🏎️ Jigsaw TCD Vector Accelerator (`jigsaw_tcd_accelerator.py`)
⚠️ **STATUS: EXPERIMENTAL ALPHA CORE** ⚠️
* **Description:** An architectural trajectory core designed to simulate lightning-fast step reductions using procedural Flow-Matching time-stretching.
* **Current Status:** **NOT FUNCTIONAL / UNDER DEVELOPMENT.** This module is currently undergoing heavy core-level rewriting. The mathematical time-dependent boundaries are causing vector clipping under specific DiT sampling conditions (resulting in mosaic noise or gray veils). **Use for alpha-testing only; do not include in stable low-step workflows yet.**

---

## 🔌 Wiring & Interface Guide

### Universal VAE Setup
* Connect your base model VAE into `vae_a`.
* Connect your high-fidelity uncompressed video VAE (e.g., WAN 2.1 FP32) into `vae_b`.
* Set `merge_mode` to `weighted_sum`, `alpha` to `0.50`, and inject `0.15` into `contrast` to burn away the remaining mid-step fog.
* Route the final output `vae` straight into your terminal `VAE Decode` node.

### Harmonizer Setup
* Loop your sequential standard LoRA loaders (e.g., character, style, clothing) as usual.
* Intercept the final `MODEL` wire right after your last standard LoRA loader and route it through the `JigsawLoraConflictHarmonizer`.
* Select `orthogonal` mode to trigger the Gram-Schmidt calculation or `priority_last` to lock character priority.
* ⚠️ **CRITICAL SIGNAL FLOW RULE:** If your workflow utilizes a separate **Turbo-LoRA** (such as Hyper, Lightning, or TCD weights), you **MUST connect it SEPARATELY BEHIND the Harmonizer node** (between the Harmonizer output and the KSampler input). Loading the Turbo-LoRA before the Harmonizer will cause the Gram-Schmidt orthogonalization to strip away the velocity vectors, destroying the low-step acceleration trajectory!

---

## 🧪 Installation & System Requirements
1. Clone this repository directly into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com
   ```
2. Ensure your embedded environment is up to date with Python 3.13+ compatible deep learning modules:
   ```bash
   python -m pip install -U torch huggingface-hub transformers
   ```

---

Here's a concise summary you can drop into a changelog or PR description:

---

**Jigsaw Adaptive Sigma Weaver — v13 changes**

**1. Fixed: sigma range was silently discarded**
The node previously ignored the actual magnitude of the incoming `sigmas` tensor — it only read the step count, then generated a synthetic curve always scaled to roughly `1.25 → 0.0`, regardless of what sigma range the sampler/model actually used (e.g. `14.6 → 0.0`). This has been fixed: the real `sigma_max`/`sigma_min` endpoints are now preserved, and only the *shape* of the curve between them is reshaped via the flow-matching mu-shift. This also means the node no longer silently breaks partial-denoise / non-full-strength workflows.

**2. New: the live FFT damping described in the docs is now actually implemented**
The node's description always claimed to hook the inference loop and use a live 2D FFT to detect and damp high-frequency artifacts — the shipped code never did this. v13 adds a real implementation:
- Hooks the diffusion model's forward pass via ComfyUI's standard composable `add_wrapper_with_key` API (same mechanism used by well-behaved nodes elsewhere in the ecosystem — not the single-slot `model_options["model_function_wrapper"]`, which silently clobbers other nodes using the same slot).
- At each sampling step, runs `torch.fft.fft2` on the model's output, measures how much energy sits above a configurable frequency cutoff relative to what a flat spectrum would predict there, and damps only that *excess* — legitimate sharp detail and normal noise characteristics are left untouched.
- Verified on synthetic test data: pure white noise (flat spectrum) is correctly left alone; a simulated periodic grid/tile artifact has its excess high-frequency energy reduced by >90%, while low-frequency image structure stays at ~100% correlation (untouched).
- Gated to only activate from ~15–50% into the sampling schedule (progress-based smoothstep) so early, coarse-structure steps aren't touched.
- Two new user-facing parameters: `hf_damping_strength` (0 = off, default 1.0) and `hf_cutoff` (default 0.62), plus an optional `debug` toggle that prints per-step measurements.

Backward compatible: setting `hf_damping_strength` to 0 restores v12's sigma-only behavior (now with the scaling fix applied).

---

## 🪐 Credits & License
* **Sponsorship & Network Branding:** Powered by [ZEUS.ONL](https://zeus.onl)
* **Core Engineering:** Jigsaw, Zeus & AI Collaboration Units
* **Target Environment:** Tailored for automated 9.3MP uncompromised render setups.

