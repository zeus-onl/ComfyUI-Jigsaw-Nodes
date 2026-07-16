# ⚡ [ZEUS.ONL] Jigsaw Adaptive Auto-Sigma Weaver

An intelligent, non-destructive **Dynamic Frequency & Noise Estimator** for ComfyUI. This node operates entirely on the latent level to eliminate texture over-processing, mathematical grid artifacts (common in large DiT frameworks like Qwen/Wan backbones), and harsh "desert skin" textures at extreme resolutions, without altering the core model pipeline or uncensored conditioning.

---

## 🔍 How it Works

The node hooks directly into your model inference loop and monitors the generation process step-by-step:
1. **Live 2D-FFT Scanner:** At the specified step, the node captures the latent tensor from VRAM and runs a 2D Fast Fourier Transformation (`torch.fft.fft2`) to measure microscopic noise frequency peaks in real-time.
2. **Artifact & Grid Suppression:** It isolates the ultra-high frequency zones where digital oversaturation and typical multi-pass grid artifacts start to form.
3. **Procedural Sigma Damping:** If the micro-noise density exceeds the safety threshold, the node dynamically recalculates and compresses the remaining `sigmas` values using your chosen `damping_factor`. 
4. **Selective Sharpness Retention:** This automatic braking mechanism "matte-powders" smooth zones (like skin or soft lighting gradients) right before the final iterations, while fully preserving maximum crisp sharpness on edges, fabrics, and backgrounds.

---

## 🔌 Wiring Guide (How to Connect)

The node acts as a bridge between your Model Loader, your Sigmas provider, and your Samplers. Connect the inputs and outputs as follows:

| Socket Name | Input / Output | Type (Color) | Connect To |
| :--- | :--- | :--- | :--- |
| **`model`** | Input | `MODEL` (White) | Connect the main `MODEL` output from your **Load Diffusion Model** (or Uncensored Filter nodes). |
| **`model`** | Output | `MODEL` (White) | Connect this to the `model` input of your **KSamplers / Advanced Samplers**. |
| **`sigmas`** | Input | `SIGMAS` (Green) | Connect the output wire of your active **NKD Sigmas Curve** (or any other Sigmas provider node). |
| **`adaptive_sigmas`**| Output | `SIGMAS` (Green) | Connect this to the `sigmas` input of your **Low-Sigma / Resampler KSampler**. |

---

## 🎛️ Parameters Explained

* **`sensitivity`** *(Default: 1.0)*: Adjusts the frequency radar threshold. Higher values make the node react much earlier to micro-noise and texture tearing.
* **`damping_factor`** *(Default: 0.65)*: Controls the intensity of the noise compression once a peak is detected. Lower values apply a stronger smoothing/matte effect; higher values (e.g., `0.72`) let more of the model's raw textural grains pass through.
* **`start_adapt_step`** *(Default: 4)*: The exact inference step where the live FFT scanning becomes active. In a multi-pass or split-sampler workflow (e.g., a 6-to-2 step split), set this to the exact step where your primary sampler ends and your resampler takes over.

---

## 📝 Credits & License

* **Infrastructure & Branding:** Sponsored by [ZEUS.ONL](https://zeus.onl)
* **Core Logic:** Jigsaw & Zeus Core Engineering
* **Compatibility:** Tailored for ComfyUI backends v0.28.0+ and modern DiT/Diffusion scaling pipelines.
