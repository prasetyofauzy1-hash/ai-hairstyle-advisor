# Compute Requirements (target: RTX 3060 Ti 8 GB)

Numbers below are **estimates** from typical published figures, not measurements on this hardware. Phase 6 starts with a benchmark script that records real VRAM and latency into `research/results/`.

| Workload | Device | Estimate |
|---|---|---|
| MediaPipe landmarks + features | CPU | tens of ms per image |
| RF / SVM / XGBoost training on ~5k feature vectors | CPU | seconds to minutes incl. grid search |
| MLP (optional) | CPU/GPU | minutes |
| SD1.5 inpainting fp16, 512x512, ~30 steps | GPU | ~4-5 GB VRAM, a few seconds per image |
| + ControlNet (one) | GPU | +~1-1.5 GB, ~1.3-1.5x time |
| SDXL inpainting fp16 1024x1024 | GPU | exceeds 8 GB without `enable_model_cpu_offload`; with offload tens of seconds |
| HairFastGAN / Stable-Hair | GPU | unknown on 8 GB, benchmark before choosing |

Other: RAM 16 GB minimum (32 GB comfortable), ~30-50 GB disk for model weights + datasets, CUDA 12.x driver.

## Trade-off proposed

Use **SD1.5 inpainting (+ ControlNet) at 512px** as the default try-on engine: fits 8 GB with margin and runs interactively. SDXL is kept as an optional "high quality" mode with offload. Training of diffusion models is out of scope; light LoRA fine-tuning is only considered later and only if evaluation shows a clear gap.
