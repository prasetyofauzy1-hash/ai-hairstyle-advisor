# Pretrained Models (Phase 1 survey)

Licenses read from Hugging Face / GitHub APIs on 2026-11-25 (verified) unless marked *reported*. No model is trained from scratch.

| Purpose | Model | License | Fit for RTX 3060 Ti 8 GB |
|---|---|---|---|
| Landmarks | MediaPipe Face Landmarker | Apache-2.0 | CPU real-time; no GPU needed |
| Hair / face masks | MediaPipe Hair Segmenter, Selfie Multiclass | Apache-2.0 | CPU |
| Inpainting (primary) | `stable-diffusion-v1-5/stable-diffusion-inpainting` | CreativeML OpenRAIL-M | Yes, fp16 at 512x512 (*expected* ~4-5 GB) |
| Inpainting (quality option) | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | OpenRAIL++ | Tight; needs fp16 + CPU offload, slower (*to benchmark*) |
| Structure control | `lllyasviel/control_v11p_sd15_inpaint`, `..._openpose` | OpenRAIL | Yes with SD1.5 (+~1.5 GB *expected*) |
| Identity conditioning | `h94/IP-Adapter-FaceID` | **Research only, non-commercial** (depends on InsightFace models) | Yes with SD1.5 |
| Dedicated hair transfer | [HairFastGAN](https://github.com/AIRI-Institute/HairFastGAN) (NeurIPS 2024) | MIT code; uses FFHQ-trained StyleGAN weights (*check weights license*) | *To benchmark*; GAN encoder pipeline, aligned 1024 faces |
| Dedicated hair transfer | [Stable-Hair](https://github.com/Xiaojiu-z/Stable-Hair) | Apache-2.0 code | SD1.5-based, *to benchmark* |
| Identity metric | InsightFace ArcFace (buffalo_l) | Code MIT; **models non-commercial research only** (verified) | Yes |
| Image metrics | LPIPS, FID/KID (torchmetrics), CLIP score | BSD/MIT/Apache | Yes |

## Implication

The project as specified is a **non-commercial research / portfolio project**. If it should ever be commercial, IP-Adapter-FaceID, InsightFace models, CelebAMask-HQ and the Kaggle set must be replaced.

## Phase 6 plan

1. Baseline: SD1.5 inpainting + hair mask (dilated) + face-skin mask protected + text prompt of the target hairstyle.
2. Add ControlNet inpaint / openpose for head-shape consistency.
3. Compare with HairFastGAN and Stable-Hair (reference-image driven) on the same 50-100 FFHQ test faces.
4. Pick the method with the best identity similarity at acceptable realism and runtime.
