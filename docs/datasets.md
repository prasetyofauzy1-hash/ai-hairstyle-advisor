# Dataset Research (Phase 1)

Phase 1 snapshot, checked on 2026-11-25. "Verified" means the fact was read from the source page/API on that date; anything else is marked *reported* or *to verify*. No dataset had been downloaded at that point. See the follow-up in §1 for the download audit and current evaluation recommendation.

## 1. Face shape classification

| Dataset | Source | License (as published) | Samples / classes | Verdict |
|---|---|---|---|---|
| Face Shape Dataset (niten19) | [Kaggle](https://www.kaggle.com/datasets/niten19/face-shape-dataset) | Kaggle page metadata says **CC0** (verified). Images are web-scraped celebrity photos; the uploader cannot relicense them, so real provenance is **unclear**. | 5,000 images, 5 classes: Heart, Oblong, Oval, Round, Square (verified via the HF mirror `bkprocovid19/face_shape`: 5,000 rows, same 5 labels). *Reported* 1,000 per class, 800/200 train/test. | Only candidate of usable size. Research/portfolio use only, never redistributed. Female celebrities only, **no Diamond class, no identity IDs** (identity-split leakage control impossible), label noise *reported* by users. Use as a benchmark, not as ground truth. |
| `codernotme/face_shape` | [HF](https://huggingface.co/datasets/codernotme/face_shape) | MIT tag (verified) | 998 rows; card has no description | Reject: no provenance or label documentation; an MIT tag on unknown images is not a valid license. |
| `Matchewmau/faceshape7` | [HF](https://huggingface.co/datasets/Matchewmau/faceshape7) | CC0 tag | **Empty** (dataset viewer: no data files, verified) | Reject. |
| `xxxhhhollll/face-shape-rule-set`, `EthanCui/face-shape-measurement-reference` | HF | CC-BY-4.0 / CC0 tags | Rule/reference sets, not images | Read only as references for rule thresholds; not training data. |

**Phase 1 conclusion:** the initial search did not find a public face-shape dataset with clear licensing, all 6 target classes, both genders and identity labels. Face-shape labels are also inherently subjective (no agreed geometric definition). The initial plan was:

1. Phase 2/3 start with a **rule-based baseline** on normalized MediaPipe geometry.
2. ML models (RF / SVM / XGBoost / MLP) are trained on the niten19 set **only if the owner approves its use**, reported with the caveats above, and evaluated against the rule baseline.
3. Build a small **self-labeled set** (target ≥ 600 images, ≥ 100 per class, male + female) from license-clear images (FFHQ subset, which keeps per-image license metadata), labeled by ≥ 2 annotators with an annotation guide, measuring inter-annotator agreement (Cohen's / Fleiss' κ). Images with no agreement are dropped. This was the Phase 1 plan to include Diamond.

### Follow-up: public, pre-labeled evaluation data (2026-11-26)

The [current FFHQ test labels](../ml/datasets/ffhq_labeling/human_labels_test.csv) contain 104 images labeled by one person: 64 Oval, 29 Round and 11 Oblong. This limits coverage; it does not by itself establish that those labels are wrong. The [recorded evaluation](../research/results/phase3/kaggle_plus_ffhq_ai/README.md) used the 101 that passed its quality gates and did not measure recall for Heart, Square or Diamond. niten19 is already used in training and is not a new external benchmark.

| Source | Size and labels | Label provenance and license | Access and evaluation verdict |
|---|---|---|---|
| [FaceBench (Wang et al., CVPR 2025)](https://huggingface.co/datasets/wxqlab/FaceBench) · [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_FaceBench_A_Multi-View_Multi-Level_Facial_Attribute_VQA_Dataset_for_Benchmarking_CVPR_2025_paper.html) | **107 distinct shape-labeled images** in the public evaluation JSONL: 5 Diamond, 8 Heart, 16 Oblong, 21 Oval, 38 Round, 19 Square. Of these, 81 are FFHQ and 26 CelebA-HQ; 49,919 is the number of *all* evaluation questions, not shape labels. | Human single-choice answers; the paper (§3.2, Fig. 3) describes majority voting. The release contains final answers, not individual votes. The HF card tags the annotations CC BY 4.0; that does not override FFHQ's [per-photo licenses and dataset CC BY-NC-SA 4.0](https://github.com/NVlabs/ffhq-dataset#licenses) or [CelebA-HQ's non-commercial research terms](https://github.com/switchablenorms/CelebAMask-HQ#license-and-citation). | Annotation file and all 81 FFHQ photos downloaded without auth; all 81 matched NVIDIA's MD5. The [official repository](https://github.com/CVI-SZU/FaceBench) supplies one CelebA-HQ example; the other 25 photos were not downloaded. **Best candidate: use the FFHQ subset as a supplemental research test**, subject to leakage review. |
| [Pasupa, Sunhem & Loo](https://github.com/dsmlr/faceshape) · [paper](https://doi.org/10.1016/j.eswa.2018.11.011) | **500**, 100 each Heart, Oblong, Oval, Round and Square. | Authors' manually labeled celebrity dataset ([2021 review, §2](https://doi.org/10.24193/subbi.2021.2.05)); repeated identities, no per-photo license or independent votes found. The repository asks for citation, which is not a photo license. | No-auth Git clone. All decoded; 289 passed our quality gates. **Reject:** 416 byte-exact matches to the niten19 mirrors and 3 further perceptual-hash candidates; no Diamond. |
| [Hussein Sajid's face-shape photos](https://github.com/hussein073/Hair_Style_Recommendation) | **1,516 original files** in the five shape folders (255 Heart, 280 Long, 362 Oval, 291 Round, 328 Square), excluding `_NEW_` cropped/rotated derivatives and hairstyle references. | The author used agreement across style websites to assign labels to 74 celebrities, scraped the photos and manually screened them for framing. The repo has an MIT license; it does not establish reuse rights for the scraped photos. | No-auth Git clone. All originals decoded; 1,071 passed quality gates. Eight byte-exact + 13 perceptual-hash candidates against niten19. Not independent as-is; no Diamond. |
| [Shashied Face Shape Detector](https://github.com/Shashied/Face-Shape-Detector/tree/53408eb00de24f46035b883dfce75aa04798760f/Face%20Shapes) | **60**, 10 each Diamond, Oblong, Oval, Round, Square and Triangle. Triangle is not Heart. | Internet images sorted by shape; annotators, protocol and image license not stated. | No-auth Git clone. All decoded; 31 passed the quality gates, including only **3 Diamond**. No byte-exact matches; 4 perceptual-hash candidates against niten19, none labeled Diamond. Not a defensible benchmark. |
| [tienDzung multi-label v1](https://universe.roboflow.com/tiendzung/custom-workflow-multi-label-classification-crea9-snkkd/dataset/1) | **3,215** listed, all six target shapes plus `female face` and `Unlabeled`; class counts and shape co-labeling not verified. | No photos' provenance, human labeling process or license verified on the listing. | Public listing; export API returned HTTP 401 (API key required). No image quality or overlap audit. Best six-shape lead beyond FaceBench if the publisher can document label and photo sources. |
| [Rendy Shape face v1](https://universe.roboflow.com/rendy-ijnny/shape-face-oumca-nz3xw/dataset/1) | **2,896** listed. Diamond, Heart, Oval, Rectangle, Round, Square, Triangle and three demographic tags. Rectangle is not verified as Oblong. | Multi-label; photo/annotator provenance and license not verified. | Public listing; export API returned HTTP 401 (API key required). No verified class counts, quality or overlap. |
| [BeautySense face shape (real) v3](https://universe.roboflow.com/face-detection-beautysense/face-shape-real-lagcm/dataset/3) · [Men's Face Shape v1](https://universe.roboflow.com/ahmer-irfan/men-s-face-shape/dataset/1) | **800** listed for BeautySense, includes Diamond, Heart and Rectangle; full mapping to our six classes not established. **604** listed for Men's Face Shape (Diamond, Oblong, Oval, Round, Square, Triangular; no Heart). | Labeling methods, photo sources and licenses not verified. The latter has bounding-box labels, not image classification labels. | Public listings; no archive or no-auth download verified. Per-class counts, quality and overlap unknown. |
| [Kaggle men's face shapes](https://www.kaggle.com/datasets/jawadkhan6575/men-face-shapes-4-types) · [Kaggle Face Shape Preprocessed](https://www.kaggle.com/datasets/zeyadkhalid/faceshape-processed) | Four folders per split for the men's set (57.42 MB; labels/counts unverified); Heart, Oblong, Oval, Round and Square for Preprocessed. | Men's card says license unknown, with no label protocol. Preprocessed card says CC0 and describes crop/rotation, not independent annotation or underlying photo rights. | Anonymous Kaggle download endpoints returned HTTP 404 in this audit; auth requirements, quality and overlap not verified. Neither is a verified six-class test. |
| [CelebHair](https://arxiv.org/abs/2104.06885) | Five face shapes; no Diamond. | The paper's face-shape labels are classifier outputs, not human ground truth. The classifier was trained on the Kaggle data cited in the paper. The paper license is not a photo license. | Not an independent target-label source; no archive downloaded for this comparison. |

**Download and overlap checks.** The [nfzmhn HF archive](https://huggingface.co/datasets/nfzmhn/face_shape) has the niten19 folder layout (800 train + 200 test per class). All 4,998 decodable files matched bytes in the [bkprocovid19 HF Parquet mirror](https://huggingface.co/datasets/bkprocovid19/face_shape); both have two undecodable files (one Heart, one Square). The Kaggle archive itself was not retrieved. The overlap counts above use these mirror images as the reference, compare SHA-256 for exact files, and use a 64-bit grayscale DCT perceptual hash (32×32 resize, low 8×8 coefficients, Hamming distance ≤ 4) to flag potential near-duplicates after excluding exact matches. This does not rule out different photographs of the same person or matches that change under crop/pose.

**Recommended evaluation.** Keep the existing test unchanged and add a *separately reported* FaceBench FFHQ check once cleared for use. All 81 files decoded at 1024 × 1024. None matched the niten19 mirror by SHA-256 or the perceptual-hash threshold above, and none shared an FFHQ ID with the project's 600-image manifest (which includes the existing train and test splits). The most common quality rejection was head turn (18 images).

| FaceBench subset | Diamond | Heart | Oblong | Oval | Round | Square | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| All shape labels in the public release | 5 | 8 | 16 | 21 | 38 | 19 | 107 |
| Downloaded and MD5-verified FFHQ photos | 2 | 6 | 10 | 18 | 33 | 12 | 81 |
| FFHQ photos passing `FaceAnalyzer` quality gates | 2 | 5 | 7 | 12 | 23 | 12 | 61 |

The original SFace-only screen flagged 50 of 61 quality-passing photos and would have left 11 to score. The [identity audit](../research/results/phase3/facebench_ffhq/identity_review.csv) corroborated two of those 50 as matches to the FFHQ train split; both references are marked `ragu` (unsure) and were not used to fit the model. We still excluded those two photos to keep the [separate FaceBench benchmark](../research/results/phase3/facebench_ffhq/README.md) independent of the full designated training pool. All six classes remain, but with just one Diamond example the scores cannot establish a reliable model ranking or replace the Uzy benchmark. If the CelebA-HQ part is approved later, verify its image-ID mapping, quality and leakage before combining it. Do not substitute Triangle for Heart or Rectangle for Oblong without provenance that supports the mapping. No retraining or test split changes were made.

| Model | FaceBench FFHQ macro-F1 (59 photos) | [Uzy FFHQ macro-F1](../research/results/phase3/kaggle_plus_ffhq_ai/README.md) (101 photos) |
|---|---:|---:|
| Phase 2 rules | 0.306 | 0.391 |
| Kaggle-only MLP | 0.251 | 0.321 |
| Kaggle + FFHQ AI pseudo-labels MLP | 0.324 | 0.340 |

The scores are not directly comparable: FaceBench averages six shapes; Uzy averages Oblong, Oval and Round. Each benchmark report includes its own confusion matrices.

To stage the FaceBench FFHQ photos and an attribution manifest locally for non-commercial research, first obtain NVIDIA's `ffhq-dataset-v2.json` from the [official FFHQ distribution](https://github.com/NVlabs/ffhq-dataset#overview). Then run:

```sh
python -m ml.datasets.facebench_ffhq data/raw/ffhq-dataset-v2.json \
    --images data/raw/facebench_ffhq/images \
    --manifest data/raw/facebench_ffhq/manifest.csv
```

This downloads the pinned FaceBench annotation release, keeps the six shape answers on FFHQ, skips IDs already in the project's FFHQ manifest, and checks photos against NVIDIA's MD5 values. It retains the upstream labels and per-photo credits; the manifest is a **candidate** evaluation set, not a quality- or identity-filtered split. Keep photos local, cite both sources, retain Flickr attribution and observe the applicable non-commercial and share-alike terms.

To reproduce the benchmark, unpack the full niten19 mirror archive into `data/raw/niten19` (the root below is the directory containing both `training_set` and `testing_set`). The [nfzmhn archive](https://huggingface.co/datasets/nfzmhn/face_shape/blob/fb498ced096bba0c679865336fb5a0a92d5a792b/archive.zip) audited here has SHA-256 `4b0c3f42f8f02ac0dda5d7b332394240d50ae9d440fea424ce953681dd025efe`.

```sh
python -m ml.datasets.build_features "data/raw/niten19/FaceShape Dataset" \
    --out data/processed/niten19
python -m ml.evaluation.facebench data/raw/facebench_ffhq/manifest.csv \
    --images data/raw/facebench_ffhq/images \
    --ffhq-train data/raw/ffhq \
    --niten-embeddings data/processed/niten19/embeddings.npy \
    --identity-review research/results/phase3/facebench_ffhq/identity_review.csv \
    --out research/results/phase3/facebench_ffhq
```

The scorer downloads any missing FFHQ training photos and checks their manifest MD5s. The exact trained-row indices are not part of either model bundle, and the two models used different niten19 splits. The check therefore uses a conservative superset: both source folders of niten19 and all 496 FFHQ train IDs, including photos not used in fitting. The scorer reproduces the original SFace screen and requires a decision for every SFace flag. After the original quality and embedding gates, it excludes only confirmed identity matches from the audit.

**Identity audit.** Cosine 0.363 is [OpenCV's SFace threshold on LFW](https://github.com/opencv/opencv/blob/4.x/doc/tutorials/dnn/dnn_face/dnn_face.markdown), not just a grouping threshold. The original benchmark applied it to the maximum of thousands of individual comparisons per query, not to a known pair or to an average-linkage group. A threshold crossing in that search is a candidate, not verified leakage.

We independently searched the two galleries using [dlib's face recognition ResNet](https://dlib.net/face_recognition.py.html) (model v1, dlib 20.0.1). For each RGB image, we downscaled the longer side to at most 480 px, used dlib's frontal HOG detector (retrying with one upsampling step if needed), aligned with its five-point predictor, and compared the unnormalized 128-D descriptors by Euclidean distance without jitter. All 81 FaceBench photos, 4,992 of 5,000 niten19 photos and all 496 FFHQ training photos had a dlib descriptor. The independent dlib screen at distance < 0.6 flagged 54 of 61 eligible photos, including eight not flagged by SFace. We visually reviewed the nearest pairs from both models for all 58 flagged photos, all 214 pairs passing both thresholds, and 26 FFHQ pairs with a shared source photographer. Shared capture context corroborated two matches; the correct reference in one of them fell below the SFace threshold despite a different pair flagging the query. The [decision CSV](../research/results/phase3/facebench_ffhq/identity_review.csv) contains FaceBench filenames, screening scores, and decisions, without reference filenames or personal details. Unconfirmed pairs remain in the benchmark; that does not prove leakage is absent. Keep photos, embeddings, and pairing evidence local.

### Follow-up: Diamond and male training/evaluation coverage

No additional source was verified for training. The existing FFHQ training labels already supply three quality-passing Diamond photos, all with male estimates; these are AI pseudo-labels, not independently annotated Diamond examples. The 59-photo FaceBench cohort stays held out. See the [retraining experiment](../research/results/phase3/diamond_retrain/README.md) for the separate Uzy and FaceBench results, gender-estimate counts, and reproduction command.

| Additional lead | What the audit found |
|---|---|
| [Men's Face Shape](https://universe.roboflow.com/ahmer-irfan/men-s-face-shape) and [Haircolor](https://universe.roboflow.com/haircolor/face-shape-vfgat) | Both list 604 images with bounding-box annotations and the same six class names, including Diamond and Triangular instead of Heart. The Men's Face Shape export returned HTTP 401; neither archive was verified. Same count and label names are not proof of independent data or of duplication. Class counts, gender composition, photo rights and annotator provenance remain unverified. |
| [tienDzung Diamond](https://universe.roboflow.com/tiendzung/diamond-face) and [multi-label v1](https://universe.roboflow.com/tiendzung/custom-workflow-multi-label-classification-crea9-snkkd/dataset/1) | The Diamond project lists 590 images and `diamond faceshape` / `female face` labels, not a verified male cohort. Both exports returned HTTP 401, and no public archive mirror was verified. No photo rights, annotation protocol, or overlap verified. |
| [LAIAIAI](https://github.com/lon5948/LAIAIAI) | The README describes Diamond, Oblong, Oval, Round, Square and Triangle classification. Both linked face-shape data downloads resolved to Google Drive sign-in pages. Counts, source-photo rights, annotation methods and overlap remain unknown. |
| [Eyes on Fandat](https://zenodo.org/records/15393475) | Public CSV preview lists 30 rows with gender and shape tags, including 12 male tags and two Diamond tags (both female). The CC BY record lists only the CSV, not the described Unsplash photos or links to recover them; labels could not be matched to usable photos. |
| [3D-Guided Face Shape Classification](http://hdl.handle.net/10211.3/215338) and [Face Shape Measurement Reference](https://huggingface.co/datasets/EthanCui/face-shape-measurement-reference) | The thesis discusses the six target shapes, but a labeled photo archive, counts and license were not verified. The measurement reference supplies rules by shape, not labeled photos. |

Sources without image archives could not be screened for image quality, exact/perceptual duplicates, or identity overlap. The original [Shashied audit](#follow-up-public-pre-labeled-evaluation-data-2026-11-26) downloaded Diamond photos but still has unverified photo rights and labels. A usable next source needs accessible photos, documented annotation and photo permissions, and enough distinct Diamond identities to reserve an independent holdout.

## 2. Facial landmarks

| Resource | License | Notes |
|---|---|---|
| MediaPipe Face Landmarker (Face Mesh v2) | Apache-2.0 (repo verified) | 478 3D landmarks + 52 blendshapes. Model file downloadable (HTTP 200 verified). Used as-is, no training. |
| 300W / WFLW | Research-only (*to verify*) | Only needed if we later want to benchmark landmark accuracy. Not needed for Phase 2. |

## 3. Hair segmentation

| Resource | License | Notes |
|---|---|---|
| MediaPipe Hair Segmenter | Apache-2.0 (MediaPipe) | Binary hair mask; model file downloadable (verified). First choice for Phase 6. |
| MediaPipe Selfie Multiclass (256x256) | Apache-2.0 (MediaPipe) | Classes: background, hair, body-skin, face-skin, clothes, others; downloadable (verified). Gives face-skin mask too, useful for protecting the face during inpainting. |
| CelebAMask-HQ | **Non-commercial research only**, no redistribution (verified in repo) | 30k images, 19-class face parsing incl. hair (*reported*). Only for evaluating our masks offline. |
| Figaro-1k | Research use (*to verify*; project page did not load) | 1,050 images, 7 hair-texture classes (*reported*). Possible source for hair-type (straight/wavy/curly) experiments. |

## 4. Hairstyle images / catalog

No license-clear public catalog with male + female hairstyle metadata was found. The catalog (Phase 4) will be **curated by us**: structured metadata written from published styling guidance, with reference images either self-sourced under CC licenses (attributed) or generated by the Phase 6 pipeline and labeled as generated. *Reported* alternative: K-Hairstyle (AI-Hub, Korea) has 500k+ images but access is restricted; not pursued.

## 5. Virtual hairstyle transfer

| Resource | License | Notes |
|---|---|---|
| FFHQ | Per-image CC-BY / CC-BY-NC / PD (verified); NVIDIA says not for face recognition development | Test images for try-on evaluation; also the planned source for self-labeling (keep per-image attribution). |
| CelebA / CelebA-HQ | Non-commercial research (*reported*) | CelebA has `Straight_Hair` / `Wavy_Hair` attributes (*reported*). Evaluation only. |

## Phase 1 decisions (historical)

- Approve (or reject) using the niten19 Kaggle set for research experiments under the caveats in §1.
- Approve the self-labeling plan (needs ~2 annotators, est. 6-10 h total for 600 images).

## Preprocessing requirements (all image sources)

- Detect exactly one dominant face; reject multi-face / no-face / heavy occlusion.
- Quality gates: min face size (e.g. ≥ 200 px inter-temple width), blur (variance of Laplacian), exposure, head pose (|yaw|, |pitch| ≤ ~15° for measurements).
- Align on eye centers (roll correction) before measuring.
- Keep a manifest CSV per dataset: file id, source URL, license, label, annotator(s), identity id if known, split.
