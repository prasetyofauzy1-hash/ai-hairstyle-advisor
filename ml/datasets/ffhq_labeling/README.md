# FFHQ labeling subset

`manifest.csv` lists FFHQ images selected by `python -m ml.datasets.ffhq_select` for human face-shape labeling:
seeded random order, attribution or public-domain Flickr licenses only (non-commercial ones skipped), MD5-checked
against NVIDIA's `ffhq-dataset-v2.json`, and passing the Phase 2 quality gates. Each row keeps the author, license and
Flickr URL for attribution. Images are not committed.

Current manifest: 600 images (Attribution 522, Public Domain Mark 62, CC0 13, US Government Work 3), reached after
checking 872 candidates in seed-42 order (272 rejected by the quality gates or the checksum). Do not re-run
`ffhq_select` to get the images: it rewrites the manifest under the current quality gates, which may no longer match
the split and labels. `python -m ml.datasets.ffhq_features --download` fetches this manifest's photos into
`data/raw/ffhq/` instead.

**Split:** `split.csv` assigns the first 104 manifest rows to the human test set, matching the photos already labeled
on the 600-photo page, and the remaining 496 to the AI training set. Do not reorder the manifest. The split separates
image IDs, not identities; the manifest has no identity groups. Regenerate it with:

```sh
python -m ml.datasets.ffhq_split ml/datasets/ffhq_labeling/manifest.csv --out ml/datasets/ffhq_labeling/split.csv
```

To generate an optional page containing only the 104 test photos, pass
`--test-split ml/datasets/ffhq_labeling/split.csv` to `ml.datasets.labeling_tool`.

**Status:** Uzy alone labeled the 104 test photos. The second annotator was dropped, so inter-annotator agreement
cannot be measured.

`human_labels_test.csv` holds Uzy's 104 test labels (oval 64, round 29, oblong 11), in split order. They are used only
for evaluation (`python -m ml.datasets.ffhq_features`, then `python -m ml.training.train_face_shape_ffhq`); results
are in `research/results/phase3/kaggle_plus_ffhq_ai/`.

`ai_pseudo_labels_train.csv` contains the 496 training photos' AI-generated labels. Devin labeled them with its
own vision from 256×256 pixel contact-sheet tiles, without an external API. These pseudo-labels are for training
only, are never ground truth, and must never be merged with the human test labels.

`ragu` marks photos excluded from training because a face shape could not be judged confidently, for example
because of occlusion from hats, sunglasses combined with other obstructions, wigs, hands, or microphones, or
because of strong head tilt or expression.

| Label | Photos |
| --- | ---: |
| round | 151 |
| oval | 92 |
| oblong | 57 |
| square | 35 |
| heart | 25 |
| diamond | 4 |
| ragu | 132 |
| **Total** | **496** |
| **Usable for training** | **364** |

Diamond is severely under-represented, with only four usable examples.

FFHQ is © NVIDIA under CC BY-NC-SA 4.0 (the dataset); individual photos carry the Flickr licenses listed here.
NVIDIA asks that FFHQ not be used to develop face recognition; identity grouping here is only for leakage-free splits.
