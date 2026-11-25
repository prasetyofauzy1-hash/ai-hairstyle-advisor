# FFHQ labeling subset

`manifest.csv` lists FFHQ images selected by `python -m ml.datasets.ffhq_select` for human face-shape labeling:
seeded random order, attribution or public-domain Flickr licenses only (non-commercial ones skipped), MD5-checked
against NVIDIA's `ffhq-dataset-v2.json`, and passing the Phase 2 quality gates. Each row keeps the author, license and
Flickr URL for attribution. Images are not committed.

Current manifest: 600 images (Attribution 522, Public Domain Mark 62, CC0 13, US Government Work 3), reached after
checking 872 candidates in seed-42 order (272 rejected by the quality gates or the checksum). Re-running `ffhq_select`
with the same metadata recreates the same list and downloads the images into `data/raw/ffhq/`.

**Split:** `split.csv` assigns the first 104 manifest rows to the human test set, matching the photos already labeled
on the 600-photo page, and the remaining 496 to the AI training set. Do not reorder the manifest. The split separates
image IDs, not identities; the manifest has no identity groups. Regenerate it with:

```sh
python -m ml.datasets.ffhq_split ml/datasets/ffhq_labeling/manifest.csv --out ml/datasets/ffhq_labeling/split.csv
```

The second annotator should label only **Gambar 1–104** on the 600-photo page. To generate an optional page containing
only those 104 photos, pass `--test-split ml/datasets/ffhq_labeling/split.csv` to `ml.datasets.labeling_tool`. Keep the
human test labels and AI training labels separate.

**Status:** One annotator has labeled the test photos; a second independent set of labels is needed
(see `docs/annotation-guide-id.md`).
AI labels are training-only pseudo-labels, never ground truth.

FFHQ is © NVIDIA under CC BY-NC-SA 4.0 (the dataset); individual photos carry the Flickr licenses listed here.
NVIDIA asks that FFHQ not be used to develop face recognition; identity grouping here is only for leakage-free splits.
