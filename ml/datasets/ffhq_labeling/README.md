# FFHQ labeling subset

`manifest.csv` lists FFHQ images selected by `python -m ml.datasets.ffhq_select` for human face-shape labeling:
seeded random order, attribution or public-domain Flickr licenses only (non-commercial ones skipped), MD5-checked
against NVIDIA's `ffhq-dataset-v2.json`, and passing the Phase 2 quality gates. Each row keeps the author, license and
Flickr URL for attribution. Images are not committed.

**Status: no labels yet.** Labels need at least two human annotators (see `docs/annotation-guide-id.md`); no
annotators have been named, so no FFHQ result exists. Labels made by an AI would not be ground truth and must not be used.

FFHQ is © NVIDIA under CC BY-NC-SA 4.0 (the dataset); individual photos carry the Flickr licenses listed here.
NVIDIA asks that FFHQ not be used to develop face recognition; identity grouping here is only for leakage-free splits.
