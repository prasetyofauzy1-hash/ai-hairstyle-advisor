# ML layer

Framework-independent Python package used by the backend.

| Folder | Content | Phase |
|---|---|---|
| datasets/ | dataset manifests and download scripts (no raw images in git) | 3 |
| notebooks/ | exploration only; logic lives in modules | 2-3 |
| preprocessing/ | image quality checks, face crop/alignment | 2 |
| feature_extraction/ | MediaPipe landmarks -> geometric features | 2 |
| classification/ | rule-based baseline, RF / SVM / XGBoost / MLP | 2-3 |
| recommendation/ | catalog filtering, scoring, ranking | 4 |
| evaluation/ | metrics, confusion matrices, CV reports | 3 |
| models/ | serialized models (git-ignored) + model cards | 3 |
