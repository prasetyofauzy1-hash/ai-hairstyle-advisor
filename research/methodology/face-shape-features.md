# Face-Shape Features: Method Draft

Status: Phase 2. Indices below were checked visually on the overlays in `research/results/phase2/`. Implemented choices: forehead 54↔284, cheekbones 234↔454, jaw 172↔397, chin 176↔400, jaw angle at 172/397 (between 132→172→148 and 361→397→377), length 10→152, landmarks rotated so the iris line (468, 473) is horizontal. See `ml/feature_extraction/geometry.py`.

| Measurement | Candidate MediaPipe landmarks | Note |
|---|---|---|
| Face length | 10 (top of face mesh) → 152 (chin) | 10 is below the real hairline; true face length is underestimated, more so with high foreheads |
| Forehead width | 54 ↔ 284 (or 103 ↔ 332) | Often covered by hair/bangs |
| Cheekbone width | 234 ↔ 454 | Widest face points; also used as scale reference |
| Jaw width | 172 ↔ 397 (or 58 ↔ 288) | Gonial angle region |
| Chin width / shape | 148 ↔ 377, angle at 152 | Helps separate Heart / Diamond / Square |
| Inter-ocular distance | eye centers (iris 468, 473) | Alternative scale reference, robust to hair |

Derived (scale-invariant) features: width/length ratio, forehead/jaw, jaw/cheekbone, forehead/cheekbone, chin width/jaw width, jaw angle, contour curvature along the face oval (landmarks of the silhouette ring).

## Rule-based baseline (to be calibrated, not tuned on the test set)

Initial rules follow common styling-guide definitions, e.g. Oblong: length/width high; Round: length≈width with soft jaw angle; Square: length≈width with jaw≈cheek and sharp jaw angle; Heart: forehead > cheek > jaw with narrow chin; Diamond: cheek clearly > forehead and > jaw; Oval: remainder with moderate ratios. Thresholds are chosen on a calibration split and frozen before test evaluation; the rule output is converted to soft scores so a confidence can be shown honestly.

## Limitations to document in results

- Labels are subjective; inter-annotator agreement bounds achievable accuracy.
- Hairline not detected by MediaPipe; bangs hide forehead; beards hide jawline.
- Head pose, lens distortion (selfie at short distance) and facial expression change ratios.
- Datasets skew to specific demographics; report per-gender results where labels allow.
