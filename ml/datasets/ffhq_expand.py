"""Stage licensed FFHQ candidates for local visual pseudo-labeling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import re
import socket
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml.classification.rule_based import SHAPES
from ml.datasets.agreement import UNSURE
from ml.datasets.build_features import analyze_items
from ml.datasets.face_crops import face_assets
from ml.datasets.ffhq_features import SUBSET, _read
from ml.datasets.ffhq_select import FIELDS, candidates, fetch
from ml.datasets.identity import SAME_PERSON_COSINE
from ml.evaluation.gender_estimates import estimates
from ml.preprocessing.image_io import ImageDecodeError, load_image
from ml.training import kaggle_kernel as kernel
from ml.training.train_face_shape_diamond import estimate

SEED = 103


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def candidate_ids(meta, forbidden, attributes, male, other):
    quotas = {True: male, False: other}
    for index in candidates(meta, SEED):
        fid = f"{index:05d}"
        if fid in forbidden:
            continue
        priority = estimate(attributes, fid) == "male"
        if quotas[priority] > 0:
            yield index
            quotas[priority] -= 1
        if not any(quotas.values()):
            return


def download(args):
    if not (args.attributes / "json").is_dir():
        raise ValueError("Missing DCGM FFHQ attributes checkout")
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    fixed = _read(SUBSET / "manifest.csv") + _read(args.facebench_manifest)
    forbidden = {f'{int(r["ffhq_id"]):05d}' for r in fixed}
    ids = list(candidate_ids(meta, forbidden, args.attributes, args.male, args.other))
    args.workspace.mkdir(mode=0o700, exist_ok=True)
    images = args.workspace / "images"
    images.mkdir(exist_ok=True)

    def one(index):
        entry = meta[str(index)]
        md5 = entry["image"]["file_md5"]
        try:
            photo = fetch(index, md5, images)
        except HTTPError as exc:
            if exc.code not in {404, 410}:
                raise
            return None
        if photo is None:
            return None
        return {field: entry["metadata"][field] for field in FIELDS[1:-2]} | {
            "ffhq_id": f"{index:05d}", "file_md5": md5, "local_file": photo.name,
        }

    socket.setdefaulttimeout(35)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for n, row in enumerate(pool.map(one, ids), 1):
            if row:
                rows.append(row)
            if n % 100 == 0:
                print(f"Fetched {len(rows)} of {n} licensed candidates checked.", flush=True)
    write_rows(args.workspace / "candidates.csv", FIELDS, rows)
    print(f"Staged {len(rows)} verified candidates of {len(ids)} checked.", flush=True)


def init_dlib(landmarks, recognizer):
    import dlib

    global detector, predictor, descriptor
    cv2.setNumThreads(1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(landmarks))
    descriptor = dlib.face_recognition_model_v1(str(recognizer))


def dlib_faces(path):
    try:
        rgb = load_image(path)
    except (ImageDecodeError, OSError):
        return None
    factor = min(1, 480 / max(rgb.shape[:2]))
    if factor < 1:
        rgb = cv2.resize(rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    boxes = detector(rgb) or detector(rgb, 1)
    return [np.asarray(descriptor.compute_face_descriptor(rgb, predictor(rgb, face)), dtype=np.float32)
            for face in boxes]


def dlib_embeddings(photos, args):
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_dlib,
                             initargs=(args.dlib_landmarks, args.dlib_recognizer),
                             mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(dlib_faces, photos, chunksize=8))


def reference_bundle(args):
    paths = [Path(line) for line in args.photos.read_text().splitlines() if line]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Supply a nonempty list of distinct reference paths")
    hashes = [kernel.digest(path) for path in paths]
    _, sface = analyze_items([(p.parent, p.name, "", "reference") for p in paths], args.workers)
    per_image = dlib_embeddings(paths, args)
    unreadable = sum(faces is None for faces in per_image)
    indices = [i for i, faces in enumerate(per_image) for _ in (faces or [])]
    dlib = np.asarray([face for faces in per_image for face in (faces or [])], dtype=np.float32).reshape(-1, 128)
    with args.references.open("wb") as archive:
        np.savez_compressed(archive, sha256=hashes, sface=sface, dlib=dlib,
                            dlib_index=np.asarray(indices, dtype=int))
    print(f"Skipped dlib embeddings for {unreadable} unreadable reference photos.", flush=True)
    print(f"Archived hashes and identity-screen features locally for {len(paths)} reference photos.", flush=True)


def identity_flags(sface, dlib, reference_sface, reference_dlib, reference_indices):
    cosine = sface @ reference_sface.T
    squared = (dlib ** 2).sum(axis=1)[:, None] + (reference_dlib ** 2).sum(axis=1) - 2 * dlib @ reference_dlib.T
    face_distance = np.full_like(cosine, np.inf)
    np.minimum.at(face_distance.T, reference_indices, squared.T)
    strong = (cosine >= 0.56) | (face_distance < 0.54 ** 2)
    joint = (cosine >= SAME_PERSON_COSINE) & (face_distance < 0.60 ** 2)
    return strong | joint


def diamond_probabilities(pixels, path):
    import torch

    torch.set_num_threads(4)
    device = torch.device("cpu")
    checkpoint = torch.load(path, weights_only=True, map_location=device)
    classes = checkpoint["classes"]
    if "diamond" not in classes:
        raise ValueError("The screening checkpoint must have a Diamond class")
    model = kernel.network(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(pixels), 32):
            batch = kernel.image_batch(pixels[start:start + 32], torch, device)
            probability = ((model(batch) + model(batch.flip(-1))) / 2).softmax(1)
            values.extend(probability[:, classes.index("diamond")].numpy())
    return values


def sheets(photos, target):
    target.mkdir(exist_ok=True)
    for page in target.iterdir():
        if re.fullmatch(r"[0-9]{2,}\.jpg", page.name) and page.is_file():
            page.unlink()
    for start in range(0, len(photos), 16):
        canvas = Image.new("RGB", (4 * 256, 4 * 278), "#eee")
        draw = ImageDraw.Draw(canvas)
        for i, photo in enumerate(photos[start:start + 16]):
            x, y = (i % 4) * 256, (i // 4) * 278
            with Image.open(photo) as image:
                canvas.paste(image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS), (x, y))
            draw.text((x + 8, y + 257), f"{i + 1:02d}   {photo.stem}", fill="black")
        canvas.save(target / f"{start // 16 + 1:02d}.jpg", quality=92)


def screen(args):
    ledger = _read(args.workspace / "candidates.csv")
    root = args.workspace / "images"
    all_photos = [root / row["local_file"] for row in ledger]
    rows, sface = analyze_items([(p.parent, p.name, "", "train") for p in all_photos], args.workers)
    good = [i for i, row in enumerate(rows) if row["passed"] and np.isfinite(sface[i]).all()]
    print(f"Quality gates and SFace: {len(good)} of {len(rows)} accepted.", flush=True)
    photos = [all_photos[i] for i in good]
    per_image = dlib_embeddings(photos, args)
    one_face = [i for i, values in enumerate(per_image) if values is not None and len(values) == 1]
    good = np.asarray(good)[one_face]
    dlib = np.asarray([per_image[i][0] for i in one_face])
    sface = sface[good]
    photos = [all_photos[i] for i in good]
    if not photos:
        raise ValueError("No candidates passed both face detectors")
    print(f"Both identity models produced one-face features for {len(photos)} candidates.", flush=True)
    with np.load(args.references, allow_pickle=False) as ref:
        hits = identity_flags(sface, dlib, ref["sface"], ref["dlib"], ref["dlib_index"]).any(axis=1)
        exact_shas = set(ref["sha256"])
    hashes = [kernel.digest(p) for p in photos]
    keep = np.asarray([i for i, sha in enumerate(hashes) if sha not in exact_shas and not hits[i]], dtype=int)
    print(f"Reference exact/identity checks retained {len(keep)} of {len(photos)} candidates.", flush=True)
    if not len(keep):
        raise ValueError("No candidates survived reference identity screening")
    photos = [photos[i] for i in keep]
    gender = estimates(photos, args.gender_model, args.workers)
    scores = diamond_probabilities(face_assets(photos, args.cache, args.workers), args.checkpoint)
    duplicates = identity_flags(sface[keep], dlib[keep], sface[keep], dlib[keep], np.arange(len(keep)))
    ranked = sorted(range(len(keep)), key=lambda i: -(scores[i] + (0.20 if gender[i] == "male" else 0)))
    reviewed, seen, shas = [], [], set()
    for i in ranked:
        sha = hashes[keep[i]]
        if sha in shas or duplicates[i, seen].any():
            continue
        seen.append(i)
        shas.add(sha)
        reviewed.append(ledger[good[keep[i]]] | {"estimated_gender": gender[i], "diamond_score": float(scores[i])})
        if len(reviewed) == args.review_count:
            break
    write_rows(args.workspace / "review.csv", FIELDS + ["estimated_gender", "diamond_score"], reviewed)
    sheets([root / row["local_file"] for row in reviewed], args.workspace / "sheets")
    print(f"Prepared {len(reviewed)} distinct photos for blind visual labeling.", flush=True)


def build_features(args):
    manifest = _read(args.manifest)
    labels = _read(args.labels)
    by_id = {row["ffhq_id"]: row for row in labels}
    if (not manifest or len(by_id) != len(labels) or len(manifest) != len(labels)
            or len({row["ffhq_id"] for row in manifest}) != len(manifest)
            or set(by_id) != {row["ffhq_id"] for row in manifest}
            or any(row["source"] != "ai_pseudo_label" or row["label"] not in {*SHAPES, UNSURE} for row in labels)):
        raise ValueError("Every reviewed candidate must have one training-only pseudo-label or ragu")
    items, ids = [], []
    for row in manifest:
        fid = row["ffhq_id"]
        label = by_id[fid]["label"]
        if label == UNSURE:
            continue
        filename = row["local_file"]
        if filename != f"{int(fid):05d}.png":
            raise ValueError("Manifest filename does not match FFHQ ID")
        if hashlib.md5((args.images / filename).read_bytes()).hexdigest() != row["file_md5"]:
            raise ValueError("Expansion photo differs from reviewed FFHQ file")
        items.append((args.images, filename, label, "train"))
        ids.append(fid)
    rows, _ = analyze_items(items, args.workers)
    if not rows or not all(row["passed"] for row in rows):
        raise ValueError("Labeled expansion photos must still pass the original quality gates")
    for fid, row in zip(ids, rows, strict=True):
        row.update(ffhq_id=fid, label_source="ai_pseudo_label")
    fields = list(dict.fromkeys(name for row in rows for name in row))
    write_rows(args.out, fields, rows)
    print(f"Wrote training features for {len(rows)} visually accepted pseudo-labels.", flush=True)


def add_identity_arguments(parser):
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--dlib-landmarks", type=Path, required=True)
    parser.add_argument("--dlib-recognizer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    fetch_parser = actions.add_parser("download")
    fetch_parser.add_argument("metadata", type=Path)
    fetch_parser.add_argument("workspace", type=Path)
    fetch_parser.add_argument("--attributes", type=Path, required=True)
    fetch_parser.add_argument("--facebench-manifest", type=Path, required=True)
    fetch_parser.add_argument("--male", type=int, default=650)
    fetch_parser.add_argument("--other", type=int, default=220)
    fetch_parser.add_argument("--workers", type=int, default=12)
    references = actions.add_parser("references")
    references.add_argument("photos", type=Path)
    add_identity_arguments(references)
    screening = actions.add_parser("screen")
    screening.add_argument("workspace", type=Path)
    add_identity_arguments(screening)
    screening.add_argument("--gender-model", type=Path, required=True)
    screening.add_argument("--checkpoint", type=Path, required=True)
    screening.add_argument("--cache", type=Path, required=True)
    screening.add_argument("--review-count", type=int, default=352)
    features = actions.add_parser("features")
    for field in ("manifest", "labels", "images", "out"):
        features.add_argument(f"--{field}", type=Path, required=True)
    features.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if (args.workers < 1 or (args.action == "download" and (min(args.male, args.other) < 0 or not args.male + args.other))
            or (args.action == "screen" and args.review_count < 1)):
        parser.error("Candidate counts and workers must be valid")
    {"download": download, "references": reference_bundle, "screen": screen,
     "features": build_features}[args.action](args)


if __name__ == "__main__":
    main()
