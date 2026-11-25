"""Build a self-contained, offline HTML page for labeling face shapes (Bahasa Indonesia UI).

    python -m ml.datasets.labeling_tool ml/datasets/ffhq_labeling/manifest.csv --images data/raw/ffhq \
        --out data/raw/ffhq_labeling.html

Each annotator opens the page in a browser, labels every image on their own (no sharing labels until done), and
exports a CSV `labels_<nama>.csv`. Images are embedded as small JPEG thumbnails so the page works offline; the page
contains faces, so it stays local (data/raw is gitignored) and must not be uploaded publicly.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import cv2

from ml.classification.rule_based import SHAPE_LABELS_ID, SHAPES

UNSURE = "ragu"
PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:1em}img{max-width:512px;display:block;margin:1em 0}
button{margin:.2em;padding:.5em 1em;font-size:1em}button.sel{background:#2d6cdf;color:#fff}#bar{position:sticky;top:0;background:#fff;padding:.5em 0}</style>
</head><body><h1>__TITLE__</h1>
<p>Baca panduan anotasi (<code>docs/annotation-guide-id.md</code>) sebelum mulai. Label tersimpan otomatis di browser ini.
Jangan melihat label anotator lain sampai Anda selesai.</p>
<div id="bar">Nama anotator: <input id="who" placeholder="nama"> <span id="progress"></span>
<button onclick="go(-1)">&larr; Sebelumnya</button><button onclick="go(1)">Berikutnya &rarr;</button>
<button onclick="exportCsv()">Ekspor CSV</button></div>
<div id="item"></div>
<script>
const ITEMS=__ITEMS__, CHOICES=__CHOICES__;
const KEY=__KEY__;
let i=0, labels={};
const who=document.getElementById("who"); who.value=localStorage.getItem(KEY+"-who")||"";
const name=()=>who.value.trim().replace(/[^A-Za-z0-9_-]+/g,"_");
function load(){labels=name()?JSON.parse(localStorage.getItem(KEY+":"+name())||"{}"):{};}
who.oninput=()=>{localStorage.setItem(KEY+"-who",who.value);load();render();};
load();
function render(){const it=ITEMS[i];
 document.getElementById("progress").textContent=`Gambar ${i+1} dari ${ITEMS.length}, sudah dilabeli ${Object.keys(labels).length}`;
 document.getElementById("item").innerHTML=`<h3>FFHQ ${it.id}</h3><img src="${it.src}"><small>Foto: ${it.credit}</small><div>`+
  CHOICES.map(([v,t],k)=>`<button class="${labels[it.id]===v?"sel":""}" onclick="pick('${v}')">${k+1}. ${t}</button>`).join("")+"</div>";}
function pick(v){if(!name()){alert("Isi nama anotator dulu.");return;}
 labels[ITEMS[i].id]=v;localStorage.setItem(KEY+":"+name(),JSON.stringify(labels));go(1);}
function go(d){i=Math.max(0,Math.min(ITEMS.length-1,i+d));render();}
document.onkeydown=e=>{if(e.target===who)return;const n=parseInt(e.key);if(n>=1&&n<=CHOICES.length)pick(CHOICES[n-1][0]);
 if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1);};
function exportCsv(){if(!name()){alert("Isi nama anotator dulu.");return;}
 const rows=["ffhq_id,annotator,label"].concat(ITEMS.filter(it=>labels[it.id]).map(it=>`${it.id},${name()},${labels[it.id]}`));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));
 a.download=`labels_${name()}.csv`;a.click();}
render();
</script></body></html>"""


def choices() -> list[tuple[str, str]]:
    return [(s, SHAPE_LABELS_ID[s]) for s in SHAPES] + [(UNSURE, "Ragu / tidak bisa ditentukan")]


def thumbnail_data_uri(path: Path, size: int = 512) -> str:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_page(manifest: Path, images: Path, test_split: Path | None = None) -> str:
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    title, key = "Pelabelan Bentuk Wajah", "faceshape-labels"
    if test_split is not None:
        with test_split.open(encoding="utf-8", newline="") as f:
            split = list(csv.DictReader(f))
        manifest_ids = {r["ffhq_id"] for r in rows}
        split_ids = {r["ffhq_id"] for r in split}
        if (len(manifest_ids) != len(rows) or len(split_ids) != len(split) or split_ids != manifest_ids
                or {r["split"] for r in split} != {"train", "test"}):
            raise ValueError("the split must assign each manifest ID exactly once to train or test, with both sets nonempty")
        test_ids = {r["ffhq_id"] for r in split if r["split"] == "test"}
        rows = [r for r in rows if r["ffhq_id"] in test_ids]
        title += f" — Set Uji ({len(rows)} Foto)"
        key += "-test-" + hashlib.sha256(json.dumps(sorted(test_ids)).encode()).hexdigest()
    items = [{"id": r["ffhq_id"], "src": thumbnail_data_uri(images / r["local_file"]),
              "credit": html.escape(f"{r['author']} ({r['license']})")} for r in rows]
    return (PAGE.replace("__TITLE__", title).replace("__KEY__", json.dumps(key))
            .replace("__CHOICES__", json.dumps(choices())).replace("__ITEMS__", json.dumps(items), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--test-split", type=Path, help="partition CSV; include only images assigned to test")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_page(args.manifest, args.images, args.test_split), encoding="utf-8")


if __name__ == "__main__":
    main()
