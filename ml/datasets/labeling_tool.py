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
import html
import json
from pathlib import Path

import cv2

from ml.classification.rule_based import SHAPE_LABELS_ID, SHAPES

UNSURE = "ragu"
PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Pelabelan Bentuk Wajah</title>
<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:1em}img{max-width:512px;display:block;margin:1em 0}
button{margin:.2em;padding:.5em 1em;font-size:1em}button.sel{background:#2d6cdf;color:#fff}#bar{position:sticky;top:0;background:#fff;padding:.5em 0}</style>
</head><body><h1>Pelabelan Bentuk Wajah</h1>
<p>Baca panduan anotasi (<code>docs/annotation-guide-id.md</code>) sebelum mulai. Label tersimpan otomatis di browser ini.
Jangan melihat label anotator lain sampai Anda selesai.</p>
<div id="bar">Nama anotator: <input id="who" placeholder="nama"> <span id="progress"></span>
<button onclick="go(-1)">&larr; Sebelumnya</button><button onclick="go(1)">Berikutnya &rarr;</button>
<button onclick="exportCsv()">Ekspor CSV</button></div>
<div id="item"></div>
<script>
const ITEMS=__ITEMS__, CHOICES=__CHOICES__;
const KEY="faceshape-labels", labels=JSON.parse(localStorage.getItem(KEY)||"{}");
let i=0;
const who=document.getElementById("who"); who.value=localStorage.getItem(KEY+"-who")||"";
who.oninput=()=>localStorage.setItem(KEY+"-who",who.value);
function render(){const it=ITEMS[i];
 document.getElementById("progress").textContent=`Gambar ${i+1} dari ${ITEMS.length}, sudah dilabeli ${Object.keys(labels).length}`;
 document.getElementById("item").innerHTML=`<h3>FFHQ ${it.id}</h3><img src="${it.src}"><small>Foto: ${it.credit}</small><div>`+
  CHOICES.map(([v,t],k)=>`<button class="${labels[it.id]===v?"sel":""}" onclick="pick('${v}')">${k+1}. ${t}</button>`).join("")+"</div>";}
function pick(v){labels[ITEMS[i].id]=v;localStorage.setItem(KEY,JSON.stringify(labels));go(1);}
function go(d){i=Math.max(0,Math.min(ITEMS.length-1,i+d));render();}
document.onkeydown=e=>{if(e.target===who)return;const n=parseInt(e.key);if(n>=1&&n<=CHOICES.length)pick(CHOICES[n-1][0]);
 if(e.key==="ArrowRight")go(1);if(e.key==="ArrowLeft")go(-1);};
function exportCsv(){if(!who.value){alert("Isi nama anotator dulu.");return;}
 const rows=["ffhq_id,annotator,label"].concat(ITEMS.filter(it=>labels[it.id]).map(it=>`${it.id},${who.value},${labels[it.id]}`));
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));
 a.download=`labels_${who.value}.csv`;a.click();}
render();
</script></body></html>"""


def choices() -> list[tuple[str, str]]:
    return [(s, SHAPE_LABELS_ID[s]) for s in SHAPES] + [(UNSURE, "Ragu / tidak bisa ditentukan")]


def thumbnail_data_uri(path: Path, size: int = 512) -> str:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_page(manifest: Path, images: Path) -> str:
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    items = [{"id": r["ffhq_id"], "src": thumbnail_data_uri(images / r["local_file"]),
              "credit": html.escape(f"{r['author']} ({r['license']})")} for r in rows]
    return PAGE.replace("__ITEMS__", json.dumps(items)).replace("__CHOICES__", json.dumps(choices()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_page(args.manifest, args.images), encoding="utf-8")


if __name__ == "__main__":
    main()
