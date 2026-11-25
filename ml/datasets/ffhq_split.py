"""Use the first 104 manifest rows for testing, matching human labels from the 600-photo page."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TEST_SIZE = 104


def write_split(manifest: Path, out: Path) -> None:
    with manifest.open(encoding="utf-8", newline="") as f:
        ids = [row["ffhq_id"] for row in csv.DictReader(f)]

    if len(ids) != len(set(ids)) or len(ids) <= TEST_SIZE:
        raise ValueError(f"the manifest must contain more than {TEST_SIZE} unique image IDs")

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("ffhq_id", "split"))
        writer.writerows((ffhq_id, "test" if i < TEST_SIZE else "train") for i, ffhq_id in enumerate(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_split(args.manifest, args.out)


if __name__ == "__main__":
    main()
