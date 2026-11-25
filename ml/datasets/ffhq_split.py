"""Partition the FFHQ manifest into a 200-photo human test set and a training set."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TEST_SIZE = 200


def write_split(manifest: Path, out: Path) -> None:
    by_license = defaultdict(list)
    with manifest.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_license[row["license"]].append(row["ffhq_id"])

    ids = sorted(ffhq_id for group in by_license.values() for ffhq_id in group)
    if len(ids) != len(set(ids)) or len(ids) <= TEST_SIZE:
        raise ValueError(f"the manifest must contain more than {TEST_SIZE} unique image IDs")

    quotas = {license: divmod(len(group) * TEST_SIZE, len(ids)) for license, group in by_license.items()}
    remaining = TEST_SIZE - sum(count for count, _ in quotas.values())
    by_remainder = sorted(quotas, key=lambda license: (-quotas[license][1], license))
    extra = set(by_remainder[:remaining])
    rng = random.Random(SEED)
    test_ids = {
        ffhq_id
        for license, group in sorted(by_license.items())
        for ffhq_id in rng.sample(sorted(group), quotas[license][0] + (license in extra))
    }

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("ffhq_id", "split"))
        writer.writerows((ffhq_id, "test" if ffhq_id in test_ids else "train") for ffhq_id in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_split(args.manifest, args.out)


if __name__ == "__main__":
    main()
