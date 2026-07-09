#!/usr/bin/env python3
"""Extract selected FinTabNet PDFs from a full Kaggle dataset zip.

Kaggle single-file download currently returns 404 for nested FinTabNet PDF paths
in this environment. This script avoids full extraction by reading the downloaded
fintabnet.zip and extracting only files listed as COMPANY/YEAR/page_N.pdf.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def read_list(path: Path) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--lists", nargs="+", required=True)
    parser.add_argument("--pdf-root", default="train_data/table/fintabnet_processed/pdf_cache")
    parser.add_argument("--prefix", default="fintabnet/pdf/")
    parser.add_argument("--manifest", default="output/fintabnet_c_official_subset_plan_20260709/extract_manifest.json")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    pdf_root = Path(args.pdf_root)
    pdf_root.mkdir(parents=True, exist_ok=True)

    wanted = []
    seen = set()
    for lp in args.lists:
        for rel in read_list(Path(lp)):
            if rel not in seen:
                seen.add(rel)
                wanted.append(rel)

    cached_before = [rel for rel in wanted if (pdf_root / rel).exists() and (pdf_root / rel).stat().st_size > 0]
    to_extract = [rel for rel in wanted if rel not in set(cached_before)]

    extracted = []
    missing_in_zip = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for i, rel in enumerate(to_extract, start=1):
            member = args.prefix + rel
            out_path = pdf_root / rel
            if member not in names:
                missing_in_zip.append(rel)
                print(f"[{i}/{len(to_extract)}] missing-in-zip {member}")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out_path.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted.append(rel)
            print(f"[{i}/{len(to_extract)}] extracted {rel}")

    manifest = {
        "zip": str(zip_path),
        "pdf_root": str(pdf_root),
        "wanted": len(wanted),
        "cached_before": len(cached_before),
        "to_extract": len(to_extract),
        "extracted": len(extracted),
        "missing_in_zip": len(missing_in_zip),
        "missing_in_zip_items": missing_in_zip[:200],
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
