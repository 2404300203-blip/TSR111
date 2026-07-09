#!/usr/bin/env python3
"""Download FinTabNet PDF files listed by the official subset plan.

The input files contain paths like COMPANY/YEAR/page_N.pdf. This script downloads
those files from Kaggle dataset jiongjiong/fintabnet into the PaddleOCR PDF cache.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def read_list(path: Path) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                rows.append(item)
    return rows


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lists", nargs="+", required=True, help="missing pdf list files")
    parser.add_argument("--pdf-root", default="train_data/table/fintabnet_processed/pdf_cache")
    parser.add_argument("--dataset", default="jiongjiong/fintabnet")
    parser.add_argument("--kaggle-bin", default="/root/miniconda3/bin/kaggle")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    pdf_root.mkdir(parents=True, exist_ok=True)

    wanted = []
    seen = set()
    for list_path in args.lists:
        for rel in read_list(Path(list_path)):
            if rel not in seen:
                seen.add(rel)
                wanted.append(rel)
    if args.limit:
        wanted = wanted[: args.limit]

    ok = skipped = failed = 0
    failed_items = []
    for idx, rel in enumerate(wanted, start=1):
        target = pdf_root / rel
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            print(f"[{idx}/{len(wanted)}] cached {rel}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.kaggle_bin,
            "datasets",
            "download",
            "-d",
            args.dataset,
            "-f",
            f"fintabnet/pdf/{rel}",
            "--force",
            "-p",
            str(target.parent),
        ]
        success = False
        last_output = ""
        for attempt in range(1, args.retries + 1):
            result = run(cmd)
            last_output = result.stdout[-2000:]
            if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
                success = True
                break
            print(f"[{idx}/{len(wanted)}] retry {attempt}/{args.retries} failed for {rel}: rc={result.returncode}")
            if last_output:
                print(last_output)
            time.sleep(args.sleep * attempt)
        if success:
            ok += 1
            print(f"[{idx}/{len(wanted)}] downloaded {rel}")
        else:
            failed += 1
            failed_items.append(rel)
            print(f"[{idx}/{len(wanted)}] FAILED {rel}")
        time.sleep(args.sleep)

    print({"total": len(wanted), "downloaded": ok, "cached": skipped, "failed": failed})
    if failed_items:
        fail_path = Path("output/fintabnet_c_official_subset_plan_20260709/download_failed_pdfs.txt")
        fail_path.write_text("\n".join(failed_items) + "\n", encoding="utf-8")
        print(f"failed list: {fail_path}")


if __name__ == "__main__":
    main()
