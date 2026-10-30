#!/usr/bin/env python3
"""Archive server experiment source without datasets, weights or runtime outputs."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-root', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = args.server_root.resolve()
    dest = args.destination.resolve()
    if dest.exists():
        parser.error('destination already exists; choose a new snapshot directory')
    sources = []
    output = root / 'PaddleOCR/output'
    if not output.is_dir():
        parser.error('PaddleOCR/output does not exist under server-root')
    for p in output.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix in {'.py', '.sh', '.yml', '.yaml', '.toml'}:
            sources.append((p, Path('runs') / p.relative_to(output)))
    heads = root / 'PaddleOCR/ppocr/modeling/heads'
    for p in heads.glob('*.py.*'):
        if p.is_file() and not p.is_symlink():
            sources.append((p, Path('history/model_heads') / p.name))
    export = root / 'DBM_SLANet_full_export_20260624'
    for name in ('code', 'configs'):
        for p in (export / name).rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in {'.py', '.sh', '.yml', '.yaml'}:
                sources.append((p, Path('export_20260624') / p.relative_to(export)))
    # Validate the complete selection before creating the destination.
    for p, rel in sources:
        if p.stat().st_size > 2 * 1024 * 1024:
            parser.error('unexpectedly large source file: ' + str(p.relative_to(root)))
        if not p.resolve().is_relative_to(root):
            parser.error('source escapes server-root: ' + str(rel))
    dest.mkdir(parents=True)
    with (dest / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(['source_relative_to_server_root', 'archived_path', 'bytes', 'sha256'])
        for p, rel in sorted(sources, key=lambda item: str(item[1])):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            data = target.read_bytes()
            writer.writerow([p.relative_to(root), rel, len(data), hashlib.sha256(data).hexdigest()])
    print('Archived', len(sources), 'files to', dest)


if __name__ == '__main__':
    main()
