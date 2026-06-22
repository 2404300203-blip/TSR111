#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

ROOT = Path('/root/autodl-tmp/PaddleOCR')
EXPS = [
    ('E0_baseline', ROOT / 'output/SLANet_plus_exp/E0_baseline'),
    ('E1_local_only', ROOT / 'output/SLANet_plus_exp/E1_local_only'),
    ('E2_mamba_only', ROOT / 'output/SLANet_plus_exp/E2_mamba_only'),
    ('E3_mamba_local_parallel', ROOT / 'output/SLANet_plus_exp/E3_mamba_local_parallel'),
    ('E4_mamba_local_no_gate', ROOT / 'output/SLANet_plus_exp/E4_mamba_local_no_gate'),
    ('E5_fsam_stable', ROOT / 'output/SLANet_plus_exp/E5_fsam_stable'),
]

def read_best(exp_dir: Path):
    state = exp_dir / 'best_accuracy.states'
    if not state.exists():
        return None
    with state.open('rb') as f:
        return pickle.load(f)

def read_last_metric(exp_dir: Path):
    log = exp_dir / 'train.log'
    if not log.exists():
        return None
    cur = None
    best = None
    nan = False
    for line in log.read_text(errors='ignore').splitlines():
        if 'nanxxx' in line:
            nan = True
        if 'cur metric' in line:
            cur = line.strip()
        if 'best metric' in line:
            best = line.strip()
    return {'cur_metric_line': cur, 'best_metric_line': best, 'has_nan': nan}

def main():
    rows = []
    for name, exp_dir in EXPS:
        best = read_best(exp_dir)
        metric = read_last_metric(exp_dir)
        row = {'name': name, 'dir': str(exp_dir), 'exists': exp_dir.exists()}
        if best:
            row.update(best.get('best_model_dict', {}))
            row['epoch'] = best.get('epoch')
            row['global_step'] = best.get('global_step')
        if metric:
            row.update(metric)
        rows.append(row)
    out = ROOT / 'output/SLANet_plus_exp/summary.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
