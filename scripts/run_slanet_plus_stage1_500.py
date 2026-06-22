#!/usr/bin/env python3
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/autodl-tmp/PaddleOCR')
PYTHON_BIN = os.environ.get('PYTHON_BIN', '/root/miniconda3/envs/slanet/bin/python')
EXPERIMENTS = [
    ('E3_mamba_local_parallel', 'configs/table/SLANet_plus_E3_mamba_local_parallel_codex.yml'),
    ('E1_local_only', 'configs/table/SLANet_plus_E1_local_only_codex.yml'),
    ('E2_mamba_only', 'configs/table/SLANet_plus_E2_mamba_only_codex.yml'),
]


def output_dir_for_config(cfg: Path) -> Path:
    for line in cfg.read_text().splitlines():
        if line.strip().startswith('save_model_dir:'):
            return ROOT / line.split(':', 1)[1].strip().lstrip('./')
    raise RuntimeError(f'No save_model_dir in {cfg}')


def run_until_first_eval(name: str, cfg_rel: str) -> int:
    cfg = ROOT / cfg_rel
    out_dir = output_dir_for_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_log = out_dir / 'stage1_500step.log'
    cmd = [PYTHON_BIN, 'tools/train.py', '-c', str(cfg)]
    print(f'[{name}] start: {" ".join(cmd)} -> {stage_log}', flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    saw_step_500 = False
    saw_eval_after_500 = False
    saw_nan = False
    with stage_log.open('w') as log:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
            if 'nanxxx' in line.lower() or 'loss: nan' in line.lower():
                saw_nan = True
            if 'global_step: 500' in line:
                saw_step_500 = True
            if saw_step_500 and 'cur metric' in line:
                saw_eval_after_500 = True
                print(f'[{name}] first eval after 500-step reached; stopping process.', flush=True)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                break
            if saw_nan:
                print(f'[{name}] nan detected; stopping process.', flush=True)
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                break
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
    status = 'nan' if saw_nan else 'first_eval_done' if saw_eval_after_500 else f'exit_{proc.returncode}'
    (out_dir / 'stage1_status.txt').write_text(status + '\n')
    print(f'[{name}] status: {status}', flush=True)
    return 1 if saw_nan else 0


def main() -> int:
    os.chdir(ROOT)
    rc = 0
    for name, cfg in EXPERIMENTS:
        rc = max(rc, run_until_first_eval(name, cfg))
    subprocess.run([PYTHON_BIN, 'scripts/summarize_slanet_plus_results.py'], cwd=str(ROOT), check=False)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
