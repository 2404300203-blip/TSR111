# TSR: Table Structure Recognition Experiments

This repository is a PaddleOCR-based experimental workspace for table structure recognition (TSR). It keeps the upstream PaddleOCR codebase as the foundation and adds a set of table-focused experiments around SLANet, OCR-cell matching, local/global feature fusion, and cell-level supervision.

The uploaded repository is a code snapshot from:

```text
/root/autodl-tmp/PaddleOCR
```

Large datasets, model checkpoints, and training outputs are intentionally excluded from Git so that the repository remains usable on GitHub.

## Project Goal

The project explores how to improve table structure recognition, especially for financial and scientific tables where empty cells, OCR-cell alignment errors, and weak local structure modeling often hurt HTML reconstruction quality.

The main experimental target is to improve SLANet-style table recognition while keeping baseline behavior reproducible.

## Main Experiment Lines

### Phase 0: Code Audit

Located the key components involved in table prediction:

- table matcher
- HTML reconstruction
- table inference entry points
- evaluation scripts
- data structures such as `dt_boxes`, `rec_res`, `structure_res`, `cell_bbox`, and HTML tokens

### Phase 1: OCR-Aware Cell Matching

Added an optional OCR-aware matching path while preserving the legacy matcher by default.

Key files:

- `ppstructure/table/matcher.py`
- `ppstructure/table/predict_table.py`
- `ppstructure/utility.py`
- `tests/test_ocr_aware_table_matcher.py`
- `scripts/eval_table_teds_codex.py`
- `eval_table_teds_codex.py`

Main idea:

- keep `match_mode=legacy` as the default
- add `match_mode=ocr_aware_hungarian`
- score OCR-cell pairs using bbox normalization, IoU, OCR coverage, cell coverage, center similarity, inside score, direction prior, and OCR confidence
- use Hungarian matching for primary assignment
- support one-cell-to-many-OCR aggregation after primary matching

### Phase 2: Cell Token MLP Head

Added a lightweight auxiliary cell-token prediction head for cell presence/content supervision.

Key files:

- `ppocr/modeling/heads/cell_token_mlp_head.py`
- `ppocr/losses/table_att_loss.py`
- `configs/table/DBM_SLANet_cell_token_mlp_codex.yml`
- `tests/test_cell_token_mlp_head.py`

### Phase 3: Dual-Branch Mamba-Style Head

Added a dependency-free dual-branch sequence modeling head for table cell tokens.

Key files:

- `ppocr/modeling/heads/cell_dual_branch_mamba_head.py`
- `configs/table/DBM_SLANet_dual_branch_mamba_codex.yml`
- `tests/test_cell_dual_branch_mamba_head.py`

Note: the current implementation uses a lightweight gated recurrent/scan-style module with a Mamba-like interface. It is not a full selective-scan Mamba implementation.

## Additional Experimental Configs

This repository also includes several SLANet/SLANet+ experimental configs under:

```text
configs/table/
```

Examples:

- `SLANet_pubtabnet_loc_codex.yml`
- `SLANet_pubtabnet_lsnet_codex.yml`
- `SLANet_pubtabnet_fsam_codex.yml`
- `SLANet_plus_E0_baseline_codex.yml`
- `SLANet_plus_E3_mamba_local_parallel_codex.yml`
- `SLANet_plus_E5_fsam_stable_codex.yml`

## What Is Not Included

The following large local artifacts are excluded:

- `train_data/`
- `output/`
- `inference/`
- `inference_results/`
- model checkpoints such as `*.pdparams`, `*.pdopt`, `*.pdstates`
- large annotation/data files such as `*.jsonl`
- archives such as `*.zip`, `*.tar.gz`

These files existed on the remote server during experiments, but they are not suitable for direct storage in a normal GitHub repository.

## Useful Commands

Run focused unit tests:

```bash
python -m pytest tests/test_ocr_aware_table_matcher.py
python -m pytest tests/test_cell_token_mlp_head.py
python -m pytest tests/test_cell_dual_branch_mamba_head.py
```

Check tracked repository size:

```bash
git ls-files -z | xargs -0 du -ch | tail -1
git ls-files -z | xargs -0 du -h | sort -hr | head -30
```

Push from the remote server through GitHub SSH-over-443:

```bash
git remote set-url tsr ssh://git@ssh.github.com:443/2404300203-blip/TSR.git
git push -u tsr tsr-code-only:main --force
```

## Upstream

This project is based on PaddleOCR:

```text
https://github.com/PaddlePaddle/PaddleOCR
```

The original PaddleOCR license is kept in `LICENSE`.
