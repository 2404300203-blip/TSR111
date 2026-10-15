# Local Folder Organization

Date: 2026-06-24

## Current Layout

| Path | Purpose |
|---|---|
| `reports/phase10/` | Phase10 experiment summaries and paper-ready result tables |
| `reports/remote/` | Remote repository README drafts and remote-side notes |
| `scripts/eval_phase8_9/` | Phase8/Phase9 evaluation and reranking scripts |
| `data/pubtabnet/` | Large local PubTabNet annotation/archive files |
| `packages/` | Model or experiment archive packages |
| `scratch/` | Local temporary files, editor swap files, caches, and macOS metadata |
| `papers/` | Reference PDFs |
| `visualizations/` | Existing visualization samples |
| `remote_patch_work/` | Earlier remote patch workspace; preserved as-is except for pre-existing git status |
| `configs/`, `ppocr/` | Newly copied local implementation/config fragments |

## Moved Files

### Phase10 Reports

Moved to `reports/phase10/`:

```text
PAPER_READY_EXPERIMENT_TABLES_20260624.md
PHASE10_LARGE_VAL_3000_RESULTS_20260624.md
PHASE10_LARGE_VAL_5000_RESULTS_20260624.md
PHASE10_QUALITATIVE_CASE_ANALYSIS_20260624.md
PHASE10_STATISTICAL_VALIDATION_20260624.md
```

### Remote README Notes

Moved to `reports/remote/`:

```text
README_CN_REMOTE_TSR.md
README_REMOTE_TSR.md
```

### Evaluation Scripts

Moved to `scripts/eval_phase8_9/`:

```text
eval_table_teds_phase8_calibrated_codex.py
eval_table_teds_phase9_topk_rerank_codex.py
eval_table_teds_phase9c_cellpattern_rerank_codex.py
eval_table_teds_phase9d_gated_rerank_codex.py
```

### Large Data and Packages

Moved to `data/pubtabnet/`:

```text
pubtabnet_part_91150(unitable版本).jsonl
archive_zip_folder/pubtabnet_part_91150.zip/
```

Moved to `packages/`:

```text
baseline_package_20260621.tar.gz
```

### Temporary Files

Moved to `scratch/`:

```text
scratch/editor_backups/.DBM_SLANet_Codex_Implementation_Roadmap.md.swp
scratch/cache/__pycache__/
scratch/mac_metadata/.DS_Store
```

## Git Notes

The following were intentionally not modified or reverted:

```text
remote_patch_work/ppocr/modeling/heads/__init__.py
```

Git still shows several untracked implementation fragments:

```text
configs/
ppocr/
remote_patch_work/configs/table/DBM_SLANet_full_mamba_codex.yml
remote_patch_work/ppocr/modeling/heads/cell_full_mamba_head.py
remote_patch_work/tests/test_cell_full_mamba_head.py
scripts/eval_phase8_9/
```

These appear to be local code/config artifacts rather than disposable clutter, so they were preserved.

## Suggested Next Cleanup

If this local folder should become a clean paper workspace, decide whether to:

1. Keep `remote_patch_work/` as historical implementation notes.
2. Move new code fragments under `remote_patch_work/` or `code_snapshots/`.
3. Add a top-level `paper/` folder for manuscript drafts, figures, and tables.
4. Keep large files under `data/` and `packages/`, which are already ignored by `.gitignore`.
