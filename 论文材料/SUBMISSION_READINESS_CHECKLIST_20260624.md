# DBM-SLANet Submission Readiness Checklist

Date: 2026-06-24

## Current Readiness

Estimated readiness for an SCI-3-oriented manuscript: 75%.

## Completed

- [x] Baseline SLANet evaluation.
- [x] Lite Mamba and full Mamba ablations.
- [x] Main Phase5-E full Mamba checkpoint selected.
- [x] 3000-sample validation.
- [x] 5000-sample validation.
- [x] Paired bootstrap statistical validation.
- [x] Negative ablations for span-aware loss and reranking.
- [x] Qualitative case mining.
- [x] Five representative qualitative figures.
- [x] Paper material folder organized.

## Still Needed

- [ ] Method architecture figure.
- [ ] Final multi-panel qualitative figure.
- [ ] More complete related-work citations.
- [ ] Final experiment table formatted for manuscript.
- [ ] Full English manuscript draft.
- [ ] Optional full 9115 validation evaluation.
- [ ] Optional second dataset or stronger external comparison.

## Risk Assessment

| Risk | Level | Mitigation |
|---|---|---|
| Only one dataset family | Medium | Add full 9115 validation or another dataset if possible |
| Speed drop vs baseline | Medium | Present as accuracy-oriented, report FPS honestly |
| Span regressions remain | Medium | Use qualitative analysis and future-work discussion |
| Novelty may be challenged | Medium | Emphasize Mamba structural decoding and statistical validation |
| Phase9-C not robust | Low | Report as negative ablation, not final method |

## Recommended Next Actions

1. Create method architecture figure.
2. Create one final qualitative figure with 3 success and 2 failure cases.
3. Build manuscript experiment section from `FINAL_PAPER_EXPERIMENT_TABLES_20260624.md`.
4. Start English draft using `PAPER_WRITING_BLUEPRINT_20260624.md`.
