# DBM-SLANet Phase4 and Phase5 Experiment Summary

Date: 2026-06-23
Server path: `/root/autodl-tmp/PaddleOCR`

## Current Phase4 Best

The current best checkpoint is:

```text
output/E4_lite_stage6_lr1e5_from_stage5_ft_30000_20260623/best_accuracy.pdparams
```

External validation on 1000 samples:

```text
Structure-TEDS: 0.8294241307122223
structure_acc: 0.691
fps: 12.000899824853926
```

## Phase4 Experiment Table

| Experiment | Init | Train samples | LR | Internal best acc | External Structure-TEDS | Decision |
|---|---|---:|---:|---:|---:|---|
| Stage4 baseline | previous lite setup | 10000 | baseline | 0.684 | 0.8187506322400308 | baseline |
| Stage5 lr=1e-4 | Stage4 best | 30000 | 1e-4 | ~0.673 | 0.8173615266483486 | rejected |
| Stage5 lr=5e-5 | Stage4 best | 30000 | 5e-5 | ~0.677 | not run | stopped early |
| Stage5 lr=2e-5 | Stage4 best | 30000 | 2e-5 | 0.686999999313 | 0.828760583423412 | improved |
| Stage6 lr=1e-5 | Stage5 best | 30000 | 1e-5 | 0.690999999309 | 0.8294241307122223 | current best |
| Stage7 lr=5e-6 | Stage6 best | 30000 | 5e-6 | 0.685999999314 | not run | stopped early |

## Phase4 Conclusion

Low-learning-rate continuation improved the lite full-Mamba head. The best validated path is:

```text
Stage4 baseline -> Stage5 lr=2e-5 -> Stage6 lr=1e-5
```

Further simple LR reduction to `5e-6` did not improve internal validation and was stopped at around 6000 steps.

## Phase5 Start Plan

Phase5 moves from lite Mamba to stronger Full Mamba capacity:

- Head: `CellFullMambaHead`
- Full setting: `mamba_d_state=16`, `mamba_expand=2`, `bidirectional=true`
- First run: 500-sample smoke/stability run
- Init: Phase4 Stage6 best checkpoint where shape-compatible
- Goal: verify forward/backward stability, memory, validation behavior
- Baseline to beat later: Stage6 Structure-TEDS `0.8294241307122223`

## Phase5 Experiment Table

Phase5 uses the stronger Full Mamba setting:

```yaml
Head:
  name: CellFullMambaHead
  mamba_d_state: 16
  mamba_expand: 2
  bidirectional: true
```

| Experiment | Init | Train samples | LR | Internal best acc | External Structure-TEDS | Decision |
|---|---|---:|---:|---:|---:|---|
| Phase5-A smoke | Phase4 Stage6 best | 500 | smoke | 0.5799999884 | not run | stable |
| Phase5-B warmup | Phase5-A best | 2000 | warmup | 0.64999999675 | not run | stable |
| Phase5-C scale-up | Phase5-B best | 10000 | scale-up | 0.681999998636 | not run | stable |
| Phase5-D lr=1e-5 | Phase5-C best | 30000 | 1e-5 | 0.692999999307 | 0.8306299579222143 | improved |
| Phase5-E lr=5e-6 | Phase5-D best | 30000 | 5e-6 | 0.692999999307 | 0.8308172189543931 | current best |
| Phase5-F lr=2e-6 | Phase5-E best | 30000 | 2e-6 | 0.693999999306 | 0.8285197072732923 | rejected |

## Current Overall Best

The current best checkpoint is:

```text
output/E5_full_mamba_stage5_30000_lr5e6_from_stage4_20260623/best_accuracy.pdparams
```

External validation on 1000 samples:

```text
Structure-TEDS: 0.8308172189543931
structure_acc: 0.693
fps: 9.565019614337581
```

Compared with the previous Phase4 lite best:

```text
TEDS gain: 0.8308172189543931 - 0.8294241307122223 = +0.0013930882421708
structure_acc gain: 0.693 - 0.691 = +0.002
```

Compared with Phase5-D:

```text
TEDS gain: 0.8308172189543931 - 0.8306299579222143 = +0.0001872610321788
structure_acc: tied at 0.693
```

Compared with Phase5-F:

```text
TEDS delta: 0.8285197072732923 - 0.8308172189543931 = -0.0022975116811008
structure_acc delta: 0.694 - 0.693 = +0.001
```

## Phase5 Conclusion

The Full Mamba head is stable on the 4090D setup and improves over the Phase4 lite Mamba best. The best path so far is:

```text
Phase4 Stage6 lite best -> Phase5-A -> Phase5-B -> Phase5-C -> Phase5-D -> Phase5-E
```

Phase5-E only gives a small improvement over Phase5-D, but it is the best validated model so far. Phase5-F shows that higher internal exact-match accuracy does not necessarily improve TEDS: structure_acc increased to `0.694`, while external TEDS dropped to `0.8285197072732923`.

## Recommended Next Step

Move to Phase6 error analysis instead of continuing blind low-LR training:

```text
Phase6
compare: Phase5-E current best vs Phase5-F rejected model
diagnostics: per-sample TEDS, exact match, predicted structure tokens, ground-truth tokens
goal: identify why Phase5-F improves structure_acc but loses TEDS
next fix target: token ordering, span-related structure errors, or loss/selection mismatch
```
