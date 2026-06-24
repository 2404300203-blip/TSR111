# DBM-SLANet Paper Figure Package

Date: 2026-06-24

## Generated Figures

| Figure | File | Recommended Section | Purpose |
|---|---|---|---|
| Method architecture | `figures/dbm_slanet_architecture.png` / `.pdf` | Method | Shows the SLANet visual backbone, cell token features, and full-Mamba structure head. |
| Qualitative cases | `figures/qualitative_cases_multipanel.png` | Experiments / Error Analysis | Shows three improvement cases and two failure cases. |

## Figure 1 Caption Draft

Overall architecture of DBM-SLANet. The model retains the SLANet-style visual feature extraction pipeline and replaces the original structure decoding head with a full-Mamba structure head. The Mamba-enhanced head models long-range dependencies among HTML structure tokens, especially span-sensitive patterns involving `rowspan` and `colspan`.

## Figure 2 Caption Draft

Qualitative examples on the PubTabNet validation split. Panels A-C show cases where Phase5-E corrects structural errors made by the original SLANet baseline, including span-sensitive table headers and cell merging patterns. Panels D-E show failure cases where Phase5-E introduces structural regressions on samples that the baseline predicted exactly. These cases indicate that span prediction remains the main bottleneck.

## Notes

- Use the PDF version of the architecture figure for paper submission when possible.
- Use the PNG multi-panel qualitative figure for quick manuscript drafts and presentations.
