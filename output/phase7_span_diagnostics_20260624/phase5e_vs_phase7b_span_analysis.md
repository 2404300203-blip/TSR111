# Phase7-B Span-Aware Loss Diagnostic Analysis

Comparison: Phase5-E current best vs Phase7-B weak span-aware loss on the same 1000-sample validation split.

## Summary

- Samples: 1000
- Same TEDS: 936
- Phase7-B worse: 35, mean delta -0.300894
- Phase7-B better: 29, mean delta 0.266095
- Overall mean delta: -0.002814536326
- Exact matches: Phase5-E 693, Phase7-B 697

## Span Count Summary

| Metric | Phase5-E | Phase7-B |
|---|---:|---:|
| Samples with extra predicted spans | 94 | 91 |
| Total extra predicted spans | 456 | 475 |
| Samples with missing spans | 127 | 131 |
| Total missing spans | 446 | 448 |

## Worst Phase7-B Regressions

| sample | delta 7B-5E | 5E TEDS | 7B TEDS | 5E exact | 7B exact | len 5E/7B/GT | spans 5E/7B/GT | filename |
|---:|---:|---:|---:|---|---|---:|---:|---|
| 964 | -0.897436 | 1.000000 | 0.102564 | True | False | 46/48/46 | 0/1/0 | PMC5986424_004_00.png |
| 776 | -0.863014 | 1.000000 | 0.136986 | True | False | 89/83/89 | 4/1/4 | PMC5880597_006_00.png |
| 685 | -0.722222 | 1.000000 | 0.277778 | True | False | 43/45/43 | 2/3/2 | PMC3466411_005_00.png |
| 756 | -0.657895 | 1.000000 | 0.342105 | True | False | 89/91/89 | 4/2/4 | PMC6001162_004_01.png |
| 474 | -0.654545 | 0.981818 | 0.327273 | False | False | 381/386/388 | 3/2/3 | PMC3752638_005_00.png |
| 27 | -0.490566 | 1.000000 | 0.509434 | True | False | 61/65/61 | 3/5/3 | PMC4825282_002_01.png |
| 934 | -0.480745 | 0.828571 | 0.347826 | False | False | 84/76/82 | 4/0/3 | PMC4082174_005_00.png |
| 182 | -0.477580 | 0.838235 | 0.360656 | False | False | 156/60/140 | 0/0/0 | PMC3520051_012_00.png |
| 496 | -0.448276 | 0.448276 | 0.000000 | False | False | 78/79/71 | 6/7/2 | PMC5913313_005_00.png |
| 55 | -0.430556 | 0.430556 | 0.000000 | False | False | 186/182/190 | 16/9/16 | PMC5106839_003_00.png |
| 198 | -0.407407 | 1.000000 | 0.592593 | False | False | 111/121/111 | 1/1/1 | PMC4494372_006_00.png |
| 45 | -0.359031 | 0.681698 | 0.322667 | False | False | 501/500/495 | 94/93/92 | PMC4262998_003_00.png |
| 637 | -0.348936 | 1.000000 | 0.651064 | True | False | 282/272/282 | 9/10/9 | PMC6031117_005_00.png |
| 143 | -0.342995 | 0.342995 | 0.000000 | False | False | 246/310/206 | 13/41/11 | PMC6055123_004_00.png |
| 636 | -0.333333 | 1.000000 | 0.666667 | True | False | 144/140/144 | 6/5/6 | PMC4016920_002_00.png |

## Best Phase7-B Improvements

| sample | delta 7B-5E | 5E TEDS | 7B TEDS | 5E exact | 7B exact | len 5E/7B/GT | spans 5E/7B/GT | filename |
|---:|---:|---:|---:|---|---|---:|---:|---|
| 611 | 0.907692 | 0.092308 | 1.000000 | False | True | 378/392/392 | 13/26/26 | PMC5880068_006_00.png |
| 907 | 0.684823 | 0.104651 | 0.789474 | False | False | 501/67/53 | 0/0/0 | PMC6023382_029_00.png |
| 50 | 0.621212 | 0.378788 | 1.000000 | False | True | 231/229/229 | 10/9/9 | PMC3824233_004_00.png |
| 316 | 0.611111 | 0.388889 | 1.000000 | False | True | 82/81/81 | 2/2/2 | PMC2974660_007_00.png |
| 951 | 0.602410 | 0.397590 | 1.000000 | False | True | 97/94/94 | 3/0/0 | PMC4143575_002_00.png |
| 342 | 0.589474 | 0.410526 | 1.000000 | False | True | 110/106/106 | 3/4/4 | PMC3580640_006_00.png |
| 284 | 0.464286 | 0.535714 | 1.000000 | False | True | 36/38/38 | 0/1/1 | PMC3259039_001_00.png |
| 324 | 0.371795 | 0.461538 | 0.833333 | False | False | 52/20/24 | 0/0/0 | PMC4052172_001_00.png |
| 634 | 0.321429 | 0.678571 | 1.000000 | False | False | 103/95/95 | 8/4/4 | PMC2435222_002_00.png |
| 457 | 0.314655 | 0.480620 | 0.795276 | False | False | 148/144/146 | 4/2/4 | PMC5427534_005_00.png |

## Token-Level Examples

### Worst regressions

#### sample 964: PMC5986424_004_00.png

- delta 7B-5E: -0.897436
- TEDS 5E/7B: 1.000000 / 0.102564
- counts 5E: `{'len': 46, 'tr': 7, 'td_open': 28, 'td_close': 0, 'empty_td': 28, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- counts 7B: `{'len': 48, 'tr': 7, 'td_open': 27, 'td_close': 1, 'empty_td': 26, 'rowspan': 0, 'colspan': 1, 'span_total': 1}`
- counts GT: `{'len': 46, 'tr': 7, 'td_open': 28, 'td_close': 0, 'empty_td': 28, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- first 5E-vs-7B diff: `replace` 5E[4:6] vs 7B[4:8]
  - 5E context: `<thead> <tr> <td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td>`
  - 7B context: `<thead> <tr> <td></td> <td></td> <td  colspan="2" > </td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td>`
- first 7B-vs-GT diff: `replace` 7B[4:8] vs GT[4:6]
  - 7B context: `<thead> <tr> <td></td> <td></td> <td  colspan="2" > </td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td>`
  - GT context: `<thead> <tr> <td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td>`

#### sample 776: PMC5880597_006_00.png

- delta 7B-5E: -0.863014
- TEDS 5E/7B: 1.000000 / 0.136986
- counts 5E: `{'len': 89, 'tr': 11, 'td_open': 51, 'td_close': 4, 'empty_td': 47, 'rowspan': 3, 'colspan': 1, 'span_total': 4}`
- counts 7B: `{'len': 83, 'tr': 11, 'td_open': 54, 'td_close': 1, 'empty_td': 53, 'rowspan': 0, 'colspan': 1, 'span_total': 1}`
- counts GT: `{'len': 89, 'tr': 11, 'td_open': 51, 'td_close': 4, 'empty_td': 47, 'rowspan': 3, 'colspan': 1, 'span_total': 4}`
- first 5E-vs-7B diff: `replace` 5E[2:10] vs 7B[2:4]
  - 5E context: `<thead> <tr> <td  rowspan="2" > </td> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  rowspan="2" > </td>`
  - 7B context: `<thead> <tr> <td></td> <td></td> <td  colspan="2" > </td> <td></td> </tr> <tr> <td></td>`
- first 7B-vs-GT diff: `replace` 7B[2:4] vs GT[2:10]
  - 7B context: `<thead> <tr> <td></td> <td></td> <td  colspan="2" > </td> <td></td> </tr> <tr> <td></td>`
  - GT context: `<thead> <tr> <td  rowspan="2" > </td> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  rowspan="2" > </td>`

#### sample 685: PMC3466411_005_00.png

- delta 7B-5E: -0.722222
- TEDS 5E/7B: 1.000000 / 0.277778
- counts 5E: `{'len': 43, 'tr': 5, 'td_open': 23, 'td_close': 2, 'empty_td': 21, 'rowspan': 0, 'colspan': 2, 'span_total': 2}`
- counts 7B: `{'len': 45, 'tr': 5, 'td_open': 22, 'td_close': 3, 'empty_td': 19, 'rowspan': 1, 'colspan': 2, 'span_total': 3}`
- counts GT: `{'len': 43, 'tr': 5, 'td_open': 23, 'td_close': 2, 'empty_td': 21, 'rowspan': 0, 'colspan': 2, 'span_total': 2}`
- first 5E-vs-7B diff: `replace` 5E[2:3] vs 7B[2:6]
  - 5E context: `<thead> <tr> <td></td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
  - 7B context: `<thead> <tr> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
- first 7B-vs-GT diff: `replace` 7B[2:6] vs GT[2:3]
  - 7B context: `<thead> <tr> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
  - GT context: `<thead> <tr> <td></td> <td  colspan="2" > </td> <td  colspan="2" > </td>`

#### sample 756: PMC6001162_004_01.png

- delta 7B-5E: -0.657895
- TEDS 5E/7B: 1.000000 / 0.342105
- counts 5E: `{'len': 89, 'tr': 12, 'td_open': 49, 'td_close': 4, 'empty_td': 45, 'rowspan': 2, 'colspan': 2, 'span_total': 4}`
- counts 7B: `{'len': 91, 'tr': 12, 'td_open': 57, 'td_close': 2, 'empty_td': 55, 'rowspan': 2, 'colspan': 0, 'span_total': 2}`
- counts GT: `{'len': 89, 'tr': 12, 'td_open': 49, 'td_close': 4, 'empty_td': 45, 'rowspan': 2, 'colspan': 2, 'span_total': 4}`
- first 5E-vs-7B diff: `replace` 5E[28:32] vs 7B[28:33]
  - 5E context: `<td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td  colspan="5" > </td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr>`
  - 7B context: `<td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr>`
- first 7B-vs-GT diff: `replace` 7B[28:33] vs GT[28:32]
  - 7B context: `<td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr>`
  - GT context: `<td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td  colspan="5" > </td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr>`

#### sample 474: PMC3752638_005_00.png

- delta 7B-5E: -0.654545
- TEDS 5E/7B: 0.981818 / 0.327273
- counts 5E: `{'len': 381, 'tr': 53, 'td_open': 262, 'td_close': 3, 'empty_td': 259, 'rowspan': 1, 'colspan': 2, 'span_total': 3}`
- counts 7B: `{'len': 386, 'tr': 54, 'td_open': 268, 'td_close': 2, 'empty_td': 266, 'rowspan': 0, 'colspan': 2, 'span_total': 2}`
- counts GT: `{'len': 388, 'tr': 54, 'td_open': 267, 'td_close': 3, 'empty_td': 264, 'rowspan': 1, 'colspan': 2, 'span_total': 3}`
- first 5E-vs-7B diff: `replace` 5E[2:6] vs 7B[2:3]
  - 5E context: `<thead> <tr> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
  - 7B context: `<thead> <tr> <td></td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
- first 7B-vs-GT diff: `replace` 7B[2:3] vs GT[2:6]
  - 7B context: `<thead> <tr> <td></td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
  - GT context: `<thead> <tr> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  colspan="2" > </td>`

### Best improvements

#### sample 611: PMC5880068_006_00.png

- delta 7B-5E: 0.907692
- TEDS 5E/7B: 0.092308 / 1.000000
- counts 5E: `{'len': 378, 'tr': 40, 'td_open': 255, 'td_close': 13, 'empty_td': 242, 'rowspan': 13, 'colspan': 0, 'span_total': 13}`
- counts 7B: `{'len': 392, 'tr': 40, 'td_open': 230, 'td_close': 26, 'empty_td': 204, 'rowspan': 26, 'colspan': 0, 'span_total': 26}`
- counts GT: `{'len': 392, 'tr': 40, 'td_open': 230, 'td_close': 26, 'empty_td': 204, 'rowspan': 26, 'colspan': 0, 'span_total': 26}`
- first 5E-vs-7B diff: `replace` 5E[31:32] vs 7B[31:35]
  - 5E context: ` rowspan="5" > </td> <td></td> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> <td></td>`
  - 7B context: ` rowspan="5" > </td> <td></td> <td></td> <td></td> <td></td> <td></td> <td  rowspan="5" > </td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr>`

#### sample 907: PMC6023382_029_00.png

- delta 7B-5E: 0.684823
- TEDS 5E/7B: 0.104651 / 0.789474
- counts 5E: `{'len': 501, 'tr': 72, 'td_open': 355, 'td_close': 0, 'empty_td': 355, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- counts 7B: `{'len': 67, 'tr': 9, 'td_open': 45, 'td_close': 0, 'empty_td': 45, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- counts GT: `{'len': 53, 'tr': 7, 'td_open': 35, 'td_close': 0, 'empty_td': 35, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- first 5E-vs-7B diff: `replace` 5E[66:501] vs 7B[66:67]
  - 5E context: `</tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td>< ...`
  - 7B context: `</tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> </tbody>`
- first 7B-vs-GT diff: `delete` 7B[52:66] vs GT[52:52]
  - 7B context: `</tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> </tbody>`
  - GT context: `</tr> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> </tbody>`

#### sample 50: PMC3824233_004_00.png

- delta 7B-5E: 0.621212
- TEDS 5E/7B: 0.378788 / 1.000000
- counts 5E: `{'len': 231, 'tr': 22, 'td_open': 153, 'td_close': 10, 'empty_td': 143, 'rowspan': 1, 'colspan': 9, 'span_total': 10}`
- counts 7B: `{'len': 229, 'tr': 22, 'td_open': 154, 'td_close': 9, 'empty_td': 145, 'rowspan': 0, 'colspan': 9, 'span_total': 9}`
- counts GT: `{'len': 229, 'tr': 22, 'td_open': 154, 'td_close': 9, 'empty_td': 145, 'rowspan': 0, 'colspan': 9, 'span_total': 9}`
- first 5E-vs-7B diff: `replace` 5E[2:6] vs 7B[2:3]
  - 5E context: `<thead> <tr> <td  rowspan="2" > </td> <td  colspan="2" > </td> <td  colspan="2" > </td>`
  - 7B context: `<thead> <tr> <td></td> <td  colspan="2" > </td> <td  colspan="2" > </td>`

#### sample 316: PMC2974660_007_00.png

- delta 7B-5E: 0.611111
- TEDS 5E/7B: 0.388889 / 1.000000
- counts 5E: `{'len': 82, 'tr': 7, 'td_open': 58, 'td_close': 2, 'empty_td': 56, 'rowspan': 0, 'colspan': 2, 'span_total': 2}`
- counts 7B: `{'len': 81, 'tr': 7, 'td_open': 57, 'td_close': 2, 'empty_td': 55, 'rowspan': 0, 'colspan': 2, 'span_total': 2}`
- counts GT: `{'len': 81, 'tr': 7, 'td_open': 57, 'td_close': 2, 'empty_td': 55, 'rowspan': 0, 'colspan': 2, 'span_total': 2}`
- first 5E-vs-7B diff: `replace` 5E[4:5] vs 7B[4:5]
  - 5E context: `<thead> <tr> <td></td> <td  colspan="3" > </td> <td></td> <td  colspan="3" > </td> </tr>`
  - 7B context: `<thead> <tr> <td></td> <td  colspan="4" > </td> <td  colspan="4" > </td> </tr> <tr>`

#### sample 951: PMC4143575_002_00.png

- delta 7B-5E: 0.602410
- TEDS 5E/7B: 0.397590 / 1.000000
- counts 5E: `{'len': 97, 'tr': 10, 'td_open': 64, 'td_close': 3, 'empty_td': 61, 'rowspan': 3, 'colspan': 0, 'span_total': 3}`
- counts 7B: `{'len': 94, 'tr': 10, 'td_open': 70, 'td_close': 0, 'empty_td': 70, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- counts GT: `{'len': 94, 'tr': 10, 'td_open': 70, 'td_close': 0, 'empty_td': 70, 'rowspan': 0, 'colspan': 0, 'span_total': 0}`
- first 5E-vs-7B diff: `replace` 5E[13:17] vs 7B[13:14]
  - 5E context: `<td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td  rowspan="3" > </td> <td></td> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr>`
  - 7B context: `<td></td> <td></td> <td></td> <td></td> </tr> </thead> <tbody> <tr> <td></td> <td></td> <td></td> <td></td> <td></td> <td></td> <td></td> </tr> <tr>`

## Interpretation

Phase7-B confirms that simple span-token reweighting is not an effective improvement path in the current setup.

- It increases exact-match accuracy from 0.693 to 0.697, but lowers Structure-TEDS from 0.830817 to 0.828003.
- The result repeats the Phase5-F pattern: exact-match accuracy can improve while the tree-edit objective gets worse.
- The remaining bottleneck is not just detecting span tokens, but deciding when spans are structurally safe. Future work should move toward span confidence calibration or constrained decoding rather than stronger loss reweighting.
