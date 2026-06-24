# DBM-SLANet 论文材料索引

Date: 2026-06-24

本目录收集 DBM-SLANet / Phase4-Phase10 实验中可直接用于论文、组会汇报或给队长查看的材料。

## 最推荐阅读顺序

1. `FINAL_PAPER_EXPERIMENT_TABLES_20260624.md`
   - 最终论文实验表。
   - 以 5000 样本结果为主，明确 Phase5-E 是最终主模型。

2. `PAPER_WRITING_BLUEPRINT_20260624.md`
   - 论文写作蓝图。
   - 包含题目、摘要骨架、章节结构、实验章节写法。

3. `SUBMISSION_READINESS_CHECKLIST_20260624.md`
   - 投稿准备清单。
   - 标出已经完成和还需要补的材料。

4. `PHASE10_LARGE_VAL_5000_RESULTS_20260624.md`
   - 当前最硬主结果。
   - 5000 样本验证集上，Phase5-E full Mamba 相比 baseline 的主表结果和统计检验。

5. `PHASE10_QUALITATIVE_CASE_ANALYSIS_20260624.md`
   - 定性案例分析。
   - 总结 Phase5-E 修复和破坏的代表性 span 案例。

6. `qualitative_figures/`
   - 已导出的代表性案例图。
   - 当前包含 3 个成功案例和 2 个失败案例。

7. `PAPER_EXPERIMENT_RESULTS_20260624.md`
   - 较完整的实验过程总结。
   - 用于回溯 Phase4-Phase10 的实验链路。

## 当前论文主结论

当前建议以 `Phase5-E full Mamba` 作为论文主方法：

```text
Baseline TEDS: 0.8330900340
Phase5-E TEDS: 0.8439856134
TEDS gain: +0.0108955794

Baseline structure_acc: 0.6898
Phase5-E structure_acc: 0.7182
structure_acc gain: +0.0284
```

5000 样本 paired bootstrap:

```text
95% CI: [+0.0059055140, +0.0161684765]
p(delta <= 0) = 0.0000
```

## 论文定位

推荐定位：

```text
一个面向表格结构识别的 Mamba-enhanced structural decoder，在 PubTabNet 5000 样本验证集上稳定提升 TEDS 和 structure accuracy。
```

需要谨慎表述：

- Phase9-C 后处理不作为最终方法，因为 3000 样本上没有超过 Phase5-E。
- 当前方法提升精度，但速度低于 baseline，应表述为 accuracy-oriented structural enhancement。
- 剩余主要失败模式仍是 `rowspan` / `colspan` 相关结构错误。

## 下一步

1. 画方法结构图。
2. 把 `qualitative_figures/` 做成论文多 panel 图。
3. 根据 `PAPER_WRITING_BLUEPRINT_20260624.md` 写英文论文初稿。
4. 如时间允许，补完整 9115 val 或第二数据集评估。
