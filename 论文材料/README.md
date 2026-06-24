# 论文材料索引

本目录收集 DBM-SLANet / Phase4-Phase10 实验中可直接用于论文、组会汇报或给队长查看的说明文件。

## 推荐阅读顺序

1. `PHASE10_LARGE_VAL_5000_RESULTS_20260624.md`
   - 当前最重要结果。
   - 5000 样本验证集上，Phase5-E full Mamba 相比 baseline 的主表结果和统计检验。

2. `PHASE10_LARGE_VAL_3000_RESULTS_20260624.md`
   - 3000 样本验证结果。
   - 说明 Phase5-E 在更大验证集上开始表现出稳定提升。

3. `PHASE10_QUALITATIVE_CASE_ANALYSIS_20260624.md`
   - 定性案例分析。
   - 总结 Phase5-E 修复和破坏的代表性 span 案例。

4. `PAPER_READY_EXPERIMENT_TABLES_20260624.md`
   - 论文实验表格草稿。
   - 汇总 baseline、Phase4/5、Phase7/8/9/10 的主要实验。

5. `PAPER_EXPERIMENT_RESULTS_20260624.md`
   - 较完整的实验过程总结。
   - 包含阶段性结果、消融解释和实验链路。

6. `PHASE10_STATISTICAL_VALIDATION_20260624.md`
   - 1000 样本上的初步统计验证。
   - 主要作为历史记录和小样本 pilot 结果。

7. `PHASE4_PHASE5_EXPERIMENT_SUMMARY_20260623.md`
   - Phase4/Phase5 早期实验总结。
   - 用于回溯 full Mamba 主模型是怎么来的。

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
p(delta <= 0) ~= 0
```

## 注意事项

- Phase9-C 后处理在 1000 样本上有提升，但 3000 样本没有超过 Phase5-E，因此不建议作为最终主方法。
- 当前方法提升精度，但速度低于 baseline，应表述为 accuracy-oriented structural enhancement。
- 剩余主要失败模式仍是 `rowspan` / `colspan` 相关结构错误。
