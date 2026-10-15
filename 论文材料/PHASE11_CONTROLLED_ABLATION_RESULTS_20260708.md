# Phase11 控制消融实验结果（Cell Token / λ_pres）

日期：2026-07-08

## 实验目的

本轮补充两个控制消融，用来回答两个审稿人很可能追问的问题：

1. 性能提升来自几何 cell token 本身，还是来自 Mamba 关系建模？
2. `cell_presence` auxiliary loss 是否是 DBM 分支生效的必要训练信号？

两个模型都从同一个 baseline 权重 `output/baseline_package_20260621/model/best_accuracy.pdparams` 初始化，并使用相同 30k 训练子集训练一轮。

## 训练集内验证结果（val_1000）

| Model | Best Acc | Latest Acc | Best FPS | 说明 |
|---|---:|---:|---:|---|
| Phase11-A Cell Token only, λ_pres=0.2 | 0.6820 | 0.6750 | 26.3067 | 只有 8D 几何 token + cell presence head，不含 Mamba |
| Phase11-B Full Mamba, λ_pres=0 | 0.6780 | 0.6720 | 11.4055 | 保留完整 Mamba 结构，但关闭 cell-presence loss |

## CTA-500 复杂结构专项结果

| Model | CTA-500 TEDS ↑ | Structure Acc ↑ | FPS ↑ | ΔTEDS vs Baseline |
|---|---:|---:|---:|---:|
| Original SLANet baseline | 0.484335 | 0.2080 | 16.3023 | +0.000000 |
| Phase11-A Cell Token only, lambda=0.2 | 0.543644 | 0.2840 | 20.3262 | +0.059308 |
| Phase11-B Full Mamba, lambda=0 | 0.539207 | 0.2820 | 9.3970 | +0.054872 |
| Exp C Horizontal-only Mamba | 0.627637 | 0.4200 | 8.6467 | +0.143302 |
| Exp D Vertical-only Mamba | 0.625761 | 0.4160 | 9.7617 | +0.141426 |
| Exp E Full Dual-Branch Mamba | 0.640673 | 0.4320 | 8.1429 | +0.156337 |

## 关键差值

- Cell Token only 相比 baseline：`+0.059308` TEDS。
- Full Mamba 但 λ_pres=0 相比 baseline：`+0.054872` TEDS。
- Full Mamba 但 λ_pres=0 相比 Cell Token only：`-0.004436` TEDS。
- 正常 Full Mamba λ_pres=0.2 相比 λ_pres=0：`+0.101465` TEDS。
- 正常 Full Mamba λ_pres=0.2 相比 Cell Token only：`+0.097029` TEDS。

## 结果解释

- `Cell Token only` 从 `0.484335` 提升到 `0.543644`，说明 8D 几何 cell token 不是无效堆叠，单独就能改善复杂表格。
- `Full Mamba, λ_pres=0` 为 `0.539207`，略低于 `Cell Token only`，说明如果关闭 cell-presence auxiliary loss，Mamba 分支虽然存在，但难以形成有效的复杂结构收益。
- 正常 `Full Dual-Branch Mamba, λ_pres=0.2` 达到 `0.640673`，相比 λ_pres=0 高 `+0.101465` TEDS。这是证明 auxiliary supervision 必要性的强证据。
- 因此论文表述应强调：DBM 分支不是直接改写 HTML logits，而是通过几何感知 cell 表征和 cell-presence auxiliary supervision 学习结构正则。

## 按复杂结构标签分解

| Tag | N | Baseline | Cell Token | λ=0 | λ=0.2 Full | λ=0.2 - λ=0 |
|---|---:|---:|---:|---:|---:|---:|
| long_range_dependency | 473 | 0.4891 | 0.5490 | 0.5425 | 0.6304 | +0.0879 |
| row_col_span | 457 | 0.4627 | 0.5250 | 0.5219 | 0.6479 | +0.1260 |
| horizontal_dependency | 376 | 0.4581 | 0.5177 | 0.5156 | 0.6461 | +0.1305 |
| tall_table | 341 | 0.4913 | 0.5396 | 0.5327 | 0.5709 | +0.0382 |
| multi_level_header | 325 | 0.4604 | 0.5274 | 0.5278 | 0.6523 | +0.1245 |
| vertical_dependency | 286 | 0.4496 | 0.5154 | 0.5111 | 0.6389 | +0.1278 |
| hierarchical_span_header | 274 | 0.4753 | 0.5448 | 0.5429 | 0.6759 | +0.1330 |
| wide_table | 149 | 0.5417 | 0.5993 | 0.5933 | 0.7083 | +0.1150 |

## 文件位置

- 训练日志：`/Users/table/TSR/remote_results/phase11_controlled_ablation_20260708/E11A_cell_token_only_train.log`
- 训练日志：`/Users/table/TSR/remote_results/phase11_controlled_ablation_20260708/E11B_full_mamba_lambda0_train.log`
- CTA-500 样本级 delta：`/Users/table/TSR/remote_results/phase11_controlled_ablation_20260708/phase11_cta500_sample_delta.jsonl`
- 按标签汇总：`/Users/table/TSR/remote_results/phase11_controlled_ablation_20260708/phase11_cta500_by_tag.csv`
- JSON 总结：`/Users/table/TSR/remote_results/phase11_controlled_ablation_20260708/phase11_controlled_ablation_summary.json`
