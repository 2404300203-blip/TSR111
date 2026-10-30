# 服务器代码归档（2026-10-30）

本次以 TSR111 main 的 `7270830292396ddf15a2dbde4875596beda5ed26` 为基础整理。
服务器 `/root/autodl-tmp/PaddleOCR` 的已跟踪代码与该提交一致；原始工作目录未移动、覆盖或删除。

## 目录索引

| 目录 | 内容 |
|---|---|
| `../../ppocr/`、`../../ppstructure/` | 当前模型、损失、表格匹配和推理实现 |
| `../../configs/table/` | 主实验配置 |
| `../../scripts/` | 数据准备、评估、作图及外部模型评估入口 |
| `runs/` | 原 `PaddleOCR/output/` 内的 Python、Shell 和 YAML 配置，保留相对目录 |
| `history/model_heads/` | 原 head 文件旁的历史备份，保留原文件名，不作为当前实现 |
| `export_20260624/` | 2026-06-24 导出包内的源码和配置快照 |
| `manifest.csv` | 每个归档文件的原路径、归档路径、字节数和 SHA-256 |

外部模型目录中的 `eval_unitable_cta500.py`、`eval_table_transformer_cta500.py` 和
`profile_fig1_gpu_memory.py` 与仓库 `scripts/` 中版本逐字节一致，直接使用既有脚本。
第三方 UniTable、Table Transformer 源码不重复复制；安装时按相应上游说明准备，保留其许可证。

## 如何使用

从仓库根目录运行当前工具。历史实验脚本仍可能引用 `/root/autodl-tmp/PaddleOCR`、
`output/`、特定 Conda 环境或权重路径；归档保留原文，执行前先检查路径和参数。
`runs/` 是源码档案，并不包含对应训练输出；不要直接把历史流水线当作全自动启动入口。
训练、评估所需的 PubTabNet / FinTabNet / CTA 数据及模型权重需另行准备。

记录本次 SSH 会话看到的环境：Ubuntu 22.04、RTX 4090 D、base Python 3.10.8、
base paddlepaddle-gpu 2.6.2；服务器另有 `slanet` Conda 环境和外部评估虚拟环境。
这不是完整、经过重新安装验证的依赖锁定文件。当前整理只验证源码和配置，不重跑训练或声称复现实验指标。

生成新的独立归档：

```bash
python scripts/archive_server_code.py --server-root /root/autodl-tmp --destination experiments/server_YYYYMMDD
```

目标目录必须不存在。脚本只选择源码、配置与 head 历史备份，并生成校验清单。
不选择数据集、预测内容、模型权重、训练日志、进程状态、缓存、虚拟环境和账号配置。
主代码同步应先检查 Git 差异，再单独提交。

## 本次归档检查

- 共 420 个来源文件：`runs/` 386 个、导出快照 28 个、head 历史备份 6 个；约 1.8 MB。
- 420 项大小和 SHA-256 校验一致。
- 33 个 Python 文件（含归档工具与历史备份）通过 AST 语法解析。
- 361 个 YAML 文件通过 safe_load 解析；27 个 Shell 文件通过 `bash -n`。
- 对新增源码进行了常见私钥、GitHub token、AWS access key 及硬编码密码模式扫描，未发现匹配项。
- 未运行训练、完整推理或指标复现；归档保留历史源码，不代表所有旧实验都能在新路径直接运行。
