# Baseline 相关参考文献整理

整理时间：2026-06-25  
项目：DBM-SLANet / SLANet baseline / PubTabNet 表格结构识别实验

## 1. 简要结论

当前实验的 baseline 可以表述为：

> 本实验以 PaddleOCR / PP-StructureV2 中的 SLANet-style 表格结构识别模型作为 baseline，在相同 PubTabNet 验证集和相同 TEDS / structure accuracy 评价指标下，对引入 full-Mamba table structure head 后的 Phase5-E 模型进行对比。

最小必引文献建议 4 篇：

1. **PP-StructureV2 / SLANet**：说明 baseline 模型来源。
2. **PubTabNet / TEDS**：说明数据集和评价指标来源。
3. **PP-LCNet**：说明 baseline backbone 来源。
4. **PaddleOCR**：说明代码框架和复现工具来源。

如果写完整 related work，可再补充 TableMaster、TableFormer、LGPMA、CascadeTabNet、TableBank、DeepDeSRT、TableNet 等代表性方法。

## 2. 当前实验必引文献

### 2.1 PP-StructureV2 / SLANet

Li, C., Guo, R., Zhou, J., An, M., Du, Y., Zhu, L., Liu, Y., Hu, X., & Yu, D.  
**PP-StructureV2: A Stronger Document Analysis System**.  
arXiv preprint arXiv:2210.05391, 2022.  
链接：<https://arxiv.org/abs/2210.05391>

使用位置：

- baseline 模型来源；
- SLANet-style table recognition pipeline 说明；
- PP-LCNet + CSP-PAN + SLAHead 架构说明。

BibTeX：

```bibtex
@article{li2022ppstructurev2,
  title={PP-StructureV2: A Stronger Document Analysis System},
  author={Li, Chenxia and Guo, Ruoyu and Zhou, Jun and An, Mengtao and Du, Yuning and Zhu, Lingfeng and Liu, Yi and Hu, Xiaoguang and Yu, Dianhai},
  journal={arXiv preprint arXiv:2210.05391},
  year={2022}
}
```

### 2.2 PubTabNet / TEDS

Zhong, X., ShafieiBavani, E., & Jimeno Yepes, A.  
**Image-based table recognition: data, model, and evaluation**.  
ECCV Workshops, 2020.  
链接：<https://arxiv.org/abs/1911.10683>

使用位置：

- PubTabNet 数据集来源；
- TEDS 指标来源；
- 当前 3000 / 5000 validation split 的实验设置说明。

BibTeX：

```bibtex
@inproceedings{zhong2020pubtabnet,
  title={Image-based table recognition: data, model, and evaluation},
  author={Zhong, Xu and ShafieiBavani, Elaheh and Jimeno Yepes, Antonio},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
```

### 2.3 PP-LCNet

Cui, C., Gao, T., Wei, S., Du, Y., Guo, R., Dong, S., Lu, B., Zhou, Y., Lv, X., Liu, Q., Hu, X., Yu, D., & Ma, Y.  
**PP-LCNet: A Lightweight CPU Convolutional Neural Network**.  
arXiv preprint arXiv:2109.15099, 2021.  
链接：<https://arxiv.org/abs/2109.15099>

使用位置：

- baseline backbone 来源；
- 说明 SLANet / PP-StructureV2 使用轻量 CNN backbone。

BibTeX：

```bibtex
@article{cui2021pplcnet,
  title={PP-LCNet: A Lightweight CPU Convolutional Neural Network},
  author={Cui, Cheng and Gao, Tingquan and Wei, Shengyu and Du, Yuning and Guo, Ruoyu and Dong, Shuilong and Lu, Bin and Zhou, Ying and Lv, Xueying and Liu, Qiwen and Hu, Xiaoguang and Yu, Dianhai and Ma, Yanjun},
  journal={arXiv preprint arXiv:2109.15099},
  year={2021}
}
```

### 2.4 PaddleOCR

PaddleOCR Contributors.  
**PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle**.  
GitHub repository.  
链接：<https://github.com/PaddlePaddle/PaddleOCR>

使用位置：

- 代码框架；
- 训练、推理、后处理和复现实验设置；
- 不建议作为主要算法创新引用。

BibTeX：

```bibtex
@misc{paddleocr,
  title={PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle},
  author={{PaddleOCR Contributors}},
  howpublished={\url{https://github.com/PaddlePaddle/PaddleOCR}},
  year={2020}
}
```

## 3. OCR 与后处理相关文献

### 3.1 PP-OCR

Du, Y., Li, C., Guo, R., Yin, X., Liu, W., Zhou, J., Bai, Y., Yu, Z., Yang, Y., Dang, Q., & Wang, H.  
**PP-OCR: A Practical Ultra Lightweight OCR System**.  
arXiv preprint arXiv:2009.09941, 2020.  
链接：<https://arxiv.org/abs/2009.09941>

用途：

- OCR 检测、识别流程背景；
- OCR-cell matching、空单元格和文本识别误差分析。

### 3.2 PP-OCRv3

Li, C., Liu, W., Guo, R., Yin, X., Jiang, K., Du, Y., Du, Y., Zhu, L., Lai, B., Hu, X., Yu, D., & Ma, Y.  
**PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System**.  
arXiv preprint arXiv:2206.03001, 2022.  
链接：<https://arxiv.org/abs/2206.03001>

用途：

- 如果实验中使用 PP-OCRv3 det / rec 模型，应在 implementation details 中引用。

## 4. Related Work 可补充方法

| 方法 | 文献 | 建议用途 |
|---|---|---|
| TableMaster | Ye et al., 2021 | table-to-HTML 方向代表方法 |
| TableFormer | Nassar et al., 2022 | Transformer-based 表格结构识别代表方法 |
| LGPMA | Qiao et al., 2021 | 复杂表格、rowspan / colspan、local-global alignment |
| CascadeTabNet | Prasad et al., 2020 | end-to-end 表格检测与结构识别背景 |
| TableBank | Li et al., 2020 | 表格检测/识别 benchmark 背景 |
| DeepDeSRT | Schreiber et al., 2017 | 早期 deep learning 表格检测与结构识别 |
| TableNet | Paliwal et al., 2019 | CNN / segmentation-style 表格检测与抽取方法 |

这些方法适合写 related work，不建议直接写成本文实验对比结果，除非后续补充同一数据集、同一指标下的复现实验。

## 5. 建议论文表述

### Baseline 描述

可以写：

> We adopt the SLANet-style table recognition model from PaddleOCR / PP-StructureV2 as the baseline. The baseline uses a lightweight PP-LCNet backbone, CSP-PAN neck, and SLAHead for HTML-like table structure decoding. Experiments are conducted on PubTabNet validation splits and evaluated using TEDS and structure accuracy.

中文解释：

> 本文采用 PaddleOCR / PP-StructureV2 中的 SLANet-style 表格结构识别模型作为 baseline。该 baseline 使用 PP-LCNet 作为轻量视觉骨干网络，结合 CSP-PAN 和 SLAHead 进行 HTML-like 表格结构解码。实验在 PubTabNet 验证集上进行，并使用 TEDS 和 structure accuracy 作为评价指标。

### 本文方法定位

可以写：

> Based on this fixed SLANet-style baseline, we replace the original table structure decoding head with a full-Mamba structure head to strengthen sequential structural dependency modeling, especially for span-sensitive table patterns involving rowspan and colspan.

中文解释：

> 在固定 SLANet-style baseline 的基础上，本文将原始表格结构解码头替换为 full-Mamba structure head，以增强表格结构 token 序列中的长程依赖建模能力，尤其针对 rowspan 和 colspan 等 span-sensitive 结构。

### 不建议写法

不建议写：

> 本文方法超过所有表格识别 SOTA。

原因：

- 当前实验主要是在 SLANet-style baseline 上做改进；
- 尚未复现所有公开方法；
- 当前最强证据是 Phase5-E 相比原始 SLANet baseline 在 PubTabNet 3000/5000 validation split 上稳定提升。

更稳妥的写法：

> 本文方法在固定 SLANet-style baseline 和相同 PubTabNet 验证设置下取得稳定提升。

## 6. 精简版

如果只需要 baseline 相关文献，可以使用下面这一段：

> Baseline 采用 PaddleOCR / PP-StructureV2 中的 SLANet-style table recognition pipeline，主要引用 PP-StructureV2 说明 baseline 模型来源，引用 PubTabNet 说明数据集和 TEDS 指标来源，引用 PP-LCNet 说明 baseline backbone 来源，引用 PaddleOCR 说明代码框架和复现工具来源。完整 related work 可补充 TableMaster、TableFormer、LGPMA、CascadeTabNet、TableBank、DeepDeSRT 和 TableNet，但这些目前主要作为背景方法，不作为直接数值对比。

最小参考文献：

1. Li et al., **PP-StructureV2: A Stronger Document Analysis System**, arXiv 2022.
2. Zhong et al., **Image-based table recognition: data, model, and evaluation**, ECCV Workshops 2020.
3. Cui et al., **PP-LCNet: A Lightweight CPU Convolutional Neural Network**, arXiv 2021.
4. PaddleOCR Contributors, **PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle**, GitHub repository.
