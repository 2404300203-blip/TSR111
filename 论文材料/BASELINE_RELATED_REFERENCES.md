# Baseline 相关参考文献整理

整理时间：2026-06-25  
适用对象：DBM-SLANet / SLANet baseline / PubTabNet 表格结构识别实验

## 1. 当前实验 baseline 必引文献

### 1.1 PP-StructureV2 / SLANet

Li, C., Guo, R., Zhou, J., An, M., Du, Y., Zhu, L., Liu, Y., Hu, X., & Yu, D.  
**PP-StructureV2: A Stronger Document Analysis System**.  
arXiv preprint arXiv:2210.05391, 2022.  
链接：<https://arxiv.org/abs/2210.05391>

用途：

- 当前实验里的 baseline 本质上是 PaddleOCR/PP-StructureV2 中的 SLANet-style table recognition pipeline。
- 这篇文章明确提到 table recognition 模型使用 PP-LCNet、CSP-PAN、SLAHead，并提升 table structure accuracy。
- 论文中介绍 baseline 架构、说明“在固定 SLANet-style baseline 上做改进”时应引用。

BibTeX：

```bibtex
@article{li2022ppstructurev2,
  title={PP-StructureV2: A Stronger Document Analysis System},
  author={Li, Chenxia and Guo, Ruoyu and Zhou, Jun and An, Mengtao and Du, Yuning and Zhu, Lingfeng and Liu, Yi and Hu, Xiaoguang and Yu, Dianhai},
  journal={arXiv preprint arXiv:2210.05391},
  year={2022}
}
```

### 1.2 PaddleOCR 工具库

PaddleOCR Contributors.  
**PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle**.  
GitHub repository.  
链接：<https://github.com/PaddlePaddle/PaddleOCR>

用途：

- 本项目代码基于 PaddleOCR，训练、推理、后处理、评估脚本都继承自该工程。
- 适合放在实现细节、复现设置、代码可用性部分。
- 不建议把它作为主要算法创新引用；主要算法引用应使用 PP-StructureV2 / SLANet。

BibTeX：

```bibtex
@misc{paddleocr,
  title={PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle},
  author={{PaddleOCR Contributors}},
  howpublished={\url{https://github.com/PaddlePaddle/PaddleOCR}},
  year={2020}
}
```

### 1.3 PP-LCNet

Cui, C., Gao, T., Wei, S., Du, Y., Guo, R., Dong, S., Lu, B., Zhou, Y., Lv, X., Liu, Q., Hu, X., Yu, D., & Ma, Y.  
**PP-LCNet: A Lightweight CPU Convolutional Neural Network**.  
arXiv preprint arXiv:2109.15099, 2021.  
链接：<https://arxiv.org/abs/2109.15099>

用途：

- SLANet/PP-StructureV2 表格识别 baseline 使用 PP-LCNet 作为轻量视觉 backbone。
- 如果论文方法部分写 baseline backbone 或轻量化模型来源，需要引用。

BibTeX：

```bibtex
@article{cui2021pplcnet,
  title={PP-LCNet: A Lightweight CPU Convolutional Neural Network},
  author={Cui, Cheng and Gao, Tingquan and Wei, Shengyu and Du, Yuning and Guo, Ruoyu and Dong, Shuilong and Lu, Bin and Zhou, Ying and Lv, Xueying and Liu, Qiwen and Hu, Xiaoguang and Yu, Dianhai and Ma, Yanjun},
  journal={arXiv preprint arXiv:2109.15099},
  year={2021}
}
```

## 2. 数据集与评价指标必引文献

### 2.1 PubTabNet / TEDS

Zhong, X., ShafieiBavani, E., & Jimeno Yepes, A.  
**Image-based table recognition: data, model, and evaluation**.  
ECCV Workshops, 2020.  
链接：<https://arxiv.org/abs/1911.10683>

用途：

- PubTabNet 数据集来源。
- TEDS 指标来源。
- 当前实验使用 PubTabNet validation split 和 TEDS/structure accuracy，必须引用。

BibTeX：

```bibtex
@inproceedings{zhong2020pubtabnet,
  title={Image-based table recognition: data, model, and evaluation},
  author={Zhong, Xu and ShafieiBavani, Elaheh and Jimeno Yepes, Antonio},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}
```

### 2.2 PubTables-1M / GriTS

Smock, B., Pesala, R., & Abraham, R.  
**PubTables-1M: Towards comprehensive table extraction from unstructured documents**.  
CVPR, 2022.  
链接：<https://arxiv.org/abs/2110.00061>

用途：

- 可用于 related work 或 limitations，说明表格结构识别还有 PubTables-1M / GriTS 这类更广泛的评估体系。
- 当前实验没有跑 PubTables-1M 时，不要写成直接对比结果。

BibTeX：

```bibtex
@inproceedings{smock2022pubtables,
  title={PubTables-1M: Towards comprehensive table extraction from unstructured documents},
  author={Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## 3. OCR / 后处理相关 baseline 文献

### 3.1 PP-OCR

Du, Y., Li, C., Guo, R., Yin, X., Liu, W., Zhou, J., Bai, Y., Yu, Z., Yang, Y., Dang, Q., & Wang, H.  
**PP-OCR: A Practical Ultra Lightweight OCR System**.  
arXiv preprint arXiv:2009.09941, 2020.  
链接：<https://arxiv.org/abs/2009.09941>

用途：

- 端到端 table-to-HTML 流程涉及 OCR 检测、识别和 OCR-cell matching。
- 写 Experiment A/F 的 OCR-aware matching、空单元格问题、文本识别误差来源时可以引用。

BibTeX：

```bibtex
@article{du2020ppocr,
  title={PP-OCR: A Practical Ultra Lightweight OCR System},
  author={Du, Yuning and Li, Chenxia and Guo, Ruoyu and Yin, Xiaoting and Liu, Weiwei and Zhou, Jun and Bai, Yifan and Yu, Zilin and Yang, Yehua and Dang, Qingqing and Wang, Hailei},
  journal={arXiv preprint arXiv:2009.09941},
  year={2020}
}
```

### 3.2 PP-OCRv3

Li, C., Liu, W., Guo, R., Yin, X., Jiang, K., Du, Y., Du, Y., Zhu, L., Lai, B., Hu, X., Yu, D., & Ma, Y.  
**PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System**.  
arXiv preprint arXiv:2206.03001, 2022.  
链接：<https://arxiv.org/abs/2206.03001>

用途：

- 如果实验 F 使用 PP-OCRv3 det/rec 模型做 OCR-aware matching，应引用。
- 适合放在 OCR system 或 implementation details。

BibTeX：

```bibtex
@article{li2022ppocrv3,
  title={PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System},
  author={Li, Chenxia and Liu, Weiwei and Guo, Ruoyu and Yin, Xiaoting and Jiang, Kaitao and Du, Yongkun and Du, Yuning and Zhu, Lingfeng and Lai, Baohua and Hu, Xiaoguang and Yu, Dianhai and Ma, Yanjun},
  journal={arXiv preprint arXiv:2206.03001},
  year={2022}
}
```

## 4. baseline 对比表可放的代表方法

### 4.1 TableMaster

Ye, J., Qi, X., He, Y., Chen, Y., Gu, D., Gao, P., & Xiao, R.  
**PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML**.  
arXiv preprint arXiv:2105.01848, 2021.  
链接：<https://arxiv.org/abs/2105.01848>

用途：

- table-to-HTML 方向的重要工程型方法。
- 和当前任务形式最接近，可作为 related work 或 baseline table 的背景方法。

BibTeX：

```bibtex
@article{ye2021tablemaster,
  title={PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML},
  author={Ye, Jiaquan and Qi, Xianbiao and He, Yelin and Chen, Yihao and Gu, Dengyi and Gao, Peng and Xiao, Rong},
  journal={arXiv preprint arXiv:2105.01848},
  year={2021}
}
```

### 4.2 TableFormer

Nassar, A., Livathinos, N., Lysak, M., & Staar, P.  
**TableFormer: Table Structure Understanding with Transformers**.  
arXiv preprint arXiv:2203.01017, 2022.  
链接：<https://arxiv.org/abs/2203.01017>

用途：

- Transformer-based table structure recognition 代表方法。
- 用于说明 Transformer/global modeling 是表格结构识别中的常见方向；你的 DBM-SLANet 可定位为 Mamba/SSM 替代路线。

BibTeX：

```bibtex
@article{nassar2022tableformer,
  title={TableFormer: Table Structure Understanding with Transformers},
  author={Nassar, Ahmed and Livathinos, Nikolaos and Lysak, Maksym and Staar, Peter},
  journal={arXiv preprint arXiv:2203.01017},
  year={2022}
}
```

### 4.3 LGPMA

Qiao, L., Li, Z., Cheng, Z., Zhang, P., Pu, S., Niu, Y., Ren, W., Tan, W., & Wu, F.  
**LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment**.  
ICDAR, 2021.  
链接：<https://arxiv.org/abs/2105.06224>

用途：

- 复杂表格结构识别代表方法，强调 local/global alignment 和复杂 cell spanning。
- 适合在 related work 中讨论复杂表格、rowspan/colspan、结构对齐问题。

BibTeX：

```bibtex
@inproceedings{qiao2021lgpma,
  title={LGPMA: Complicated Table Structure Recognition with Local and Global Pyramid Mask Alignment},
  author={Qiao, Liang and Li, Zaisheng and Cheng, Zhanzhan and Zhang, Peng and Pu, Shiliang and Niu, Yi and Ren, Wenqi and Tan, Wenming and Wu, Fei},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2021}
}
```

### 4.4 CascadeTabNet

Prasad, D., Gadpal, A., Kapadni, K., Visave, M., & Sultanpure, K.  
**CascadeTabNet: An Approach for End to End Table Detection and Structure Recognition from Image-Based Documents**.  
CVPR Workshops, 2020.  
链接：<https://arxiv.org/abs/2004.12629>

用途：

- end-to-end table detection + structure recognition 代表方法。
- 更适合放 broad related work，不建议作为你当前 PubTabNet/SLANet 实验的直接可比 baseline。

BibTeX：

```bibtex
@inproceedings{prasad2020cascadetabnet,
  title={CascadeTabNet: An Approach for End to End Table Detection and Structure Recognition from Image-Based Documents},
  author={Prasad, Devashish and Gadpal, Ayan and Kapadni, Kshitij and Visave, Manish and Sultanpure, Kavita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={2439--2447},
  year={2020}
}
```

### 4.5 TableBank

Li, M., Cui, L., Huang, S., Wei, F., Zhou, M., & Li, Z.  
**TableBank: Table Benchmark for Image-based Table Detection and Recognition**.  
LREC, 2020.  
链接：<https://aclanthology.org/2020.lrec-1.236/>

用途：

- 表格检测/识别数据集与 benchmark 背景。
- 可用于介绍 image-based table recognition 的发展，但当前实验主数据集是 PubTabNet。

BibTeX：

```bibtex
@inproceedings{li2020tablebank,
  title={TableBank: Table Benchmark for Image-based Table Detection and Recognition},
  author={Li, Minghao and Cui, Lei and Huang, Shaohan and Wei, Furu and Zhou, Ming and Li, Zhoujun},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={1918--1925},
  year={2020}
}
```

### 4.6 DeepDeSRT

Schreiber, S., Agne, S., Wolf, I., Dengel, A., & Ahmed, S.  
**DeepDeSRT: Deep Learning for Detection and Structure Recognition of Tables in Document Images**.  
ICDAR, 2017.  
链接：<https://ieeexplore.ieee.org/document/8270123/>

用途：

- 早期 deep-learning table detection + structure recognition 方法。
- 可放 related work 的历史脉络，不适合作为当前实验的直接数字对比。

BibTeX：

```bibtex
@inproceedings{schreiber2017deepdesrt,
  title={DeepDeSRT: Deep Learning for Detection and Structure Recognition of Tables in Document Images},
  author={Schreiber, Sebastian and Agne, Stefan and Wolf, Ivo and Dengel, Andreas and Ahmed, Sheraz},
  booktitle={2017 14th IAPR International Conference on Document Analysis and Recognition},
  pages={1162--1167},
  year={2017},
  organization={IEEE}
}
```

### 4.7 TableNet

Paliwal, S. S., Vishwanath, D., Rahul, R., Sharma, M., & Vig, L.  
**TableNet: Deep Learning Model for End-to-End Table Detection and Tabular Data Extraction from Scanned Document Images**.  
ICDAR, 2019.  
链接：<https://ieeexplore.ieee.org/document/8978013/>

用途：

- 表格检测与结构提取的端到端方法之一。
- 可作为 broad related work 或传统 CNN/segmentation-style baseline 背景。

BibTeX：

```bibtex
@inproceedings{paliwal2019tablenet,
  title={TableNet: Deep Learning Model for End-to-End Table Detection and Tabular Data Extraction from Scanned Document Images},
  author={Paliwal, Shubham Singh and Vishwanath, D and Rahul, R and Sharma, Monika and Vig, Lovekesh},
  booktitle={2019 International Conference on Document Analysis and Recognition},
  year={2019},
  organization={IEEE}
}
```

## 5. 建议论文里怎么用

如果队长说“baseline 给我一下”，建议 baseline 相关文献最小集合先给这 4 篇：

1. PP-StructureV2 / SLANet：说明 baseline 模型来源。
2. PubTabNet / TEDS：说明数据集和评价指标来源。
3. PP-LCNet：说明 baseline backbone 来源。
4. PaddleOCR：说明代码框架和复现工具来源。

如果要写完整 related work，再补：

1. TableMaster
2. TableFormer
3. LGPMA
4. CascadeTabNet
5. TableBank
6. DeepDeSRT
7. TableNet

注意：当前论文最稳妥的表述是“在固定 SLANet-style baseline 上取得提升”，不要写成“超过所有表格识别 SOTA”，除非后续补全公开 benchmark 的直接对比实验。
