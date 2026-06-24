# DBM-SLANet Related Work References

Date: 2026-06-24

Purpose: this file collects the references that are most directly useful for writing the DBM-SLANet manuscript. It is organized by the argument structure of the paper rather than by a flat bibliography.

## How to Use These References

- Use Section 1 to introduce the task, dataset, and evaluation protocol.
- Use Section 2 to position the SLANet/PaddleOCR baseline and the engineering starting point.
- Use Section 3 to compare against representative table structure recognition methods.
- Use Section 4 to justify why sequence modeling and Mamba/SSM are relevant to HTML-token structure prediction.
- Do not claim superiority over methods that were not evaluated in our experiments. For this project, the strongest confirmed claim is improvement over the fixed SLANet-style baseline under the same validation split and evaluation script.

## 1. Dataset, Metric, and Task Definition

### PubTabNet and TEDS

**Reference**

Zhong, X., ShafieiBavani, E., & Jimeno Yepes, A. Image-based table recognition: data, model, and evaluation. *European Conference on Computer Vision Workshops*, 2020.

**Why it matters**

- Introduced PubTabNet, the main benchmark dataset used in this project.
- Defined the image-to-HTML table recognition setting.
- Proposed TEDS, a tree-edit-distance-based similarity metric that evaluates table structure and content.

**Use in paper**

- Cite in Introduction and Experiments when defining PubTabNet, TEDS, and the table structure recognition objective.
- Our main reported metric, TEDS, should be described as following this evaluation tradition.

**Suggested wording**

> PubTabNet established large-scale image-based table recognition and introduced TEDS as a tree-edit-distance-based metric for comparing predicted and ground-truth table HTML structures.

### PubTables-1M and GriTS

**Reference**

Smock, B., Pesala, R., & Abraham, R. PubTables-1M: Towards comprehensive table extraction from unstructured documents. *CVPR*, 2022.

**Why it matters**

- Provides a large-scale table extraction dataset from scientific documents.
- Introduces GriTS-style evaluation for grid table structure similarity.
- Useful for discussing broader table extraction evaluation beyond PubTabNet.

**Use in paper**

- Cite in Related Work or Limitations to show that a stronger future version should also test on additional datasets and metrics.
- Since our current core experiments use PubTabNet-style evaluation, do not present PubTables-1M results unless they are actually run.

## 2. SLANet and PaddleOCR Baseline

### PP-StructureV2 / SLANet

**Reference**

PaddleOCR contributors. PP-StructureV2: A stronger document analysis system. arXiv preprint, 2022.

**Why it matters**

- Describes the PaddleOCR document analysis system and table recognition pipeline.
- SLANet is the practical baseline family that this project modifies.
- Relevant for explaining that DBM-SLANet keeps the visual backbone and focuses on the structure prediction head.

**Use in paper**

- Cite in Method when introducing the baseline architecture.
- Cite in Experiments when saying the baseline is kept fixed and compared against the Mamba-enhanced version.

**Suggested wording**

> We build on the SLANet-style table recognition pipeline in PaddleOCR and focus on improving the structure decoding component while keeping the broader training and evaluation setting aligned with the original baseline.

### PaddleOCR Toolkit

**Reference**

PaddleOCR contributors. PaddleOCR: an open-source OCR toolkit. GitHub repository and technical documentation.

**Why it matters**

- The project is implemented inside PaddleOCR.
- Useful for reproducibility statements and code availability.

**Use in paper**

- Cite as software/tooling, not as the main scientific novelty.

## 3. Representative Table Structure Recognition Methods

### Image-Based Encoder-Decoder Table Recognition

**Reference**

Zhong, X., ShafieiBavani, E., & Jimeno Yepes, A. Image-based table recognition: data, model, and evaluation. *ECCV Workshops*, 2020.

**Why it matters**

- PubTabNet paper also provides the canonical encoder-decoder baseline for image-to-HTML table recognition.
- This is the closest conceptual ancestor of HTML-token sequence prediction for tables.

**Use in paper**

- Cite when describing table recognition as sequence generation over HTML-like tokens.

### TableFormer

**Reference**

Nassar, A., Livathinos, N., Lysak, M., & Staar, P. TableFormer: Table structure understanding with transformers. arXiv preprint, 2022.

**Why it matters**

- Uses transformer-based modeling for table structure understanding.
- Important comparison point for sequence and attention-based table structure modeling.

**Use in paper**

- Cite in Related Work to show that prior work has used Transformer-style global modeling for table structure.
- Position DBM-SLANet as exploring a state-space alternative to attention-heavy sequence modeling.

### LGPMA

**Reference**

Qiao, L., Li, Z., Cheng, Z., et al. LGPMA: Complicated table structure recognition with local and global pyramid mask alignment. *International Conference on Document Analysis and Recognition*, 2021.

**Why it matters**

- Representative method for complicated table structure recognition.
- Strongly tied to cell alignment and local/global structural cues.

**Use in paper**

- Cite when discussing span-heavy or irregular tables and the need for robust structural reasoning.

### TableMaster

**Reference**

Ye, J., Qi, X., He, Y., et al. PingAn-VCGroup's solution for ICDAR 2021 competition on scientific literature parsing task B: table recognition to HTML. arXiv preprint, 2021.

**Why it matters**

- Often discussed as the TableMaster-style approach for table recognition to HTML.
- Useful for showing that robust table-to-HTML generation has been explored with stronger decoder designs and training recipes.

**Use in paper**

- Cite carefully as a competition-system / arXiv reference unless a formal paper version is available in the target bibliography.

### CascadeTabNet

**Reference**

Prasad, D., Gadpal, A., Kapadni, K., Visave, M., & Sultanpure, K. CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents. *CVPR Workshops*, 2020.

**Why it matters**

- End-to-end table detection and structure recognition from document images.
- Useful as a broader document table extraction reference, although it is less directly tied to our SLANet/PubTabNet setup.

**Use in paper**

- Cite in the broad Related Work paragraph, not as a direct experimental comparator.

## 4. Sequence Modeling, State Space Models, and Mamba

### Structured State Space Models / S4

**Reference**

Gu, A., Goel, K., & Re, C. Efficiently modeling long sequences with structured state spaces. *International Conference on Learning Representations*, 2022.

**Why it matters**

- Establishes structured state space sequence modeling as an efficient long-sequence alternative.
- Provides background for why SSMs are relevant when token dependencies are long-range.

**Use in paper**

- Cite before Mamba to introduce the SSM lineage.

### Mamba

**Reference**

Gu, A., & Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint, 2023.

**Why it matters**

- Core reference for selective state spaces and linear-time sequence modeling.
- Directly justifies the full-Mamba structure head used in Phase5-E.

**Use in paper**

- Cite in Method when introducing the Mamba-based structure head.
- Cite in Introduction when motivating a long-range dependency module for HTML token prediction.

**Suggested wording**

> Mamba provides selective state-space sequence modeling with linear-time complexity, making it attractive for structured token prediction where long-range dependencies are important but full attention may be costly.

### Vision Mamba

**Reference**

Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., & Wang, X. Vision Mamba: Efficient visual representation learning with bidirectional state space model. arXiv preprint, 2024.

**Why it matters**

- Shows the adaptation of Mamba-style state space modeling to visual representation learning.
- Useful for arguing that Mamba is not limited to language and can be applied in vision/document understanding contexts.

**Use in paper**

- Cite in Related Work for vision-side state-space modeling.
- Do not overstate it as a table recognition method.

### VMamba

**Reference**

Liu, Y., Tian, Y., Zhao, Y., Yu, H., Xie, L., Wang, Y., Ye, Q., Jiao, J., & Liu, Y. VMamba: Visual state space model. arXiv preprint, 2024.

**Why it matters**

- Another influential vision state-space model.
- Useful for positioning DBM-SLANet within the trend of adapting SSMs to visual and structured prediction tasks.

**Use in paper**

- Cite in the Mamba/SSM paragraph as broader evidence of the move from attention-only designs to state-space visual modeling.

## 5. Current Paper Positioning

The safest paper positioning is:

> We do not propose a completely new table recognition system from scratch. Instead, we study whether a SLANet-style table recognizer can benefit from a Mamba-enhanced structure prediction head. Under a fixed PubTabNet validation protocol, the final Phase5-E model improves TEDS from 0.8331 to 0.8440 on 5000 samples, with a paired bootstrap 95% confidence interval of +0.0059 to +0.0162.

Avoid these claims unless more experiments are added:

- "State-of-the-art on PubTabNet."
- "Outperforms all Transformer-based methods."
- "Generalizes to all document table datasets."
- "Solves complex table structure recognition."

Good defensible claims:

- "Improves a fixed SLANet-style baseline."
- "Shows statistically significant gains on a 5000-sample PubTabNet validation subset."
- "Improves exact structure accuracy from 0.6898 to 0.7182 under the same evaluation script."
- "Qualitative cases suggest the gain is concentrated in span-sensitive structural patterns, while span regressions remain the key failure mode."

## 6. Draft Related Work Structure

### Paragraph 1: Image-Based Table Recognition

Discuss table structure recognition as converting document table images into structured HTML or grid representations. Cite PubTabNet/TEDS and PubTables-1M/GriTS. Emphasize that accurate prediction of `rowspan` and `colspan` is essential for scientific tables.

### Paragraph 2: Neural Table Structure Models

Discuss encoder-decoder and Transformer-style methods such as the PubTabNet baseline, TableFormer, LGPMA, TableMaster, and CascadeTabNet. Explain that these methods improve structure recognition through stronger visual encoders, alignment strategies, or global sequence modeling.

### Paragraph 3: State Space Models for Long-Range Structure

Introduce S4 and Mamba as efficient long-sequence models, then mention Vision Mamba and VMamba as evidence that SSMs are being adapted to vision tasks. Position DBM-SLANet as applying this idea specifically to the HTML-token structure head of a table recognizer.

### Paragraph 4: Gap

State the gap conservatively:

> Despite progress in table structure recognition and state-space modeling, the effect of a Mamba-style structure decoder inside a practical SLANet-based table recognition pipeline remains underexplored. This work evaluates that design under controlled baseline conditions.

## 7. BibTeX Drafts

```bibtex
@inproceedings{zhong2020pubtabnet,
  title={Image-based table recognition: data, model, and evaluation},
  author={Zhong, Xu and ShafieiBavani, Elaheh and Jimeno Yepes, Antonio},
  booktitle={European Conference on Computer Vision Workshops},
  year={2020}
}

@inproceedings{smock2022pubtables,
  title={PubTables-1M: Towards comprehensive table extraction from unstructured documents},
  author={Smock, Brandon and Pesala, Rohith and Abraham, Robin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{paddleocr2022ppstructurev2,
  title={PP-StructureV2: A stronger document analysis system},
  author={{PaddleOCR Contributors}},
  journal={arXiv preprint},
  year={2022}
}

@article{nassar2022tableformer,
  title={TableFormer: Table structure understanding with transformers},
  author={Nassar, Ahmed and Livathinos, Nikolaos and Lysak, Maksym and Staar, Peter},
  journal={arXiv preprint},
  year={2022}
}

@inproceedings{qiao2021lgpma,
  title={LGPMA: Complicated table structure recognition with local and global pyramid mask alignment},
  author={Qiao, Liang and Li, Zhanzhan and Cheng, Zhanzhan and others},
  booktitle={International Conference on Document Analysis and Recognition},
  year={2021}
}

@inproceedings{prasad2020cascadetabnet,
  title={CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents},
  author={Prasad, Devashish and Gadpal, Ayan and Kapadni, Kshitij and Visave, Manish and Sultanpure, Kavita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}

@inproceedings{gu2022s4,
  title={Efficiently modeling long sequences with structured state spaces},
  author={Gu, Albert and Goel, Karan and R{\\'e}, Christopher},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@article{gu2023mamba,
  title={Mamba: Linear-time sequence modeling with selective state spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{zhu2024visionmamba,
  title={Vision Mamba: Efficient visual representation learning with bidirectional state space model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}

@article{liu2024vmamba,
  title={VMamba: Visual state space model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Jiao, Jianbin and Liu, Yunfan},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```

