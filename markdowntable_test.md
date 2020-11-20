
| **攻击策略** | **目标函数** | **约束条件** | **变换方式** | **搜索方法** | **主要思想** |
| :-- | :-- | :-- | :-- | :-- | :-- |
<td colspan="6"><strong>对于分类任务的攻击策略，例如情感分类和文本蕴含任务：</strong></td> 
| `alzantot` | 无目标 <br/> {分类，蕴含} | 被扰动词的比例，语言模型的困惑度，词嵌入的距离 | Counter-fitted 词嵌入替换 | 遗传算法 | 来自 (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998)) |
| `bae` | 无目标 <br/> 分类 | USE 通用句子编码向量的 cosine 相似度 | BERT 遮罩词预测 | 对 WIR 的贪心搜索 | 使用 BERT 语言模型作为变换的攻击方法，来自 (["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019)](https://arxiv.org/abs/2004.01970)) |
