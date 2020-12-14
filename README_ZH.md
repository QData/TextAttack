<h1 align="center">TextAttack 🐙</h1>

<p align="center">为 NLP 模型生成对抗样本</p>

<p align="center">
  <a href="https://textattack.readthedocs.io/">[TextAttack 的 ReadTheDocs 文档]</a> 
  <br> <br>
  <a href="#简介">简介</a> •
  <a href="#环境配置">环境配置</a> •
  <a href="#使用方法textattack---help">使用方法</a> •
  <a href="#设计模式">设计模式</a> 
  <br> <br>
  <a target="_blank">
    <img src="https://github.com/QData/TextAttack/workflows/Github%20PyTest/badge.svg" alt="Github Runner Covergae Status">
  </a>
  <a href="https://badge.fury.io/py/textattack">
    <img src="https://badge.fury.io/py/textattack.svg" alt="PyPI version" height="18">
  </a>
</p>

<img src="http://jackxmorris.com/files/textattack.gif" alt="TextAttack Demo GIF" style="display: block; margin: 0 auto;" />

## 简介

TextAttack 是一个可以实行自然语言处理的 Python 框架，用于方便快捷地进行对抗攻击，增强数据，以及训练模型。

> 如果你在寻找 TextAttacks 支持的预训练模型，请访问 [TextAttack Model Zoo](https://textattack.readthedocs.io/en/latest/3recipes/models.html)。

## Slack 频道

加入[TextAttack Slack](https://join.slack.com/t/textattack/shared_invite/zt-huomtd9z-KqdHBPPu2rOP~Z8q3~urgg) 频道，获取在线帮助与更新提示！

### *选择 TextAttack 的原因*

1. **深入理解 NLP 模型**： 通过使用各种对抗攻击，观察模型的表现
2. **研究与开发 NLP 对抗攻击**： 在你的项目中使用 TextAttack 的框架与组件库
3. **进行数据增强**： 提升模型的泛化性与鲁棒性
3. **训练 NLP 模型**： 只需一行命令，轻松训练模型 (包括下载所有的依赖资源！)

## 环境配置

### 安装

支持 Python 3.6 及以上。支持 CPU ，使用兼容 CUDA 的 GPU ，还可以大幅度提高代码运行速度。使用 pip 轻松安装 TextAttack:

```bash
pip install textattack
```

当 TextAttack 安装完成，可以通过命令行 (`textattack ...`)
或者通过 python 模块 (`python -m textattack ...`) 运行 TextAttack。

> **小提醒**：TextAttack 默认将文件下载保存在 `~/.cache/textattack/` 路径。这些文件包括预训练模型，数据集，以及配置文件 `config.yaml`。若需更改缓存路径，可以通过设置环境变量 `TA_CACHE_DIR`。(例如: `TA_CACHE_DIR=/tmp/ textattack attack ...`).

## 使用方法：`textattack --help`

TextAttack 的主要功能均可通过 `textattack` 命令运行。常用的两个命令为 `textattack attack <args>` 和 `textattack augment <args>`。你可以通过如下命令获取关于所有命令的介绍：
```bash
textattack --help 
```
或者获取具体命令的用法，例如：
```bash
textattack attack --help
```

文件夹 [`examples/`](examples/) 里是一些示例脚本，展示了 TextAttack 的常用方法，包括训练模型，对抗攻击，以及数据增强。[文档网站](https://textattack.readthedocs.io/en/latest) 中有 TextAttack 基本用法的详尽说明与示例，包括自定义攻击的变换与约束。

### 运行对抗攻击：`textattack attack --help`

尝试运行对抗攻击，最快捷的方法是通过命令行接口：`textattack attack` 

> **小提醒**：如果你的机器有多个 GPU，可以通过 `--parallel` 参数将对抗攻击分布在多个 GPU 上。这对一些攻击策略的性能提升巨大。

下面是几个具体的例子：

*对 MR 情感分类数据集上训练的 BERT 模型进行 TextFooler 攻击*: 

```bash
textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 100
```

*对 Quora 问句对数据集上训练的 DistilBERT 模型进行 DeepWordBug 攻击*: 

```bash
textattack attack --model distilbert-base-uncased-qqp --recipe deepwordbug --num-examples 100
```

*对 MR 数据集上训练的 LSTM 模型：设置束搜索宽度为 4，使用词嵌入转换进行无目标攻击*:

```bash
textattack attack --model lstm-mr --num-examples 20 \
 --search-method beam-search^beam_width=4 --transformation word-swap-embedding \
 --constraints repeat stopword max-words-perturbed^max_num_words=2 embedding^min_cos_sim=0.8 part-of-speech \
 --goal-function untargeted-classification
```

> **小提醒**：除了设置具体的数据集与样本数量，你还可以通过传入 `--interactive` 参数，对用户输入的文本进行攻击。

### 攻击策略：`textattack attack --recipe [recipe_name]`

我们实现了一些文献中的攻击策略（Attack recipe）。使用 `textattack list attack-recipes` 命令可以列出所有内置的攻击策略。

运行攻击策略：`textattack attack --recipe [recipe_name]`


<table>
<thead>
<tr class="header">
<th><strong>—————— 攻击策略 ——————</strong></th>
<th><strong>—————— 目标函数 ——————</strong></th>
<th><strong>—————— 约束条件 ——————</strong></th>
<th><strong>—————— 变换方式 ——————</strong></th>
<th><strong>——————— 搜索方法 ———————</strong></th>
<th><strong>主要思想</strong></th>
</tr>
</thead>
<tbody>
  <tr><td colspan="6"><strong><br>对于分类任务的攻击策略，例如情感分类和文本蕴含任务：<br></strong></td></tr>

<tr>
<td><code>alzantot</code>  <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>无目标<br/>{分类，蕴含}</sub></td>
<td><sub>被扰动词的比例，语言模型的困惑度，词嵌入的距离</sub></td>
<td><sub>Counter-fitted 词嵌入替换</sub></td>
<td><sub>遗传算法</sub></td>
<td ><sub>来自 (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998))</sub></td>
</tr>
<tr>
<td><code>bae</code> <span class="citation" data-cites="garg2020bae"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td><sub>USE 通用句子编码向量的 cosine 相似度</sub></td>
<td><sub>BERT 遮罩词预测</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td><sub>使用 BERT 语言模型作为变换的攻击方法，来自 (["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019)](https://arxiv.org/abs/2004.01970)). </sub></td>
</tr>
<tr>
<td><code>bert-attack</code> <span class="citation" data-cites="li2020bertattack"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td><sub>USE 通用句子编码向量的 cosine 相似度, 被扰动词的最大数量</sub></td>
<td><sub>BERT 遮罩词预测 (包括对 subword 的扩充)</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub> (["BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al., 2020)](https://arxiv.org/abs/2004.09984))</sub></td>
</tr>
<tr>
<td><code>checklist</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{无目标，有目标}<br/>分类</sub></td>
<td><sub>checklist 距离</sub></td>
<td><sub>简写，扩写，以及命名实体替换</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>CheckList 中实现的不变性检验(["Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (Ribeiro et al., 2020)](https://arxiv.org/abs/2005.04118))</sub></td>
</tr>
<tr>
<td> <code>clare (*coming soon*)</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>无目标<br/>{分类，蕴含}</sub></td>
<td><sub>RoBERTa 掩码语言模型</sub></td>
<td><sub>词的替换，插入，合并</sub></td>
<td><sub>贪心搜索</sub></td>
<td ><sub>["Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)](https://arxiv.org/abs/2009.07502))</sub></td>
</tr>
<tr>
<td><code>deepwordbug</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{无目标，有目标}<br/>分类</sub></td>
<td><sub>Levenshtein 编辑距离</sub></td>
<td><sub>{字符的插入，删除，替换，以及临近字符交换}</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>贪心搜索 replace-1 分数，多种变换的字符交换式的攻击 (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)</sub></td>
</tr>
<tr>
<td> <code>fast-alzantot</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>无目标<br/>{分类，蕴含}</sub></td>
<td><sub>被扰动词的比例，语言模型的困惑度，词嵌入的距离</sub></td>
<td><sub>Counter-fitted 词嵌入替换</sub></td>
<td><sub>遗传算法</sub></td>
<td ><sub>改进过的更快的 Alzantot et al. 遗传算法, 来自 (["Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)](https://arxiv.org/abs/1909.00986))</sub></td>
</tr>
<tr>
<td><code>hotflip</code> (word swap) <span class="citation" data-cites="Ebrahimi2017HotFlipWA"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td><sub>词嵌入的 cosine 相似度，词性的匹配，被扰动词的数量</sub></td>
<td><sub>基于梯度的词的交换</sub></td>
<td><sub>束搜索</sub></td>
<td ><sub> (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751))</sub></td>
</tr>
<tr>
<td><code>iga</code> <span class="citation" data-cites="iga-wang2019natural"></span></td>
<td><sub>无目标<br/>{分类，蕴含}</sub></td>
<td><sub>被扰动词的比例，词嵌入的距离</sub></td>
<td><sub>Counter-fitted 词嵌入替换</sub></td>
<td><sub>遗传算法</sub></td>
<td ><sub>改进的基于遗传算法的词替换，来自 (["Natural Language Adversarial Attacks and Defenses in Word Level (Wang et al., 2019)"](https://arxiv.org/abs/1909.06723)</sub></td>
</tr>
<tr>
<td><code>input-reduction</code> <span class="citation" data-cites="feng2018pathologies"></span></td>
<td><sub>输入归约</sub></td>
<td></td>
<td><sub>词的删除</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>基于词重要性排序的贪心攻击方法，在缩减输入词的同时保持预测结果不变 (["Pathologies of Neural Models Make Interpretation Difficult" (Feng et al., 2018)](https://arxiv.org/pdf/1804.07781.pdf))</sub></td>
</tr>
<tr>
<td><code>kuleshov</code> <span class="citation" data-cites="Kuleshov2018AdversarialEF"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td><sub>Thought vector 编码的 cosine 相似度, 语言模型给出的相似度概率</sub></td>
<td><sub>Counter-fitted 词嵌入替换</sub></td>
<td><sub>贪心的词的替换</sub></td>
<td ><sub>(["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)) </sub></td>
</tr>
<tr>
<td><code>pruthi</code> <span class="citation" data-cites="pruthi2019combating"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td><sub>词的最短长度，被扰动词的最大数量</sub></td>
<td><sub>{临近字符替换，字符的插入与删除，基于键盘字符位置的字符替换}</sub></td>
<td><sub>贪心搜索</sub></td>
<td ><sub>模拟常见的打字错误 (["Combating Adversarial Misspellings with Robust Word Recognition" (Pruthi et al., 2019)](https://arxiv.org/abs/1905.11268) </sub></td>
</tr>
<tr>
<td><code>pso</code> <span class="citation" data-cites="pso-zang-etal-2020-word"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td></td>
<td><sub>基于 HowNet 的词替换</sub></td>
<td><sub>粒子群优化算法</sub></td>
<td ><sub>(["Word-level Textual Adversarial Attacking as Combinatorial Optimization" (Zang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.540/)) </sub></td>
</tr>
<tr>
<td><code>pwws</code> <span class="citation" data-cites="pwws-ren-etal-2019-generating"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td></td>
<td><sub>基于 WordNet 的同义词替换</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>贪心的攻击方法，基于词重要性排序，词的显著性，以及同义词替换分数(["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/))</sub> </td>
</tr>
<tr>
<td><code>textbugger</code> : (black-box) <span class="citation" data-cites="Li2019TextBuggerGA"></span></td>
<td><sub>无目标<br/>分类</sub></td>
<td><sub>USE 通用句子编码向量的 cosine 相似度</sub></td>
<td><sub>{字符的插入、删除、替换，以及临近字符交换}</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).</sub></td>
</tr>
<tr>
<td><code>textfooler</code> <span class="citation" data-cites="Jin2019TextFooler"></span></td>
<td><sub>无目标<br/>{分类，蕴含}</sub></td>
<td><sub>词嵌入的距离，词性的匹配，USE 通用句子编码向量的 cosine 相似度</sub></td>
<td><sub>Counter-fitted 词嵌入替换</sub></td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>对词重要性排序的贪心攻击方法(["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))</sub> </td>
</tr>

<tr><td colspan="6"><strong><br>对 seq2seq 模型的攻击策略：<br></strong></td></tr>

<tr>
<td><code>morpheus</code> <span class="citation" data-cites="morpheus-tan-etal-2020-morphin"></span></td>
<td><sub>最小 BLEU 分数</sub> </td>
<td></td>
<td><sub>词的屈折变化</sub> </td>
<td><sub>贪心搜索</sub> </td>
<td ><sub>贪心的用词的屈折变化进行替换，来最小化 BLEU 分数(["It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations"](https://www.aclweb.org/anthology/2020.acl-main.263.pdf)</sub> </td>
</tr>

</tr>
<tr>
<td><code>seq2sick</code> :(black-box) <span class="citation" data-cites="cheng2018seq2sick"></span></td>
<td><sub>翻译结果无重叠</sub> </td>
<td></td>
<td><sub>Counter-fitted 词嵌入替换</sub> </td>
<td><sub>对 WIR 的贪心搜索</sub></td>
<td ><sub>贪心攻击方法，以改变全部的翻译结果为目标。目前实现的是黑盒攻击，计划改为与论文中一样的白盒攻击(["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples" (Cheng et al., 2018)](https://arxiv.org/abs/1803.01128)) </sub>  </td>
</tr>

</tbody>
</font>
</table>

> WIR 为 word word importance ranking 的缩写，即词重要性排序。



#### 运行攻击的例子

下面是几个样例，在命令行中验证上述实现的攻击方法:

*对在 SST-2 上精调的 BERT 模型进行 TextFooler 攻击：*
```bash
textattack attack --model bert-base-uncased-sst2 --recipe textfooler --num-examples 10
```

*对用于英语-德语翻译的 T2 模型进行 seq2sick (黑盒) 攻击：*
```bash
 textattack attack --model t5-en-de --recipe seq2sick --num-examples 100
```

### 增强文本数据：`textattack augment`

TextAttack 的组件中，有很多易用的数据增强工具。`textattack.Augmenter` 类使用 *变换* 与一系列的 *约束* 进行数据增强。我们提供了 5 中内置的数据增强策略：
- `textattack.WordNetAugmenter` 通过基于 WordNet 同义词替换的方式增强文本
- `textattack.EmbeddingAugmenter` 通过邻近词替换的方式增强文本，使用 counter-fitted 词嵌入空间中的邻近词进行替换，约束二者的 cosine 相似度不低于 0.8
- `textattack.CharSwapAugmenter` 通过字符的增删改，以及临近字符交换的方式增强文本
- `textattack.EasyDataAugmenter` 通过对词的增删改来增强文本
- `textattack.CheckListAugmenter` 通过简写，扩写以及对实体、地点、数字的替换来增强文本

#### 数据增强的命令行接口
使用 textattack 来进行数据增强，最快捷的方法是通过 `textattack augment <args>` 命令行接口。 `textattack augment` 使用 CSV 文件作为输入，在参数中设置需要增强的文本列，每个样本允许改变的比例，以及对于每个输入样本生成多少个增强样本。输出的结果保存为与输入文件格式一致的 CSV 文件，结果文件中为对指定的文本列生成的增强样本。

比如，对于下面这个 `examples.csv` 文件:

```csv
"text",label
"the rock is destined to be the 21st century's new conan and that he's going to make a splash even greater than arnold schwarzenegger , jean- claud van damme or steven segal.", 1
"the gorgeously elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .", 1
"take care of my cat offers a refreshingly different slice of asian cinema .", 1
"a technically well-made suspenser . . . but its abrupt drop in iq points as it races to the finish line proves simply too discouraging to let slide .", 0
"it's a mystery how the movie could be released in this condition .", 0
```

使用命令 `textattack augment --csv examples.csv --input-column text --recipe embedding --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original`
会增强 `text` 列，约束对样本中 10% 的词进行修改，生成输入数据两倍的样本，同时结果文件中不保存 csv 文件的原始输入。(默认所有结果将会保存在 `augment.csv` 文件中)

数据增强后，下面是 `augment.csv` 文件的内容:
```csv
text,label
"the rock is destined to be the 21st century's newest conan and that he's gonna to make a splashing even stronger than arnold schwarzenegger , jean- claud van damme or steven segal.",1
"the rock is destined to be the 21tk century's novel conan and that he's going to make a splat even greater than arnold schwarzenegger , jean- claud van damme or stevens segal.",1
the gorgeously elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of expression significant adequately describe co-writer/director pedro jackson's expanded vision of j . rs . r . tolkien's middle-earth .,1
the gorgeously elaborate continuation of 'the lordy of the piercings' trilogy is so huge that a column of mots cannot adequately describe co-novelist/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .,1
take care of my cat offerings a pleasantly several slice of asia cinema .,1
taking care of my cat offers a pleasantly different slice of asiatic kino .,1
a technically good-made suspenser . . . but its abrupt drop in iq points as it races to the finish bloodline proves straightforward too disheartening to let slide .,0
a technically well-made suspenser . . . but its abrupt drop in iq dot as it races to the finish line demonstrates simply too disheartening to leave slide .,0
it's a enigma how the film wo be releases in this condition .,0
it's a enigma how the filmmaking wo be publicized in this condition .,0
```

在 'embedding' 增强策略中，使用 counterfitted 词嵌入空间的最近邻来增强数据。

#### 数据增强的 Python 接口
除了使用命令行接口，你还可以在自己的代码中导入 `Augmenter` 来进行动态的数据增强。所有的 `Augmenter` 对象都实现了 `augment` 和 `augment_many` 方法，用于对单个 string 和一个 list 的 string 进行数据增强。下面是在 python 脚本中使用 `EmbeddingAugmenter` 的例子：

```python
>>> from textattack.augmentation import EmbeddingAugmenter
>>> augmenter = EmbeddingAugmenter()
>>> s = 'What I cannot create, I do not understand.'
>>> augmenter.augment(s)
['What I notable create, I do not understand.', 'What I significant create, I do not understand.', 'What I cannot engender, I do not understand.', 'What I cannot creating, I do not understand.', 'What I cannot creations, I do not understand.', 'What I cannot create, I do not comprehend.', 'What I cannot create, I do not fathom.', 'What I cannot create, I do not understanding.', 'What I cannot create, I do not understands.', 'What I cannot create, I do not understood.', 'What I cannot create, I do not realise.']
```
你还可以通过从 `textattack.transformations` 和 `textattack.constraints` 导入 *变换* 与 *约束* 来从头创建自己的数据增强方法。下面是一个使用  `WordSwapRandomCharacterDeletion` *变换* 进行数据增强的例子：

```python
>>> from textattack.transformations import WordSwapRandomCharacterDeletion
>>> from textattack.transformations import CompositeTransformation
>>> from textattack.augmentation import Augmenter
>>> transformation = CompositeTransformation([WordSwapRandomCharacterDeletion()])
>>> augmenter = Augmenter(transformation=transformation, transformations_per_example=5)
>>> s = 'What I cannot create, I do not understand.'
>>> augmenter.augment(s)
['What I cannot creae, I do not understand.', 'What I cannot creat, I do not understand.', 'What I cannot create, I do not nderstand.', 'What I cannot create, I do nt understand.', 'Wht I cannot create, I do not understand.']
```

### 训练模型：`textattack train`

通过 `textattack train` 可以便捷地使用 TextAttack 框架来训练 LSTM，CNN，以及 `transofrmers` 模型。数据集会通过 `datasets` 包自动加载。

#### 运行训练的例子
*在 Yelp 分类数据集上对 TextAttack 中默认的 LSTM 模型训练 50 个 epoch：*
```bash
textattack train --model lstm --dataset yelp_polarity --batch-size 64 --epochs 50 --learning-rate 1e-5
```

训练接口中同样内置了数据增强功能：
```bash
textattack train --model lstm --dataset rotten_tomatoes --augment eda --pct-words-to-swap .1 --transformations-per-example 4
```
上面这个例子在训练之前使用 `EasyDataAugmenter` 策略对 `rotten_tomatoes` 数据集进行数据增强。

*在 `CoLA` 数据集上对 `bert-base` 模型精调 5 个 epoch：*
```bash
textattack train --model bert-base-uncased --dataset glue^cola --batch-size 32 --epochs 5
```


### 检查数据集：`textattack peek-dataset`

使用 `textattack peek-dataset` 可以进一步的观察数据。TextAttack 会打印出数据集粗略的统计信息，包括数据样例，输入文本的统计信息以及标签分布。比如，运行 `textattack peek-dataset --dataset-from-huggingface snli` 命令，会打印指定 NLP 包中 SNLI 数据集的统计信息。


### 列出功能组件：`textattack list`

TextAttack 中有很多组件，有时很难跟进所有组件的情况。你可以使用 `textattack list` 列出所有的组件。比如，列出预训练模型  (`textattack list models`)，或是列出可用的搜索方法 (`textattack list search-methods`)。


## 设计模式


### 模型

TextAttack 不依赖具体模型！你可以使用 TextAttack 来分析任何模型，只要模型的输出是 ID，张量，或者字符串。为了方便使用，TextAttack 内置了常见 NLP 任务的各种预训练模型。你可以轻松愉悦地上手 TextAttack。同时还可以更公平的比较不同文献的 attack 策略。



#### 内置的模型

TextAttack 提供了各种内置模型和数据集。使用 TextAttack 命令行接口，可以自动匹配模型和数据集。
我们为 [GLUE](https://gluebenchmark.com/) 中的九个任务内置了多种预训练模型，并且还内置了很多常见的分类任务、翻译任务和摘要任务的数据集。

[textattack/models/README.md](textattack/models/README.md) 这个列表包含可用的预训练模型以及这些模型的准确率。你还可以通过 `textattack attack --help` 查看完整列表，包括所有的内置模型与数据集。

下面是一个使用内置模型的例子（SST-2 数据集会自动的加载）：
```bash
textattack attack --model roberta-base-sst2 --recipe textfooler --num-examples 10
```

#### HuggingFace 支持 ：`transformers` 模型和 `datasets` 数据集

TextAttack 兼容 [`transformers` 预训练模型](https://huggingface.co/models) 
和 [`datasets` 数据集](https://github.com/huggingface/datasets)! 下面是一个例子，加载数据集并攻击相应预训练模型：

```bash
textattack attack --model-from-huggingface distilbert-base-uncased-finetuned-sst-2-english --dataset-from-huggingface glue^sst2 --recipe deepwordbug --num-examples 10
```

你还可以通过 `--model-from-huggingface` 参数探索更多支持的预训练模型，或是通过 
`--dataset-from-huggingface` 参数指定其他数据集。


#### 加载本地文件中的模型与数据集

你可以快捷地对本地模型或数据样本进行攻击：创建一个简单的文件就可以加载预训练模型，然后在文件中可以通过对象 `model` 与 `tokenizer` 对象加载模型。`tokenizer` 对象必须实现 `encode()` 方法，该方法将输入字符串转为一个列表或一个 ID 张量。`model` 对象必须通过实现 `__call__` 方法来加载模型。

##### 使用本地模型
对于你已经训练完成的模型，可以通过创建下面这样的文件，将其命名为 `my_model.py`:

```python
model = load_your_model_with_custom_code() # replace this line with your model loading code
tokenizer = load_your_tokenizer_with_custom_code() # replace this line with your tokenizer loading code
```

然后，在运行攻击时指定参数 `--model-from-file my_model.py`，就可以自动载入你的模型与分词器。

### 数据集

#### 使用本地数据集

加载本地数据集与加载本地预训练模型的方法相似。`dataset` 对象可以是任意可迭代的`(input, output)` 对。下面这个例子演示了如何在 `my_dataset.py` 脚本中加载一个情感分类数据集：

```python
dataset = [('Today was....', 1), ('This movie is...', 0), ...]
```

然后，在运行攻击时指定参数 `--dataset-from-file my_dataset.py`，就可以对这个本地数据集进行攻击。

#### 通过 AttackedText 类调用数据集

为了对分词后的句子运行攻击方法，我们设计了 `AttackedText` 对象。它同时维护 token 列表与含有标点符号的原始文本。我们使用这个对象来处理原始的与分词后的文本。

#### 通过 Data Frames 调用数据集(*即將上線*)


### 何为攻击 & 如何设计新的攻击 

`Attack` 中的 `attack_one` 方法以 `AttackedText` 对象作为输入，若攻击成功，返回 `SuccessfulAttackResult`，若攻击失败，返回 `FailedAttackResult`。


我们将攻击划分并定义为四个组成部分：**目标函数** 定义怎样的攻击是一次成功的攻击，**约束条件** 定义怎样的扰动是可行的，**变换规则** 对输入文本生成一系列可行的扰动结果，**搜索方法** 在搜索空间中遍历所有可行的扰动结果。每一次攻击都尝试对输入的文本添加扰动，使其通过目标函数（即判断攻击是否成功），并且扰动要符合约束（如语法约束，语义相似性约束）。最后用搜索方法在所有可行的变换结果中，挑选出优质的对抗样本。


这种模块化的设计可以将各种对抗攻击策略整合在一个系统里。这使得我们可以方便地将文献中的方法集成在一起，同时复用攻击策略之间相同的部分。我们已经实现了 16 种简明易读的攻击策略（见上表）。史上首次！各种攻击方法终于可以在标准的设置下作为基准方法，进行比较与分析。


TextAttack 是不依赖具体模型的，这意味着可以对任何深度学习框架训练的模型进行攻击。只要被攻击的模型可以读取字符串（或一组字符串），并根据目标函数返回一个结果。比如说，机器翻译模型读取一句话，返回一句对应的翻译结果。分类或蕴含任务的模型输入字符串，返回一组分数。只要你的模型满足这两点，就可以使用 TextAttack 进行攻击。



### 目标函数

目标函数 `GoalFunction` 以 `AttackedText` 对象作为输入，为输入对象打分，并且判别这次攻击是否满足目标函数定义的成功条件，返回一个 `GoalFunctionResult` 对象。

### 约束条件

约束条件 `Constraint` 以 `AttackedText` 对象作为输入，返回一个变换后的 `AttackedText` 列表。对于每条变换，返回一个布尔值表示这条变换是否满足约束条件。

### 变换规则

变换规则 `Transformation` 以 `AttackedText` 对象作为输入，返回对于 `AttackedText` 所有可行变换的列表。例如，一个变换规则可以是返回所有可能的同义词替换结果。

### 搜索方法

搜索方法 `SearchMethod` 以初始的 `GoalFunctionResult` 作为输入，返回最终的 `GoalFunctionResult`。`get_transformations` 方法，以一个 `AttackedText` 对象作为输入，返还所有符合约束条件的变换结果。搜索方法不断地调用 `get_transformations` 函数，直到攻击成功 (由 `get_goal_results` 决定) 或搜索结束。

### 公平比较攻击策略（Benchmarking Attacks）

- 详细情况参见我们的分析文章：Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples at [EMNLP BlackBoxNLP](https://arxiv.org/abs/2009.06368). 

- 正如我们在上面的文章中所强调的，我们不推荐在对攻击策略没有约束的情况下直接进行比较。

- 对这点进行强调，是由于最近的文献中在设置约束时使用了不同的方法或者阈值。在不固定约束空间时，攻击成功率的增加可能是源于改进的搜索方法或变换方式，又或是降低了对搜索空间的约束。

## 帮助改进 TextAttack

我们欢迎任何建议与改进！请提交 Issues（议题）和 Pull requests（拉取请求），我们会竭尽所能的做出即时反馈。TextAttack 当前处于 "alpha" 版本，我们仍在完善它的设计与功能。

关于提交建议与改进的详细指引，查看 [CONTRIBUTING.md](https://github.com/QData/TextAttack/blob/master/CONTRIBUTING.md) 。

## 引用 TextAttack

如果 TextAttack 对你的研究工作有所帮助，欢迎在论文中引用 [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909)。

```bibtex
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Jake Grigsby and Di Jin and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


