<h1 align="center">TextAttack 🐙</h1>

<p align="center">Generating adversarial examples for NLP models</p>

<p align="center">
  <a href="https://textattack.readthedocs.io/">[TextAttack Documentation on ReadTheDocs]</a> 
  <br> <br>
  <a href="#about">About</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#design">Design</a> 
  <br> <br>
  <a target="_blank">
    <img src="https://github.com/QData/TextAttack/workflows/Github%20PyTest/badge.svg" alt="Github Runner Covergae Status">
  </a>
  <a href="https://badge.fury.io/py/textattack">
    <img src="https://badge.fury.io/py/textattack.svg" alt="PyPI version" height="18">
  </a>
</p>

<img src="http://jackxmorris.com/files/textattack.gif" alt="TextAttack Demo GIF" style="display: block; margin: 0 auto;" />
  
## About

TextAttack is a Python framework for adversarial attacks, data augmentation, and model training in NLP.

> If you're looking for information about TextAttack's menagerie of pre-trained models, you might want the [TextAttack Model Zoo](https://textattack.readthedocs.io/en/latest/3recipes/models.html) page.

## Slack Channel

For help and realtime updates related to TextAttack, please [join the TextAttack Slack](https://join.slack.com/t/textattack/shared_invite/zt-huomtd9z-KqdHBPPu2rOP~Z8q3~urgg)!

### *Why TextAttack?*

There are lots of reasons to use TextAttack:

1. **Understand NLP models better** by running different adversarial attacks on them and examining the output
2. **Research and develop different NLP adversarial attacks** using the TextAttack framework and library of components
3. **Augment your dataset** to increase model generalization and robustness downstream
3. **Train NLP models** using just a single command (all downloads included!)

## Setup

### Installation

You should be running Python 3.6+ to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed. TextAttack is available through pip:

```bash
pip install textattack
```

Once TextAttack is installed, you can run it via command-line (`textattack ...`)
or via python module (`python -m textattack ...`).

> **Tip**: TextAttack downloads files to `~/.cache/textattack/` by default. This includes pretrained models, 
> dataset samples, and the configuration file `config.yaml`. To change the cache path, set the 
> environment variable `TA_CACHE_DIR`. (for example: `TA_CACHE_DIR=/tmp/ textattack attack ...`).

## Usage:  `textattack --help`

TextAttack's main features can all be accessed via the `textattack` command. Two very
common commands are `textattack attack <args>`, and `textattack augment <args>`. You can see more
information about all commands using 
```bash
textattack --help 
```
or a specific command using, for example,
```bash
textattack attack --help
```

The [`examples/`](examples/) folder includes scripts showing common TextAttack usage for training models, running attacks, and augmenting a CSV file. The [documentation website](https://textattack.readthedocs.io/en/latest) contains walkthroughs explaining basic usage of TextAttack, including building a custom transformation and a custom constraint..

### Running Attacks: `textattack attack --help`

The easiest way to try out an attack is via the command-line interface, `textattack attack`. 

> **Tip:** If your machine has multiple GPUs, you can distribute the attack across them using the `--parallel` option. For some attacks, this can really help performance.

Here are some concrete examples:

*TextFooler on BERT trained on the MR sentiment classification dataset*: 
```bash
textattack attack --recipe textfooler --model bert-base-uncased-mr --num-examples 100
```

*DeepWordBug on DistilBERT trained on the Quora Question Pairs paraphrase identification dataset*: 
```bash
textattack attack --model distilbert-base-uncased-qqp --recipe deepwordbug --num-examples 100
```

*Beam search with beam width 4 and word embedding transformation and untargeted goal function on an LSTM*:
```bash
textattack attack --model lstm-mr --num-examples 20 \
 --search-method beam-search^beam_width=4 --transformation word-swap-embedding \
 --constraints repeat stopword max-words-perturbed^max_num_words=2 embedding^min_cos_sim=0.8 part-of-speech \
 --goal-function untargeted-classification
```

> **Tip:** Instead of specifying a dataset and number of examples, you can pass `--interactive` to attack samples inputted by the user.

### Attacks and Papers Implemented ("Attack Recipes"): `textattack attack --recipe [recipe_name]`

We include attack recipes which implement attacks from the literature. You can list attack recipes using `textattack list attack-recipes`.

To run an attack recipe: `textattack attack --recipe [recipe_name]`


<table  style="width:100%" border="1">
<thead>
<tr class="header">
<th><strong>Attack Recipe Name</strong></th>
<th><strong>Goal Function</strong></th>
<th><strong>ConstraintsEnforced</strong></th>
<th><strong>Transformation</strong></th>
<th><strong>Search Method</strong></th>
<th><strong>Main Idea</strong></th>
</tr>
</thead>
<tbody>
  <tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on classification tasks, like sentiment classification and entailment:<br></strong></td></tr>

<tr>
<td><code>alzantot</code>  <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998))</sub></td>
</tr>
<tr>
<td><code>bae</code> <span class="citation" data-cites="garg2020bae"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>BERT Masked Token Prediction</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>BERT masked language model transformation attack from (["BAE: BERT-based Adversarial Examples for Text Classification" (Garg & Ramakrishnan, 2019)](https://arxiv.org/abs/2004.01970)). </td>
</tr>
<tr>
<td><code>bert-attack</code> <span class="citation" data-cites="li2020bertattack"></span></td>
<td><sub>Untargeted Classification</td>
<td><sub>USE sentence encoding cosine similarity, Maximum number of words perturbed</td>
<td><sub>BERT Masked Token Prediction (with subword expansion)</td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub> (["BERT-ATTACK: Adversarial Attack Against BERT Using BERT" (Li et al., 2020)](https://arxiv.org/abs/2004.09984))</sub></td>
</tr>
<tr>
<td><code>checklist</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>checklist distance</sub></td>
<td><sub>contract, extend, and substitutes name entities</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Invariance testing implemented in CheckList . (["Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (Ribeiro et al., 2020)](https://arxiv.org/abs/2005.04118))</sub></td>
</tr>
<tr>
<td> <code>clare</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>RoBERTa Masked Prediction for token swap, insert and merge</sub></td>
<td><sub>Greedy</sub></td>
<td ><sub>["Contextualized Perturbation for Textual Adversarial Attack" (Li et al., 2020)](https://arxiv.org/abs/2009.07502))</sub></td>
</tr>
<tr>
<td><code>deepwordbug</code> <span class="citation" data-cites="Gao2018BlackBoxGO"></span></td>
<td><sub>{Untargeted, Targeted} Classification</sub></td>
<td><sub>Levenshtein edit distance</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)</sub></td>
</tr>
<tr>
<td> <code>fast-alzantot</code> <span class="citation" data-cites="Alzantot2018GeneratingNL Jia2019CertifiedRT"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Language Model perplexity, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Modified, faster version of the Alzantot et al. genetic algorithm, from (["Certified Robustness to Adversarial Word Substitutions" (Jia et al., 2019)](https://arxiv.org/abs/1909.00986))</sub></td>
</tr>
<tr>
<td><code>hotflip</code> (word swap) <span class="citation" data-cites="Ebrahimi2017HotFlipWA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Word Embedding Cosine Similarity, Part-of-speech match, Number of words perturbed</sub></td>
<td><sub>Gradient-Based Word Swap</sub></td>
<td><sub>Beam search</sub></td>
<td ><sub> (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751))</sub></td>
</tr>
<tr>
<td><code>iga</code> <span class="citation" data-cites="iga-wang2019natural"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Percentage of words perturbed, Word embedding distance</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Genetic Algorithm</sub></td>
<td ><sub>Improved genetic algorithm -based word substitution from (["Natural Language Adversarial Attacks and Defenses in Word Level (Wang et al., 2019)"](https://arxiv.org/abs/1909.06723)</sub></td>
</tr>
<tr>
<td><code>input-reduction</code> <span class="citation" data-cites="feng2018pathologies"></span></td>
<td><sub>Input Reduction</sub></td>
<td></td>
<td><sub>Word deletion</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking , Reducing the input while maintaining the prediction through word importance ranking (["Pathologies of Neural Models Make Interpretation Difficult" (Feng et al., 2018)](https://arxiv.org/pdf/1804.07781.pdf))</sub></td>
</tr>
<tr>
<td><code>kuleshov</code> <span class="citation" data-cites="Kuleshov2018AdversarialEF"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Thought vector encoding cosine similarity, Language model similarity probability</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy word swap</sub></td>
<td ><sub>(["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)) </sub></td>
</tr>
<tr>
<td><code>pruthi</code> <span class="citation" data-cites="pruthi2019combating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>Minimum word length, Maximum number of words perturbed</sub></td>
<td><sub>{Neighboring Character Swap, Character Deletion, Character Insertion, Keyboard-Based Character Swap}</sub></td>
<td><sub>Greedy search</sub></td>
<td ><sub>simulates common typos (["Combating Adversarial Misspellings with Robust Word Recognition" (Pruthi et al., 2019)](https://arxiv.org/abs/1905.11268) </sub></td>
</tr>
<tr>
<td><code>pso</code> <span class="citation" data-cites="pso-zang-etal-2020-word"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>HowNet Word Swap</sub></td>
<td><sub>Particle Swarm Optimization</sub></td>
<td ><sub>(["Word-level Textual Adversarial Attacking as Combinatorial Optimization" (Zang et al., 2020)](https://www.aclweb.org/anthology/2020.acl-main.540/)) </sub></td>
</tr>
<tr>
<td><code>pwws</code> <span class="citation" data-cites="pwws-ren-etal-2019-generating"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td></td>
<td><sub>WordNet-based synonym swap</sub></td>
<td><sub>Greedy-WIR (saliency)</sub></td>
<td ><sub>Greedy attack with word importance ranking based on word saliency and synonym swap scores (["Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency" (Ren et al., 2019)](https://www.aclweb.org/anthology/P19-1103/))</sub> </td>
</tr>
<tr>
<td><code>textbugger</code> : (black-box) <span class="citation" data-cites="Li2019TextBuggerGA"></span></td>
<td><sub>Untargeted Classification</sub></td>
<td><sub>USE sentence encoding cosine similarity</sub></td>
<td><sub>{Character Insertion, Character Deletion, Neighboring Character Swap, Character Substitution}</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).</sub></td>
</tr>
<tr>
<td><code>textfooler</code> <span class="citation" data-cites="Jin2019TextFooler"></span></td>
<td><sub>Untargeted {Classification, Entailment}</sub></td>
<td><sub>Word Embedding Distance, Part-of-speech match, USE sentence encoding cosine similarity</sub></td>
<td><sub>Counter-fitted word embedding swap</sub></td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with word importance ranking  (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))</sub> </td>
</tr>

<tr><td style="text-align: center;" colspan="6"><strong><br>Attacks on sequence-to-sequence models: <br></strong></td></tr>

<tr>
<td><code>morpheus</code> <span class="citation" data-cites="morpheus-tan-etal-2020-morphin"></span></td>
<td><sub>Minimum BLEU Score</sub> </td>
<td></td>
<td><sub>Inflection Word Swap</sub> </td>
<td><sub>Greedy search</sub> </td>
<td ><sub>Greedy to replace words with their inflections with the goal of minimizing BLEU score (["It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations"](https://www.aclweb.org/anthology/2020.acl-main.263.pdf)</sub> </td>
</tr>

</tr>
<tr>
<td><code>seq2sick</code> :(black-box) <span class="citation" data-cites="cheng2018seq2sick"></span></td>
<td><sub>Non-overlapping output</sub> </td>
<td></td>
<td><sub>Counter-fitted word embedding swap</sub> </td>
<td><sub>Greedy-WIR</sub></td>
<td ><sub>Greedy attack with goal of changing every word in the output translation. Currently implemented as black-box with plans to change to white-box as done in paper (["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples" (Cheng et al., 2018)](https://arxiv.org/abs/1803.01128)) </sub>  </td>
</tr>


</tbody>
</font>
</table>



#### Recipe Usage Examples

Here are some examples of testing attacks from the literature from the command-line:

*TextFooler against BERT fine-tuned on SST-2:*
```bash
textattack attack --model bert-base-uncased-sst2 --recipe textfooler --num-examples 10
```

*seq2sick (black-box) against T5 fine-tuned for English-German translation:*
```bash
 textattack attack --model t5-en-de --recipe seq2sick --num-examples 100
```

### Augmenting Text: `textattack augment`

Many of the components of TextAttack are useful for data augmentation. The `textattack.Augmenter` class
uses a transformation and a list of constraints to augment data. We also offer five built-in recipes
for data augmentation:
- `textattack.WordNetAugmenter` augments text by replacing words with WordNet synonyms
- `textattack.EmbeddingAugmenter` augments text by replacing words with neighbors in the counter-fitted embedding space, with a constraint to ensure their cosine similarity is at least 0.8
- `textattack.CharSwapAugmenter` augments text by substituting, deleting, inserting, and swapping adjacent characters
- `textattack.EasyDataAugmenter` augments text with a combination of word insertions, substitutions and deletions.
- `textattack.CheckListAugmenter` augments text by contraction/extension and by substituting names, locations, numbers.

#### Augmentation Command-Line Interface
The easiest way to use our data augmentation tools is with `textattack augment <args>`. `textattack augment`
takes an input CSV file and text column to augment, along with the number of words to change per augmentation
and the number of augmentations per input example. It outputs a CSV in the same format with all the augmentation
examples corresponding to the proper columns.

For example, given the following as `examples.csv`:

```csv
"text",label
"the rock is destined to be the 21st century's new conan and that he's going to make a splash even greater than arnold schwarzenegger , jean- claud van damme or steven segal.", 1
"the gorgeously elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .", 1
"take care of my cat offers a refreshingly different slice of asian cinema .", 1
"a technically well-made suspenser . . . but its abrupt drop in iq points as it races to the finish line proves simply too discouraging to let slide .", 0
"it's a mystery how the movie could be released in this condition .", 0
```

The command `textattack augment --csv examples.csv --input-column text --recipe embedding --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original`
will augment the `text` column by altering 10% of each example's words, generating twice as many augmentations as original inputs, and exclude the original inputs from the
output CSV. (All of this will be saved to `augment.csv` by default.)

After augmentation, here are the contents of `augment.csv`:
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

The 'embedding' augmentation recipe uses counterfitted embedding nearest-neighbors to augment data.

#### Augmentation Python Interface
In addition to the command-line interface, you can augment text dynamically by importing the
`Augmenter` in your own code. All `Augmenter` objects implement `augment` and `augment_many` to generate augmentations
of a string or a list of strings. Here's an example of how to use the `EmbeddingAugmenter` in a python script:

```python
>>> from textattack.augmentation import EmbeddingAugmenter
>>> augmenter = EmbeddingAugmenter()
>>> s = 'What I cannot create, I do not understand.'
>>> augmenter.augment(s)
['What I notable create, I do not understand.', 'What I significant create, I do not understand.', 'What I cannot engender, I do not understand.', 'What I cannot creating, I do not understand.', 'What I cannot creations, I do not understand.', 'What I cannot create, I do not comprehend.', 'What I cannot create, I do not fathom.', 'What I cannot create, I do not understanding.', 'What I cannot create, I do not understands.', 'What I cannot create, I do not understood.', 'What I cannot create, I do not realise.']
```
You can also create your own augmenter from scratch by importing transformations/constraints from `textattack.transformations` and `textattack.constraints`. Here's an example that generates augmentations of a string using `WordSwapRandomCharacterDeletion`:

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

### Training Models:  `textattack train`

Our model training code is available via `textattack train` to help you train LSTMs,
CNNs, and `transformers` models using TextAttack out-of-the-box. Datasets are
automatically loaded using the `datasets` package.

#### Training Examples
*Train our default LSTM for 50 epochs on the Yelp Polarity dataset:*
```bash
textattack train --model lstm --dataset yelp_polarity --batch-size 64 --epochs 50 --learning-rate 1e-5
```

The training process has data augmentation built-in:
```bash
textattack train --model lstm --dataset rotten_tomatoes --augment eda --pct-words-to-swap .1 --transformations-per-example 4
```
This uses the `EasyDataAugmenter` recipe to augment the `rotten_tomatoes` dataset before training.

*Fine-Tune `bert-base` on the `CoLA` dataset for 5 epochs**:
```bash
textattack train --model bert-base-uncased --dataset glue^cola --batch-size 32 --epochs 5
```


### To check datasets: `textattack peek-dataset`

To take a closer look at a dataset, use `textattack peek-dataset`. TextAttack will print some cursory statistics about the inputs and outputs from the dataset. For example, `textattack peek-dataset --dataset-from-huggingface snli` will show information about the SNLI dataset from the NLP package.


### To list functional components: `textattack list`

There are lots of pieces in TextAttack, and it can be difficult to keep track of all of them. You can use `textattack list` to list components, for example, pretrained models (`textattack list models`) or available search methods (`textattack list search-methods`).


## Design


### Models 

TextAttack is model-agnostic!  You can use `TextAttack` to analyze any model that outputs IDs, tensors, or strings. To help users, TextAttack includes pre-trained models for different common NLP tasks. This makes it easier for
users to get started with TextAttack. It also enables a more fair comparison of attacks from
the literature.



#### Built-in Models and Datasets

TextAttack also comes built-in with models and datasets. Our command-line interface will automatically match the correct 
dataset to the correct model. We include 82 different (Oct 2020) pre-trained models for each of the nine [GLUE](https://gluebenchmark.com/) 
tasks, as well as some common datasets for classification, translation, and summarization. 

A list of available pretrained models and their validation accuracies is available at
[textattack/models/README.md](textattack/models/README.md). You can also view a full list of provided models 
& datasets via `textattack attack --help`.

Here's an example of using one of the built-in models (the SST-2 dataset is automatically loaded):

```bash
textattack attack --model roberta-base-sst2 --recipe textfooler --num-examples 10
```

#### HuggingFace support: `transformers` models and `datasets` datasets

We also provide built-in support for [`transformers` pretrained models](https://huggingface.co/models) 
and datasets from the [`datasets` package](https://github.com/huggingface/datasets)! Here's an example of loading
and attacking a pre-trained model and dataset:

```bash
textattack attack --model-from-huggingface distilbert-base-uncased-finetuned-sst-2-english --dataset-from-huggingface glue^sst2 --recipe deepwordbug --num-examples 10
```

You can explore other pre-trained models using the `--model-from-huggingface` argument, or other datasets by changing 
`--dataset-from-huggingface`.


#### Loading a model or dataset from a file

You can easily try out an attack on a local model or dataset sample. To attack a pre-trained model,
create a short file that loads them as variables `model` and `tokenizer`.  The `tokenizer` must
be able to transform string inputs to lists or tensors of IDs using a method called `encode()`. The
model must take inputs via the `__call__` method.

##### Model from a file
To experiment with a model you've trained, you could create the following file
and name it `my_model.py`:

```python
model = load_your_model_with_custom_code() # replace this line with your model loading code
tokenizer = load_your_tokenizer_with_custom_code() # replace this line with your tokenizer loading code
```

Then, run an attack with the argument `--model-from-file my_model.py`. The model and tokenizer will be loaded automatically.



### Custom Datasets


#### Dataset from a file

Loading a dataset from a file is very similar to loading a model from a file. A 'dataset' is any iterable of `(input, output)` pairs.
The following example would load a sentiment classification dataset from file `my_dataset.py`:

```python
dataset = [('Today was....', 1), ('This movie is...', 0), ...]
```

You can then run attacks on samples from this dataset by adding the argument `--dataset-from-file my_dataset.py`.


#### Dataset via AttackedText class

To allow for word replacement after a sequence has been tokenized, we include an `AttackedText` object
which maintains both a list of tokens and the original text, with punctuation. We use this object in favor of a list of words or just raw text.



#### Dataset via Data Frames (*coming soon*)



### Attacks and how to design a new attack 

The `attack_one` method in an `Attack` takes as input an `AttackedText`, and outputs either a `SuccessfulAttackResult` if it succeeds or a `FailedAttackResult` if it fails. 


We formulate an attack as consisting of four components: a **goal function** which determines if the attack has succeeded, **constraints** defining which perturbations are valid, a **transformation** that generates potential modifications given an input, and a **search method** which traverses through the search space of possible perturbations. The attack attempts to perturb an input text such that the model output fulfills the goal function (i.e., indicating whether the attack is successful) and the perturbation adheres to the set of constraints (e.g., grammar constraint, semantic similarity constraint). A search method is used to find a sequence of transformations that produce a successful adversarial example.


This modular design unifies adversarial attack methods into one system, enables us to easily assemble attacks from the literature while re-using components that are shared across attacks. We provides clean, readable implementations of 16 adversarial attack recipes from the literature (see above table). For the first time, these attacks can be benchmarked, compared, and analyzed in a standardized setting.


TextAttack is model-agnostic - meaning it can run attacks on models implemented in any deep learning framework. Model objects must be able to take a string (or list of strings) and return an output that can be processed by the goal function. For example, machine translation models take a list of strings as input and produce a list of strings as output. Classification and entailment models return an array of scores. As long as the user's model meets this specification, the model is fit to use with TextAttack.



#### Goal Functions

A `GoalFunction` takes as input an `AttackedText` object, scores it, and determines whether the attack has succeeded, returning a `GoalFunctionResult`.

#### Constraints

A `Constraint` takes as input a current `AttackedText`, and a list of transformed `AttackedText`s. For each transformed option, it returns a boolean representing whether the constraint is met.

#### Transformations

A `Transformation` takes as input an `AttackedText` and returns a list of possible transformed `AttackedText`s. For example, a transformation might return all possible synonym replacements.

#### Search Methods

A `SearchMethod` takes as input an initial `GoalFunctionResult` and returns a final `GoalFunctionResult` The search is given access to the `get_transformations` function, which takes as input an `AttackedText` object and outputs a list of possible transformations filtered by meeting all of the attack’s constraints. A search consists of successive calls to `get_transformations` until the search succeeds (determined using `get_goal_results`) or is exhausted.


### Benchmarking Attacks

- See our analysis paper: Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples at [EMNLP BlackBoxNLP](https://arxiv.org/abs/2009.06368). 

- As we emphasized in the above paper, we don't recommend to directly compare Attack Recipes out of the box. 

- This comment is due to that attack recipes in the recent literature used different ways or thresholds in setting up their constraints. Without the constraint space held constant, an increase in attack success rate could come from an improved search or transformation method or a less restrictive search space. 


## Multi-lingual Support

- See [README_ZH.md](https://github.com/QData/TextAttack/blob/master/README_ZH.md) for our README in Chinese 






## Contributing to TextAttack

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner. TextAttack is currently in an "alpha" stage in which we are working to improve its capabilities and design.

See [CONTRIBUTING.md](https://github.com/QData/TextAttack/blob/master/CONTRIBUTING.md) for detailed information on contributing.

## Citing TextAttack

If you use TextAttack for your research, please cite [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909).

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


