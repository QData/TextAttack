

<h1 align="center">TextAttack 🐙</h1>

<p align="center">Generating adversarial examples for NLP models</p>

<p align="center">
  <a href="https://textattack.readthedocs.io/">Docs</a> •
  <a href="#about">About</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#design">Design</a> 
  <br> <br>
  <a target="_blank" href="https://travis-ci.org/QData/TextAttack">
    <img src="https://travis-ci.org/QData/TextAttack.svg?branch=master" alt="Coverage Status">
  </a>
  <a href="https://badge.fury.io/py/textattack">
    <img src="https://badge.fury.io/py/textattack.svg" alt="PyPI version" height="18">
  </a>

</p>
  
## About

TextAttack is a Python framework for running adversarial attacks against NLP models. TextAttack builds attacks from four components: a search method, goal function, transformation, and set of constraints. TextAttack's modular design makes it easily extensible to new NLP tasks, models, and attack strategies. TextAttack currently supports attacks on models trained for classification, entailment, and translation.

## Setup

### Installation

You should be running Python 3.6+ to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed. TextAttack is available through pip:

```bash
pip install textattack
```

Once TextAttack is installed, you can run it via command-line (`textattack ...`)
or via the python module (`python -m textattack ...`).

> Note on dependencies: TextAttack consists of different modules, many of which require their own specific dependencies. For this reason, we leave it up to the user to install specific required packages on their own. If you'd like to install all of the optional dependencies at once, clone the repository and install with `pip install -e . "[.full]"` 
### Configuration
TextAttack downloads files to `~/.cache/textattack/` by default. This includes pretrained models, 
dataset samples, and the configuration file `config.yaml`. To change the cache path, set the 
environment variable `TA_CACHE_DIR`.

## Usage

TextAttack's main features can all be accessed via the `textattack` command. Two very
common commands are `textattack attack <args>`, and `textattack augment <args>`. You can see more
information about all commands using `textattack --help`, or a specific command using, for example,
`textattack attack --help`.

### Running Attacks

The [`examples/`](docs/examples/) folder contains notebooks explaining basic usage of TextAttack, including building a custom transformation and a custom constraint. These examples can also be viewed through the [documentation website](https://textattack.readthedocs.io/en/latest).

The easiest way to try out an attack is via the command-line interface, `textattack attack`. Here are some concrete examples:

*TextFooler on an LSTM trained on the MR sentiment classification dataset*: 
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
 --search-method beam-search:beam_width=4 --transformation word-swap-embedding \
 --constraints repeat stopword max-words-perturbed:max_num_words=2 embedding:min_cos_sim=0.8 part-of-speech \
 --goal-function untargeted-classification
```

*Non-overlapping output attack using a greedy word swap and WordNet word substitutions on T5 English-to-German translation:*
```bash
textattack attack --attack-n --goal-function non-overlapping-output \
    --model t5-en2de --num-examples 10 --transformation word-swap-wordnet \
    --constraints edit-distance:12 max-words-perturbed:max_percent=0.75 repeat stopword \
    --search greedy
```

> **Tip:** If your machine has multiple GPUs, you can distribute the attack across them using the `--parallel` option. For some attacks, this can really help performance.

### Attacks and Papers Implemented ("Attack Recipes")

We include attack recipes which implement attacks from the literature. You can list attack recipes using `textattack list attack-recipes`.

To run an attack recipe: `textattack attack --recipe [recipe_name]`

The first are for classification tasks, like sentiment classification and entailment:
- **alzantot**: Genetic algorithm attack from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998)).
- **deepwordbug**: Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)).
- **hotflip**: Beam search and gradient-based word swap (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751)).
- **kuleshov**: Greedy search and counterfitted embedding swap (["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)).
- **textbugger**: Greedy attack with word importance ranking and character-based swaps ([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).
- **textfooler**: Greedy attack with word importance ranking and counter-fitted embedding swap (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932)).

The final is for sequence-to-sequence models:
- **seq2sick**: Greedy attack with goal of changing every word in the output translation. Currently implemented as black-box with plans to change to white-box as done in paper (["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples" (Cheng et al., 2018)](https://arxiv.org/abs/1803.01128)).

#### Recipe Usage Examples

Here are some exampes of testing attacks from the literature from the command-line:

*TextFooler against BERT fine-tuned on SST-2:*
```bash
textattack attack --model bert-base-uncased-sst2 --recipe textfooler --num-examples 10
```

*seq2sick (black-box) against T5 fine-tuned for English-German translation:*
```bash
textattack attack --recipe seq2sick --model t5-en2de --num-examples 100
```

### Augmenting Text

Many of the components of TextAttack are useful for data augmentation. The `textattack.Augmenter` class
uses a transformation and a list of constraints to augment data. We also offer three built-in recipes
for data augmentation:
- `textattack.WordNetAugmenter` augments text by replacing words with WordNet synonyms
- `textattack.EmbeddingAugmenter` augments text by replacing words with neighbors in the counter-fitted embedding space, with a constraint to ensure their cosine similarity is at least 0.8
- `textattack.CharSwapAugmenter` augments text by substituting, deleting, inserting, and swapping adjacent characters

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

The command `textattack augment --csv examples.csv --input-column text --recipe embedding --num-words-to-swap 4 --transformations-per-example 2 --exclude-original`
will augment the `text` column with four swaps per augmentation, twice as many augmentations as original inputs, and exclude the original inputs from the
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

## Design

### TokenizedText

To allow for word replacement after a sequence has been tokenized, we include a `TokenizedText` object
which maintains both a list of tokens and the original text, with punctuation. We use this object in favor of a list of words or just raw text.

### Models and Datasets

TextAttack is model-agnostic! You can use `TextAttack` to analyze any model that outputs IDs, tensors, or strings.

#### Built-in Models

TextAttack also comes built-in with models and datasets. Our command-line interface will automatically match the correct 
dataset to the correct model. We include various pre-trained models for each of the nine [GLUE](https://gluebenchmark.com/) 
tasks, as well as some common classification datasets, translation, and summarization. You can 
see the full list of provided models & datasets via `textattack attack --help`.

Here's an example of using one of the built-in models:

```bash
textattack attack --model roberta-base-sst2 --recipe textfooler --num-examples 10
```

#### HuggingFace support: `transformers` models and `nlp` datasets

We also provide built-in support for [`transformers` pretrained models](https://huggingface.co/models) 
and datasets from the [`nlp` package](https://github.com/huggingface/nlp)! Here's an example of loading
and attacking a pre-trained model and dataset:

```bash
textattack attack --model_from_huggingface distilbert-base-uncased-finetuned-sst-2-english --dataset_from_nlp glue:sst2 --recipe deepwordbug --num-examples 10
```

You can explore other pre-trained models using the `--model_from_huggingface` argument, or other datasets by changing 
`--dataset_from_nlp`.


#### Loading a model or dataset from a file

You can easily try out an attack on a local model or dataset sample. To attack a pre-trained model,
create a short file that loads them as variables `model` and `tokenizer`.  The `tokenizer` must
be able to transform string inputs to lists or tensors of IDs using a method called `encode()`. The
model must take inputs via the `__call__` method.

##### Model from a file
To experiment with a model you've trained, you could create the following file
and name it `my_model.py`:

```python
model = load_model()
tokenizer = load_tokenizer()
```

Then, run an attack with the argument `--model_from_file my_model.py`. The model and tokenizer will be loaded automatically.

#### Dataset from a file

Loading a dataset from a file is very similar to loading a model from a file. A 'dataset' is any iterable of `(input, output)` pairs.
The following example would load a sentiment classification dataset from file `my_dataset.py`:

```python
dataset = [('Today was....', 1), ('This movie is...', 0), ...]
```

You can then run attacks on samples from this dataset by adding the argument `--dataset_from_file my_dataset.py`.

### Attacks

The `attack_one` method in an `Attack` takes as input a `TokenizedText`, and outputs either a `SuccessfulAttackResult` if it succeeds or a `FailedAttackResult` if it fails. We formulate an attack as consisting of four components: a **goal function** which determines if the attack has succeeded, **constraints** defining which perturbations are valid, a **transformation** that generates potential modifications given an input, and a **search method** which traverses through the search space of possible perturbations. 

### Goal Functions

A `GoalFunction` takes as input a `TokenizedText` object and the ground truth output, and determines whether the attack has succeeded, returning a `GoalFunctionResult`.

### Constraints

A `Constraint` takes as input a current `TokenizedText`, and a list of transformed `TokenizedText`s. For each transformed option, it returns a boolean representing whether the constraint is met.

### Transformations

A `Transformation` takes as input a `TokenizedText` and returns a list of possible transformed `TokenizedText`s. For example, a transformation might return all possible synonym replacements.

### Search Methods

A `SearchMethod` takes as input an initial `GoalFunctionResult` and returns a final `GoalFunctionResult` The search is given access to the `get_transformations` function, which takes as input a `TokenizedText` object and outputs a list of possible transformations filtered by meeting all of the attack’s constraints. A search consists of successive calls to `get_transformations` until the search succeeds (determined using `get_goal_results`) or is exhausted.

## Contributing to TextAttack

We welcome suggestions and contributions! Submit an issue or pull request and we will do our best to respond in a timely manner. TextAttack is currently in an "alpha" stage in which we are working to improve its capabilities and design.

## Citing TextAttack

If you use TextAttack for your research, please cite [TextAttack: A Framework for Adversarial Attacks in Natural Language Processing](https://arxiv.org/abs/2005.05909).

```bibtex
@misc{Morris2020TextAttack,
    Author = {John X. Morris and Eli Lifland and Jin Yong Yoo and Yanjun Qi},
    Title = {TextAttack: A Framework for Adversarial Attacks in Natural Language Processing},
    Year = {2020},
    Eprint = {arXiv:2005.05909},
}
```


