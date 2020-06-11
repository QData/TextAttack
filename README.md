

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

```
pip install textattack
```

### Configuration
TextAttack downloads files to `~/.cache/textattack/` by default. This includes pretrained models, 
dataset samples, and the configuration file `config.yaml`. To change the cache path, set the 
environment variable `TA_CACHE_DIR`.

## Usage

### Running Attacks

The [`examples/`](docs/examples/) folder contains notebooks walking through examples of basic usage of TextAttack, including building a custom transformation and a custom constraint. These examples can also be viewed through the [documentation website](https://textattack.readthedocs.io/en/latest).

We also have a command-line interface for running attacks. See help info and list of arguments with `python -m textattack --help`.

#### Sample Attack Commands

*TextFooler on an LSTM trained on the MR sentiment classification dataset*: 
```
python -m textattack --recipe textfooler --model bert-base-uncased-mr --num-examples 100
```

*DeepWordBug on DistilBERT trained on the Quora Question Pairs paraphrase identification dataset*: 
```
python -m textattack --model distilbert-base-uncased-qqp --recipe deepwordbug --num-examples 100
```

*Beam search with beam width 4 and word embedding transformation and untargeted goal function on an LSTM*:
```
python -m textattack --model lstm-mr --num-examples 20 \
 --search-method beam-search:beam_width=4 --transformation word-swap-embedding \
 --constraints repeat stopword max-words-perturbed:max_num_words=2 embedding:min_cos_sim=0.8 part-of-speech \
 --goal-function untargeted-classification
```

*Non-overlapping output attack using a greedy word swap and WordNet word substitutionson T5 English-to-German translation:*
```
python -m textattack --attack-n --goal-function non-overlapping-output \
    --model t5-en2de --num-examples 10 --transformation word-swap-wordnet \
    --constraints edit-distance:12 max-words-perturbed:max_percent=0.75 repeat stopword \
    --search greedy
```

### Attacks and Papers Implemented ("Attack Recipes")

We include attack recipes which build an attack such that only one command line argument has to be passed. To run an attack recipes, run `python -m textattack --recipe [recipe_name]`

The first are for classification tasks, like sentiment classification and entailment:
- **alzantot**: Genetic algorithm attack from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998)).
- **deepwordbug**: Greedy replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers" (Gao et al., 2018)](https://arxiv.org/abs/1801.04354)).
- **hotflip**: Beam search and gradient-based word swap (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2017)](https://arxiv.org/abs/1712.06751)).
- **kuleshov**: Greedy search and counterfitted embedding swap (["Adversarial Examples for Natural Language Classification Problems" (Kuleshov et al., 2018)](https://openreview.net/pdf?id=r1QZ3zbAZ)).
- **textbugger**: Greedy attack with word importance ranking and character-based swaps ([(["TextBugger: Generating Adversarial Text Against Real-world Applications" (Li et al., 2018)](https://arxiv.org/abs/1812.05271)).
- **textfooler**: Greedy attack with word importance ranking and counter-fitted embedding swap (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932)).

The final is for sequence-to-sequence models:
- **seq2sick**: Greedy attack with goal of changing every word in the output translation. Currently implemented as black-box with plans to change to white-box as done in paper (["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples" (Cheng et al., 2018)](https://arxiv.org/abs/1803.01128)).

### Augmenting Text

Many of the components of TextAttack are useful for data augmentation. The `textattack.Augmenter` class
uses a transformation and a list of constraints to augment data. We also offer three built-in recipes
for data augmentation:
- `textattack.WordNetAugmenter` augments text by replacing words with WordNet synonyms
- `textattack.EmbeddingAugmenter` augments text by replacing words with neighbors in the counter-fitted embedding space, with a constraint to ensure their cosine similarity is at least 0.8
- `textattack.CharSwapAugmenter` augments text by substituting, deleting, inserting, and swapping adjacent characters

All `Augmenter` objects implement `augment` and `augment_many` to generate augmentations
of a string or a list of strings. Here's an example of how to use the `EmbeddingAugmenter`:

```
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
tasks, as well as some common classification datasets (MR, IMDB, Yelp, AGNews), translation, and summarization. You can 
see the full list of provided models & datasets via `python -m textattack --help`.

Here's an example of using one of the built-in models:

```
pythom -m textattack --model roberta-base-sst2 --recipe textfooler --num-examples 10
```

#### HuggingFace support: `transformers` models and `nlp` datasets

We also provide built-in support for [`transformers` pretrained models](https://huggingface.co/models) 
and datasets from the [`nlp` package](https://github.com/huggingface/nlp)! Here's an example of loading
and attacking a pre-trained model and dataset:

```
python -m textattack --model_from_huggingface distilbert-base-uncased-finetuned-sst-2-english --dataset_from_nlp glue:sst2 --recipe deepwordbug --num-examples 10
```

You can explore other pre-trained models using the `--model_from_huggingface` argument, or other datasets by changing 
`--dataset_from_nlp`.


#### Loading a model or dataset from a file

You can easily try out an attack on a local model or dataset sample. To attack a pre-trained model,
create a short file that loads them as variables `model` and `tokenizer`.  The `tokenizer` must
be able to transform string inputs to lists or tensors of IDs using a method called `encode()`. The
model must take inputs via the `__call__` method.

##### Model from a file
, you could create the following file
and name it `my_model.py`:

```
model = load_model()
tokenizer = load_tokenizer()
```

Then, run an attack with the argument `--model_from_file my_model.py`. The model and tokenizer will be loaded automatically.

#### Dataset from a file

Loading a dataset from a file is very similar to loading a model from a file. A 'dataset' is any iterable of `(input, output)` pairs.
The following example would load a sentiment classification dataset from file `my_dataset.py`:

```
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


