

<h1 align="center">TextAttack 🐙</h1>

<p align="center">Generating adversarial examples for NLP models</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#design">Design</a> 
  <br> <br>
  <a target="_blank" href="https://travis-ci.org/QData/TextAttack">
    <img src="https://travis-ci.org/QData/TextAttack.svg?branch=master" alt="Coverage Status">
  </a>

</p>
  
## About

TextAttack is a Python framework for running adversarial attacks against NLP models. TextAttack builds attacks from four components: a serach method, goal function, transformation, and set of constraints. TextAttack's modular design makes it easily extensible to new NLP tasks, models, and attack strategies. TextAttack currently supports attacks on models trained for classification, entailment, and translation.

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

The [`examples/`](examples/) folder contains notebooks walking through examples of basic usage of TextAttack, including building a custom transformation and a custom constraint.

We also have a command-line interface for running attacks. See help info and list of arguments with `python -m textattack --help`.

### Attack Recipes

We include attack recipes which build an attack such that only one command line argument has to be passed. To run an attack recipes, run `python -m textattack --recipe [recipe_name]`
Currently, we include six recipes, all synonym substitution-based.

The first are for classification and entailment attacks:
- **textfooler**: Greedy attack with word importance ranking (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932)).
- **alzantot**: Genetic algorithm attack from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998)).
- **tf-adjusted**: TextFooler attack with constraint thresholds adjusted based on human evaluation and grammaticality enforced.
- **alz-adjusted**: Alzantot's attack adjusted to follow the same constraints as tf-adjusted such that the only difference is the search method.
- **deepwordbug**: Replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers (Gao et al., 2018)"](https://arxiv.org/abs/1801.04354)).
- **hotflip**: Beam search and gradient-based word swap (["HotFlip: White-Box Adversarial Examples for Text Classification (Ebrahimi et al., 2017)"](https://arxiv.org/abs/1712.06751)).
- **kuleshov**: Greedy search and counterfitted embedding swap (["Adversarial Examples for Natural Language Classification Problems (Kuleshov et al., 2018)"](https://openreview.net/pdf?id=r1QZ3zbAZ)).

The final is for translation attacks:
- **seq2sick**: Greedy attack with goal of changing every word in the output translation. Currently implemented as black-box with plans to change to white-box as done in paper (["Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples (Cheng et al., 2018)"](https://arxiv.org/abs/1803.01128)).

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

To allow for word replacement after a sequence has been tokenized, we include a `TokenizedText` object which maintains both a list of tokens and the original text, with punctuation. We use this object in favor of a list of words or just raw text.

### Models and Datasets

TextAttack is model-agnostic! Anything that overrides `__call__`, takes in `TokenizedText`, and correctly formats output works. However, TextAttack provides pre-trained models and samples for the following datasets:

#### Classification:
* AG News dataset topic classification
* IMDB dataset sentiment classification
* Movie Review dataset sentiment classification
* Yelp dataset sentiment classification

#### Entailment:
* SNLI datastet
* MNLI dataset (matched & unmatched)

#### Translation:
* newstest2013 English to German dataset

### Attacks

The `attack_one` method in an `Attack` takes as input a `TokenizedText`, and outputs either a `SuccessfulAttackResult` if it succeeds or a `FailedAttackResult` if it fails. We formulate an attack as consisting of four components: a **goal function** which determines if the attack has succeeded, **constraints** defining which perturbations are valid, a **transformation** that generates potential modifications given an input, and a **search method** which traverses through the search space of possible perturbations. 

### Goal Functions

A `GoalFunction` takes as input a `TokenizedText` object and the ground truth output, and determines whether the attack has succeeded. 

### Constraints

A `Constraint` takes as input an original `TokenizedText`, and a list of transformed `TokenizedText`s. For each transformed option, it returns a boolean representing whether the constraint is met.

### Transformations

A `Transformation` takes as input a `TokenizedText` and returns a list of possible transformed `TokenizedText`s. For example, a transformation might return all possible synonym replacements.

### Search Methods

A search method is currently implemented in an extension of the `Attack` class, through implementing the `attack_one` method. The `get_transformations` function takes as input a `TokenizedText` object and outputs a list of possible transformations filtered by meeting all of the attack’s constraints. A search consists of successive calls to `get_transformations` until the search succeeds or is exhausted.

## Contributing to TextAttack

We welcome contributions and suggestions! Submit a pull request or issue and we will do our best to respond in a timely manner.

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


