<h1 align="center">TextAttack 🐙</h1>

<p align="center">Generating adversarial examples for NLP models</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#setup">Setup</a> •
  <a href="#usage">Usage</a> •
  <a href="#design">Design</a> 
  <br> <br>
  <a target="_blank" href="https://travis-ci.com/UVA-MachineLearningBioinformatics/TextAttack">
    <img src="https://travis-ci.com/UVA-MachineLearningBioinformatics/TextAttack.svg?token=Kpx8qxw1sbsdXyhVxRq3&branch=master" alt="Coverage Status">
  </a>

</p>
  
## About

TextAttack is a library for running adversarial attacks against NLP models. These may be useful for evaluating attack methods and evaluating model robustness. TextAttack is designed in order to be easily extensible to new NLP tasks, models, attack methods, and attack constraints. The separation between these aspects of an adversarial attack and standardization of constraint evaluation allows for easier ablation studies. TextAttack supports attacks on models trained for classification and entailment.

## Setup

### Installation

You should be running Python 3.6+ to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed. After cloning this git repository, run the following commands to install the `textattack` page a `conda` environment:

```
conda create -n text-attack python=3.7
conda activate text-attack
pip install -e .
```

We use the NLTK package for its list of stopwords and access to the WordNet lexical database. To download them run in Python shell:

```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

We use spaCy's English model. To download it, after installing spaCy run:

```
python -m spacy download en
```

### Cache
TextAttack provides pretrained models and datasets for user convenience. By default, all this stuff is downloaded to `~/.cache`. You can change this location by editing the `CACHE_DIR` field in `config.json`.

### Common Errors

#### Errors regarding GCC
If you see an error that GCC is incompatible, make sure your system has an up-to-date version of the GCC compiler.

#### Errors regarding Java
Using the LanguageTool constraint relies on Java 8 internally (it's not ideal, we know). Please install Java 8 if you're interested in using the LanguageTool grammaticality constraint.

## Usage

### Basic Usage

![TextAttack Demo GIF](https://i.imgur.com/hOCDhQf.gif)

We have a command-line interface for running different attacks on different datasets. Run it with default arguments with `python scripts/run_attack.py`. See help info and list of arguments with `python scripts/run_attack.py --help`.

### Attack Recipes

We include attack recipes which build an attack such that only one command line argument has to be passed. To run an attack recipes, run `python scripts/run_attack.py --recipe [recipe_name]`
Currently, we include four recipes, each for synonym substitution-based classification and entailment attacks:
- **deepwordbug**: Replace-1 scoring and multi-transformation character-swap attack (["Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers"](https://arxiv.org/abs/1801.04354))
- **textfooler**: Greedy attack with word importance ranking (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))
- **alzantot**: Genetic algorithm attack from (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998))
- **tf-adjusted**: TextFooler attack with constraint thresholds adjusted based on human evaluation and grammaticality enforced.
- **alz-adjusted**: Alzantot's attack adjusted to follow the same constraints as tf-adjusted such that the only difference is the search method.

### Adding to TextAttack

Clone the repository, add your code, and run run\_attack to get results. If you would like your contribution to be added to the library, submit a pull request. Instructions for using TextAttack as a pip library coming soon!

## Design

### TokenizedText

To allow for word replacement after a sequence has been tokenized, we include a TokenizedText object which maintains both a list of tokens and the original text, with punctuation. We use this object in favor of a list of words or just raw text.

### Models and Datasets

We've included a few pretrained models that you can download and run out-of-the-box. However, TextAttack is model agnostic! Anything that overrides \_\_call\_\_, takes in tokenized text, and outputs probabilities works. 

### Attacks

Attacks all take as input a TokenizedText, and output either an AttackResult if it succeeds or a FailedAttackResult if it fails. We split attacks into black box, which only have access to the model’s call function, and white box, which have access to the whole model. For standardization and ease of ablation, we formulate an attack as a series of transformations in a search space, subject to certain constraints. An attack may call get\_transformations for a given transformation to get a list of possible transformations filtered by meeting all of the attack’s constraints.

### Transformations

Transformations take as input a TokenizedText and return a list of possible transformed TokenizedTexts. For example, a transformation might return all possible synonym replacements.

### Constraints

Constraints take as input an original TokenizedText, and a list of transformed TokenizedTexts. For each transformed option, the constraint returns a boolean representing whether the constraint is met.
