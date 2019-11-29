<h1 align="center">TextAttack 🐙</h1>

<p align="center">generating adversarial examples for NLP models</p>

<p align="center">
  <a href="#about">About</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#features">Features</a> •
  
</p>

![TextAttack Demo GIF](https://github.com/UVA-MachineLearningBioinformatics/TextAttack/blob/readme_gif/text_attack_demo.gif?raw=true)

## About

TextAttack is research library built for finding adversarial examples in NLP. TextAttack includes implementations of state-of-the-art attacks on NLP models, both black-box and white-box. It ships with pre-trained models and datasets for various tasks.

## Getting Started

### Installation

You should be running Python 3.6+ to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed. After cloning this git repository, run the following commands to install the `textattack` page a `conda` environment:

```
conda create -n text-attack python=3.7
conda activate text-attack
pip install -e .
```

We use the list of stopwords from nltk. To download them run in Python shell:

```
import nltk
nltk.download('stopwords')
```

We use spaCy's English model. To download it, after installing spaCy type in terminal:

```
python -m spacy download en
```

### Basic Usage

We have a command-line interface for running different attacks on different datasets. Run it with default arguments with `python textattack/run_attack.py`. See help info and list of arguments with `python textattack/run_attack.py --help`.

### Common Errors

#### Errors regarding GCC
If you see an error that GCC is incompatible, make sure your system has an up-to-date version of the GCC compiler. On distributed systems with a `module` system, typing `module load gcc` may be sufficient.

#### Errors regarding Java
Using the LanguageTool constraint relies on Java 8 internally (it's not ideal, we know). Please install Java 8 if you're interested in using the LanguageTool grammaticality constraint. If your system supports the `module` command, try typing `module load java8`.

## Features

### Models

We've also included a few pre-trained models for common tasks that you can download and run out-of-the-box. However, TextAttack is *model_agnostic*! Anything that overrides __call__, takes in tokenized text, and outputs probabilities works for us. This includes your favorite models in both Pytorch and Tensorflow.

@TODO show examples of each in `examples/`.

### Attack Methods


#### Black-box attacks

- Greedy selection over all input words
- Greedy selection with Word Importance Ranking (["Is Bert Really Robust?" (Jin et al., 2019)](https://arxiv.org/abs/1907.11932))
- Genetic algorithm (["Generating Natural Language Adversarial Examples" (Alzantot et al., 2018)](https://arxiv.org/abs/1804.07998))

#### White-box attacks

- HotFlip (["HotFlip: White-Box Adversarial Examples for Text Classification" (Ebrahimi et al., 2018)](https://arxiv.org/abs/1712.06751)

### Transformations

A black-box attack method selects the best possible transformation from a set of all transformations.

We support the following transformations:

- Homoglyph swap
- Counterfit synonym swap

### Constraints

Text input perturbations are only valid if they meet the constraints of the attack. This library supports constraints on the semantic, syntactic, and morphological level. Some constraints measure changes in the entire input, while others only take into account substitutions of a single word.

#### Semantics
- Word embedding nearest-neighbor distance
- Universal Sentence Encoder
- Google 1-billion words language model

#### Syntax
- Grammatical errors measured with [LanguageTool](https://languagetool.org/)

### Datasets

We include a few popular datasets to get you started.

- The [Yelp Sentiment Analysis dataset](https://www.yelp.com/dataset/challenge) includes inputs from yelp reviews manually labeled as positive (1) or negative (0).
- IMDB Movie Review Sentiment
- Movie Review Sentiment
- Kaggle Fake News

### Models 
Out of the box, textattack comes with several pretrained models for each dataset.
