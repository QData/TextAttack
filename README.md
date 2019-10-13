# TextAttack

## Installation

You must be running Python 3.6+ to use this package. A CUDA-compatible GPU is 
optional but will greatly improve code speed.

'conda create -n text-attack python=3.7'
'conda activate text-attack'
`pip install -r requirements.txt`
'mkdir outputs'


## Features

We include a few popular datasets to get you started.

#### Yelp Sentiment Analysis

### Models

We've also included a few pre-trained models for common
tasks that you can download and run out-of-the-box. However,
anything that overrides __call__, takes in tokenized text, and 
outputs probabilities works for us. This includes your favorite
models in both Pytorch and Tensorflow.

@TODO show examples of each in `examples/`.

### Attack Search Methods

### Transformations

### Constraints

#### Semantics

#### Syntax

#### Morphology

### Datasets
