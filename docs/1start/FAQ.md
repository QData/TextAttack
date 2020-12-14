Frequently Asked Questions
========================================

## Via Slack: Where to Ask Questions: 

For help and realtime updates related to TextAttack, please [join the TextAttack Slack](https://join.slack.com/t/textattack/shared_invite/zt-huomtd9z-KqdHBPPu2rOP~Z8q3~urgg)!


## Via CLI: `--help`

+ Easiest self help:   `textattack --help`
+ More concrete self help: 
  - `textattack attack --help`  
  - `textattack augment --help`
  - `textattack train --help`
  - `textattack peek-dataset --help`
  - `textattack list`, e.g., `textattack list search-methods`


## Via our papers: More details on results  
+ [references](https://textattack.readthedocs.io/en/latest/1start/references.html)


## Via readthedocs: More details on APIs
+ [complete API reference on TextAttack](https://textattack.readthedocs.io/en/latest/apidoc/textattack.html) 


## More Concrete Questions: 


### 1. How to Train

For example, you can *Train our default LSTM for 50 epochs on the Yelp Polarity dataset:*
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




### 2. Use Custom  Models  

TextAttack is model-agnostic!  You can use `TextAttack` to analyze any model that outputs IDs, tensors, or strings. To help users, TextAttack includes pre-trained models for different common NLP tasks. This makes it easier for
users to get started with TextAttack. It also enables a more fair comparison of attacks from the literature. A list of available pretrained models and their validation accuracies is available at [HERE](https://textattack.readthedocs.io/en/latest/3recipes/models.html).


You can easily try out an attack on a local model you prefer. To attack a pre-trained model, create a short file that loads them as variables `model` and `tokenizer`.  The `tokenizer` must
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

TextAttack is model-agnostic - meaning it can run attacks on models implemented in any deep learning framework. Model objects must be able to take a string (or list of strings) and return an output that can be processed by the goal function. For example, machine translation models take a list of strings as input and produce a list of strings as output. Classification and entailment models return an array of scores. As long as the user's model meets this specification, the model is fit to use with TextAttack.


### 3. Use Custom Datasets 


#### From a file

Loading a dataset from a file is very similar to loading a model from a file. A 'dataset' is any iterable of `(input, output)` pairs.
The following example would load a sentiment classification dataset from file `my_dataset.py`:

```python
dataset = [('Today was....', 1), ('This movie is...', 0), ...]
```

You can then run attacks on samples from this dataset by adding the argument `--dataset-from-file my_dataset.py`.



#### Custom Dataset via AttackedText class

To allow for word replacement after a sequence has been tokenized, we include an `AttackedText` object
which maintains both a list of tokens and the original text, with punctuation. We use this object in favor of a list of words or just raw text.


#### Custome Dataset via Data Frames or other python data objects (*coming soon*)


### 4. Benchmarking Attacks

- See our analysis paper: Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples at [EMNLP BlackBoxNLP](https://arxiv.org/abs/2009.06368). 

- As we emphasized in the above paper, we don't recommend to directly compare Attack Recipes out of the box. 

- This comment is due to that attack recipes in the recent literature used different ways or thresholds in setting up their constraints. Without the constraint space held constant, an increase in attack success rate could from an improved search or transformation method or a less restrictive search space. 


### 5. Create Custom or New Attacks

The `attack_one` method in an `Attack` takes as input an `AttackedText`, and outputs either a `SuccessfulAttackResult` if it succeeds or a `FailedAttackResult` if it fails. 

- [Here is an example of using TextAttack to create a new attack method](https://github.com/jxmorris12/second-order-adversarial-examples) 


We formulate an attack as consisting of four components: a **goal function** which determines if the attack has succeeded, **constraints** defining which perturbations are valid, a **transformation** that generates potential modifications given an input, and a **search method** which traverses through the search space of possible perturbations. The attack attempts to perturb an input text such that the model output fulfills the goal function (i.e., indicating whether the attack is successful) and the perturbation adheres to the set of constraints (e.g., grammar constraint, semantic similarity constraint). A search method is used to find a sequence of transformations that produce a successful adversarial example.


This modular design unifies adversarial attack methods into one system, enables us to easily assemble attacks from the literature while re-using components that are shared across attacks. We provides clean, readable implementations of 16 adversarial attack recipes from the literature (see [our tool paper](https://arxiv.org/abs/2005.05909) and [our benchmark search paper](https://arxiv.org/abs/2009.06368)). For the first time, these attacks can be benchmarked, compared, and analyzed in a standardized setting.



