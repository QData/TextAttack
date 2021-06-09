Command-Line Usage
=======================================

The easiest way to use textattack is from the command-line. Installing textattack
will provide you with the handy `textattack` command which will allow you to do
just about anything TextAttack offers in a single bash command.

> *Tip*: If you are for some reason unable to use the `textattack` command, you
> can access all the same functionality by prepending `python -m` to the command
> (`python -m textattack ...`).

To see all available commands, type `textattack --help`. This page explains
some of the most important functionalities of textattack: NLP data augmentation,
adversarial attacks, and training and evaluating models.

## Data Augmentation with `textattack augment`

The easiest way to use our data augmentation tools is with `textattack augment <args>`. `textattack augment`
takes an input CSV file and text column to augment, along with the percentage of words to change per augmentation
and the number of augmentations per input example. It outputs a CSV in the same format with all the augmentation
examples corresponding to the proper columns.

For example, given the following as `examples.csv`:

```
"text",label
"the rock is destined to be the 21st century's new conan and that he's going to make a splash even greater than arnold schwarzenegger , jean- claud van damme or steven segal.", 1
"the gorgeously elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .", 1
"take care of my cat offers a refreshingly different slice of asian cinema .", 1
"a technically well-made suspenser . . . but its abrupt drop in iq points as it races to the finish line proves simply too discouraging to let slide .", 0
"it's a mystery how the movie could be released in this condition .", 0
```

The command: 
```
textattack augment --csv examples.csv --input-column text --recipe eda --pct-words-to-swap .1 \
--transformations-per-example 2 --exclude-original
``` 
will augment the `text` column with 10% of words edited per augmentation, twice as many augmentations as original inputs, and exclude the original inputs from the
output CSV. (All of this will be saved to `augment.csv` by default.)

After augmentation, here are the contents of `augment.csv`:
```
text,label
"the rock is destined to be the 21st century's new conan and that he's to make splash even greater arnold schwarzenegger , jean- claud van damme or steven segal.",1
"the Arnold rock is destined to be the 21st vanguard century's new specify conan and that he's going to make a splash even greater than arnold schwarzenegger , jean- claud van damme or steven segal.",1
the gorgeously continuation of 'the lord of the rings' trilogy is so huge that a column of cannot adequately describe co-writer/ peter jackson's expanded vision of j . r . r . tolkien's middle-earth .,1
the splendidly elaborate continuation of 'the lord of the rings' trilogy is so huge that a column of parole cannot adequately describe co-writer/director peter jackson's expanded vision of J . r . r . tolkien's middle-earth .,1
take care of my cat offers a refreshingly slice different of asian cinema .,1
take care of my cast offers a refreshingly different slice of asian cinema .,1
a technically well-made suspenser . . . but its abrupt drop in iq points as it races to the finish line proves simply too discouraging to rush let IT slide .,0 
a technically well-made suspenser . . . but its abrupt drop in iq points as it races to the finish line proves just too discouraging to let chute .,0 
it's a mystery how the movie could this released in be condition .,0
it's a whodunit how the movie could be released in this condition .,0
```

The 'eda' augmentation uses a combination of word swaps, insertions, and substitutions to generate new examples.

## Adversarial Attacks with `textattack attack`

The heart of textattack is running adversarial attacks on NLP models with 
`textattack attack`. You can build an attack from the command-line in several ways:
1. Use an **attack recipe** to launch an attack from the literature: `textattack attack --recipe deepwordbug`
2. Build your attack from components: 
```
textattack attack --model lstm-mr --num-examples 20 --search-method beam-search^beam_width=4 \
--transformation word-swap-embedding \
--constraints repeat stopword max-words-perturbed^max_num_words=2 embedding^min_cos_sim=0.8 part-of-speech \
--goal-function untargeted-classification
```
3. Create a python file that builds your attack and load it: `textattack attack --attack-from-file my_file.py^my_attack_name`

## Training Models with `textattack train`

With textattack, you can train models on any classification or regression task
from [`datasets`](https://github.com/huggingface/datasets/) using a single line.

### Available Models
#### TextAttack Models
TextAttack has two build-in model types, a 1-layer bidirectional LSTM with a hidden
state size of 150 (`lstm`), and a WordCNN with 3 window sizes
(3, 4, 5) and 100 filters for the window size (`cnn`). Both models set dropout
to 0.3 and use a base of the 200-dimensional GLoVE embeddings.

#### `transformers` Models
Along with the `lstm` and `cnn`, you can theoretically fine-tune any model based
in the huggingface [transformers](https://github.com/huggingface/transformers/)
repo. Just type the model name (like `bert-base-cased`) and it will be automatically 
loaded.

Here are some models from transformers that have worked well for us:
- `bert-base-uncased` and `bert-base-cased`
- `distilbert-base-uncased` and `distilbert-base-cased`
- `albert-base-v2` 
- `roberta-base` 
- `xlnet-base-cased`

## Evaluating Models with `textattack eval-model`

Any TextAttack-compatible model can be evaluated using `textattack eval-model`. TextAttack-trained models can be evaluated using `textattack eval --num-examples <num-examples> --model /path/to/trained/model/`

## Other Commands

### Checkpoints and `textattack attack-resume`

Some attacks can take a very long time. Sometimes this is because they're using
a very slow search method (like beam search with a high beam width) or sometimes
they're just attacking a large number of samples. In these cases, it can be 
useful to save attack checkpoints throughout the course of the attack. Then,
if the attack crashes for some reason, you can resume without restarting from
scratch.

- To save checkpoints while running an attack, add the argument `--checkpoint-interval X`,
where X is the number of attacks you want to run between checkpoints (for example `textattack attack <args> --checkpoint-interval 5`).
- To load an attack from a checkpoint, use `textattack attack-resume --checkpoint-file <checkpoint-file>`.

### Listing features with `textattack list`

TextAttack has a lot of built-in features (models, search methods, constraints, etc.)
and it can get overwhelming to keep track of all the options. To list all of the
options within a given category, use `textattack list`.

For example:
- list all the built-in models: `textattack list models`
- list all constraints: `textattack list constraints`
- list all search methods: `textattack list search-methods`

### Examining datasets with `textattack peek-dataset`
It can be useful to take a cursory look at and compute some basic statistics of
whatever dataset you're working with. Whether you're loading a dataset of your
own from a file, or one from NLP, you can use `textattack peek-dataset` to 
see some basic information about the dataset.

For example, use `textattack peek-dataset --dataset-from-huggingface glue^mrpc` to see
information about the MRPC dataset (from the GLUE set of datasets). This will
print statistics like the number of labels, average number of words, etc.


