.. TextAttack documentation master file, created by
   sphinx-quickstart on Sat Oct 19 20:54:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TextAttack
======================================

`TextAttack <https://github.com/QData/TextAttack>`__ is a Python library for running adversarial attacks against NLP models. These may be useful for evaluating attack methods and evaluating model robustness. TextAttack is designed in order to be easily extensible to new NLP tasks, models, attack methods, and attack constraints. The separation between these aspects of an adversarial attack and standardization of constraint evaluation allows for easier ablation studies. TextAttack supports attacks on models trained for classification, entailment, and translation. 

Features 
-----------

TextAttack isn't just a Python library; it's a framework for constructing adversarial attacks in NLP. TextAttack builds attacks from four components:

- **Goal Functions** stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output
- **Search Methods** explores the space of transformations and attempt to find a successful perturbtion
- **Transformations** takes a text input and transform it by replacing characters, words, or phrases
- **Constraints**: Determines if a potential perturbtion is valid with respect to the original input

TextAttack provides a set of **attack recipes** that assemble attacks from the literature from these four components.

TextAttack has some other features that make it a pleasure to use:

- **Data augmentation** using transformations & constraints
- **Built-in Datasets** for running attacks without supplying your own data
- **Pre-trained Models** for testing attacks and evaluating constraints
- **Built-in tokenizers** so you don't have to worry about tokenizing the inputs
- **Visualization options** like Visdom and Weights & Biases


.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart/installation
   quickstart/overview
   
.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/1_Introduction_and_Transformtions.ipynb
   examples/2_Constraints.ipynb


.. toctree::
   :maxdepth: 2
   :caption: Search Methods:

   search_methods/attack
   search_methods/attack_result

.. toctree::
   :maxdepth: 2
   :caption: Transformations:

   transformations/transformation

.. toctree::
   :maxdepth: 2
   :caption: Constraints:

   constraints/constraint
   constraints/semantics
   constraints/grammaticality
   constraints/overlap


.. toctree::
   :maxdepth: 2
   :caption: Goal Functions:

   goal_functions/goal_functions
   goal_functions/goal_function_results

.. toctree::
   :maxdepth: 2
   :caption: Loggers:

   loggers/loggers

   
.. toctree::
   :maxdepth: 2
   :caption: Attack Recipes:

   attack_recipes/attack_recipes


.. toctree::
   :maxdepth: 2
   :caption: Tokenizers & Tokenized Text:

   tokenizers/tokenized_text
   tokenizers/tokenizer


.. toctree::
   :maxdepth: 2
   :caption: Models:

   models/bert
   models/lstm
   models/cnn


.. toctree::
   :maxdepth: 2
   :caption: Datasets:

   datasets/datasets


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
