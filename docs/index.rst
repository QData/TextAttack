.. TextAttack documentation master file, created by
   sphinx-quickstart on Sat Oct 19 20:54:30 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TextAttack
======================================

`TextAttack <https://github.com/QData/TextAttack>`__ repository is a library for running adversarial attacks against NLP models. These may be useful for evaluating attack methods and evaluating model robustness. TextAttack is designed in order to be easily extensible to new NLP tasks, models, attack methods, and attack constraints. The separation between these aspects of an adversarial attack and standardization of constraint evaluation allows for easier ablation studies. TextAttack supports attacks on models trained for classification and entailment.

Features 
-----------

- **Search Methods**: Explores the transformation space and attempts to find a successful attack
- **Transformations**: Takes a text input and transforms it by replacing words and phrases while attempting to retain the meaning
- **Constraints**: Determines if a given transformation is valid
- **Built-in Datasets** and **Pre-trained Models** for ease of use


.. toctree::
   :maxdepth: 2
   :caption: Quickstart

   quickstart/installation
   quickstart/examples


.. toctree::
   :maxdepth: 2
   :caption: Search Methods:

   search_methods/attack

.. toctree::
   :maxdepth: 2
   :caption: Attack Recipes:

   attack_recipes/alzantot_genetic
   attack_recipes/gao_deepwordbug
   attack_recipes/jin_textfooler


.. toctree::
   :maxdepth: 2
   :caption: Attack Results:

   attack_results/attack_result 


.. toctree::
   :maxdepth: 2
   :caption: Transformations:

   transformations/transformation
   transformations/composite_transformation
   transformations/word_swap 


.. toctree::
   :maxdepth: 2
   :caption: Constraints:

   contraints/constraint
   contraints/semantics
   contraints/syntax
   contraints/overlap


.. toctree::
   :maxdepth: 2
   :caption: Goal Functions:

   goal_functions/goal_functions


.. toctree::
   :maxdepth: 2
   :caption: Tokenizers:

   tokenizers/tokenizer 
   tokenizers/bert_tokenizer
   tokenizers/spacy_tokenizer


.. toctree::
   :maxdepth: 2
   :caption: Models:

   models/bert
   models/lstm
   models/cnn


.. toctree::
   :maxdepth: 2
   :caption: Datasets:

   datasets/generic
   datasets/classification
   datasets/entailment 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
