TextAttack
======================================

`TextAttack <https://github.com/QData/TextAttack>`__ is a Python framework for adversarial attacks and data augmentation in NLP.

NLP Attacks
-----------

TextAttack provides a framework for constructing and thinking about attacks via perturbation in NLP. TextAttack builds attacks from four components:

- `Goal Functions <attacks/goal_function.html>`__ stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output.
- `Constraints <attacks/constraint.html>`__ determine if a potential perturbation is valid with respect to the original input.
- `Transformations <attacks/transformation.html>`__ take a text input and transform it by inserting and deleting characters, words, and/or phrases.
- `Search Methods <attacks/search_method.html>`__ explore the space of possible **transformations** within the defined **constraints** and attempt to find a successful perturbation which satisfies the **goal function**.

TextAttack provides a set of `Attack Recipes <attacks/attack_recipes.html>`__ that assemble attacks from the literature from these four components.

Data Augmentation
-------------
Data augmentation is easy and extremely common in computer vision but harder and less common in NLP. We provide a `Data Augmentation <augmentation/augmenter.html>`__ module using transformations and constraints.

Features
------------
TextAttack has some other features that make it a pleasure to use:

- `Built-in Datasets <datasets_models/datasets.html>`__ for running attacks without supplying your own data
- `Pre-trained Models <datasets_models/models.html>`__ for testing attacks and evaluating constraints
- `Built-in Tokenizers <datasets_models/tokenizers.html>`__ so you don't have to worry about tokenizing the inputs
- `Visualization options <misc/loggers.html>`__ like Weights & Biases and Visdom

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Quickstart

   quickstart/installation
   quickstart/overview
   Example 1: Transformations <examples/1_Introduction_and_Transformations.ipynb>
   Example 2: Constraints <examples/2_Constraints.ipynb>

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: NLP Attacks

   attacks/attack
   attacks/attack_result
   attacks/goal_function
   attacks/goal_function_result
   attacks/constraint
   attacks/transformation
   attacks/search_method
   attacks/attack_recipes
   
.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Data Augmentation

   augmentation/augmenter

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Models, Datasets and Tokenizers

   datasets_models/models
   datasets_models/datasets
   datasets_models/tokenizers

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Miscellaneous
   
   misc/loggers
   misc/validators
   misc/tokenized_text
