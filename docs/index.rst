TextAttack
==============

Welcome to the documentation for TextAttack!

What is TextAttack?
----------------------
`TextAttack <https://github.com/QData/TextAttack>`__ is a Python framework for adversarial attacks, adversarial training, and data augmentation in NLP. 

TextAttack makes experimenting with the robustness of NLP models seamless, fast, and easy. It's also useful for NLP model training, adversarial training, and data augmentation. 

TextAttack provides components for common NLP tasks like sentence encoding, grammar-checking, and word replacement that can be used on their own.

Where should I start?
----------------------

This is a great question, and one we get a lot. First of all, almost everything in TextAttack can be done in two ways: via the command-line or via the Python API. If you're looking to integrate TextAttack into an existing project, the Python API is likely for you. If you'd prefer to use built-in functionality end-to-end (training a model, running an adversarial attack, augmenting a CSV) then you can just use the command-line API.

For future developers, visit the `Installation <https://github.com/QData/TextAttack/blob/master/docs/quickstart/installation.rst>`__ page for more details about installing TextAttack onto your own computer. To start making contributions, read the detailed instructions `here <https://github.com/QData/TextAttack/blob/master/CONTRIBUTING.md>`__.

TextAttack does three things very well:

1. Adversarial attacks (Python: ``textattack.shared.Attack``, Bash: ``textattack attack``)
2. Data augmentation (Python: ``textattack.augmentation.Augmenter``, Bash: ``textattack augment``)
3. Model training (Python: ``textattack.commands.train.*``, Bash: ``textattack train``)

Adversarial training can be achieved as a combination of [1] and/or [2] with [3] (via ``textattack train --attack``). To see all this in action, see :ref:`the TextAttack End-to-End tutorial </examples/0_End_to_End.ipynb>`.

All of the other components: datasets, models & model wrappers, loggers, transformations, constraints, search methods, goal functions, etc., are developed to support one or more of these three functions. Feel free though to install textattack to include just one of those components! (For example, TextAttack provides a really easy Python interface for accessing and using word embeddings that will automatically download and save them on the first use.)


NLP Attacks
-----------

TextAttack provides a framework for constructing and thinking about attacks via perturbation in NLP. TextAttack builds attacks from four components:

- `Goal Functions <attacks/goal_function.html>`__ stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output.
- `Constraints <attacks/constraint.html>`__ determine if a potential perturbation is valid with respect to the original input.
- `Transformations <attacks/transformation.html>`__ take a text input and transform it by inserting and deleting characters, words, and/or phrases.
- `Search Methods <attacks/search_method.html>`__ explore the space of possible **transformations** within the defined **constraints** and attempt to find a successful perturbation which satisfies the **goal function**.

TextAttack provides a set of `Attack Recipes <attacks/attack_recipes.html>`__ that assemble attacks from the literature from these four components. Take a look at these recipes (or our `paper on ArXiv <https://arxiv.org/abs/2005.05909>`__) to get a feel for how the four components work together to create an adversarial attack.

Data Augmentation
--------------------
Data augmentation is easy and extremely common in computer vision but harder and less common in NLP. We provide a `Data Augmentation <augmentation/augmenter.html>`__ module using transformations and constraints.

Features
------------
TextAttack has some other features that make it a pleasure to use:

- `Pre-trained Models <datasets_models/models.html>`__ for testing attacks and evaluating constraints
- `Visualization options <misc/loggers.html>`__ like Weights & Biases and Visdom
- `AttackedText <misc/attacked_text.rst>`__, a utility class for strings that includes tools for tokenizing and editing text

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Installation <quickstart/installation>
   Command-Line Usage <quickstart/command_line_usage>
   What is an adversarial attack in NLP? <quickstart/what_is_an_adversarial_attack.md>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Tutorials
   
   Tutorial 0: TextAttack End-To-End (Train, Eval, Attack) <examples/0_End_to_End.ipynb>
   Tutorial 1: Transformations <examples/1_Introduction_and_Transformations.ipynb>
   Tutorial 2: Constraints <examples/2_Constraints.ipynb>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Models

   datasets_models/models
   Example: Attacking TensorFlow models <datasets_models/Example_0_tensorflow>
   Example: Attacking scikit-learn models <datasets_models/Example_1_sklearn.ipynb>
   Example: Attacking AllenNLP models <datasets_models/Example_2_allennlp.ipynb>
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Multilingual Examples

   CamemBERT & French WordNet <multilingual/Example_1_CamemBERT.ipynb>

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Attacks

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
   :caption: Miscellaneous
   
   misc/attacked_text
   misc/checkpoints
   misc/loggers
   misc/validators
