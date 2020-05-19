TextAttack
======================================

`TextAttack <https://github.com/QData/TextAttack>`__ is a Python library for adversarial attacks and data augmentation in NLP.

Features 
-----------

TextAttack isn't just a Python library; it's a framework for constructing and thinking about adversarial attacks in NLP. TextAttack builds attacks from four components:

- **Goal Functions** stipulate the goal of the attack, like to change the prediction score of a classification model, or to change all of the words in a translation output
- **Search Methods** explore the space of transformations and attempt to find a successful perturbation
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

   attacks/search_methods
   attacks/transformations
   attacks/constraints
   attacks/goal_functions
   attacks/goal_function_results
   attacks/attacks
   attacks/attack_recipess
   
.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Data Augmentation

   augmentation/augmentation

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Datasets, Models, and Tokenizers

   datasets_models/models
   datasets_models/datasets
   datasets_models/tokenizers

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Miscellaneous:
   
   misc/loggers
   misc/validators
   misc/tokenized_text