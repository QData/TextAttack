TextAttack Documentation
=======================================


.. toctree::
   :maxdepth: 6
   :caption: Get Started

   Basic-Introduction <0_get_started/basic-Intro.rst>
   Installation <0_get_started/installation.md>
   Command-Line Usage <0_get_started/command_line_usage.md>
   Quick API Usage  <0_get_started/quick_api_tour.rst>
   FAQ <1start/FAQ.md>

.. toctree::
   :maxdepth: 6
   :caption: Recipes
   
   3recipes/attack_recipes_cmd.md
   3recipes/attack_recipes.rst
   3recipes/augmenter_recipes_cmd.md
   3recipes/augmenter_recipes.rst
   3recipes/models.md

.. toctree::
   :maxdepth: 6
   :caption: Using TextAttack

   1start/what_is_an_adversarial_attack.md
   1start/references.md
   1start/attacks4Components.md
   1start/benchmark-search.md
   1start/quality-SOTA-recipes.md
   1start/A2TforVanillaAT.md
   1start/api-design-tips.md
   1start/multilingual-visualization.md
   1start/support.md
 
.. toctree::
   :maxdepth: 6
   :caption: Notebook Tutorials
   
   Tutorial 0: TextAttack End-To-End (Train, Eval, Attack) <2notebook/0_End_to_End.ipynb>
   Tutorial 1: Transformations <2notebook/1_Introduction_and_Transformations.ipynb>
   Tutorial 2: Constraints <2notebook/2_Constraints.ipynb>
   Tutorial 3: Augmentation <2notebook/3_Augmentations.ipynb>
   Tutorial 4: Custom Word Embeddings <2notebook/4_Custom_Datasets_Word_Embedding.ipynb>
   Tutorial 5: Attacking TensorFlow models <2notebook/Example_0_tensorflow.ipynb>
   Tutorial 6: Attacking scikit-learn models <2notebook/Example_1_sklearn.ipynb>
   Tutorial 7: Attacking AllenNLP models <2notebook/Example_2_allennlp.ipynb>
   Tutorial 8: Attacking Keras models <2notebook/Example_3_Keras.ipynb>
   Tutorial 9: Attacking multilingual models <2notebook/Example_4_CamemBERT.ipynb>
   Tutorial10: Explaining Attacking BERT model using Captum <2notebook/Example_5_Explain_BERT.ipynb>

.. toctree::
   :maxdepth: 6
   :caption: API User Guide

   Attack <api/attack.rst>
   Attacker <api/attacker.rst>
   AttackResult <api/attack_results.rst>
   Trainer <api/trainer.rst>
   Datasets <api/datasets.rst>
   GoalFunction <api/goal_functions.rst>
   Constraints <api/constraints.rst>
   Transformations <api/transformations.rst>
   SearchMethod <api/search_methods.rst>
   

.. toctree::
   :maxdepth: 6
   :glob:
   :caption: Full Reference

   apidoc/textattack
