TextAttack Documentation
=======================================


.. toctree::
   :maxdepth: 6
   :caption: About

   1start/basic-Intro.rst
   1start/what_is_an_adversarial_attack.md
   1start/references.md
   1start/attacks4Components.md
   1start/benchmark-search.md
   3recipes/models.md
   1start/FAQ.md

.. toctree::
   :maxdepth: 6
   :caption: Get Started

   Installation <1start/installation>
   Command-Line Usage <1start/command_line_usage.md>

 
.. toctree::
   :maxdepth: 6
   :caption: Notebook Tutorials
   
   Tutorial 0: TextAttack End-To-End (Train, Eval, Attack) <2notebook/0_End_to_End.ipynb>
   Tutorial 1: Transformations <2notebook/1_Introduction_and_Transformations.ipynb>
   Tutorial 2: Constraints <2notebook/2_Constraints.ipynb>
   Tutorial 3: Augmentation <2notebook/3_Augmentations.ipynb>
   Tutorial 4: Attacking TensorFlow models <2notebook/Example_0_tensorflow.ipynb>
   Tutorial 5: Attacking scikit-learn models <2notebook/Example_1_sklearn.ipynb>
   Tutorial 6: Attacking AllenNLP models <2notebook/Example_2_allennlp.ipynb>
   Tutorial 7: Attacking multilingual models <2notebook/Example_4_CamemBERT.ipynb>
   Tutorial 8: Explaining Attacking BERT model using Captum <2notebook/Example_5_Explain_BERT.ipynb>


.. toctree::
   :maxdepth: 6
   :glob:
   :caption: Developer Guide
   
   1start/support.md
   1start/api-design-tips.md
   3recipes/attack_recipes
   3recipes/augmenter_recipes
   apidoc/textattack


