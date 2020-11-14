Attack Recipes
===============

We provide a number of pre-built attack recipes, which correspond to attacks from the literature. To run an attack recipe from the command line, run::

    textattack attack --recipe [recipe_name]

To initialize an attack in Python script, use::

    <recipe name>.build(model_wrapper)

For example, ``attack = InputReductionFeng2018.build(model)`` creates `attack`, an object of type ``Attack`` with the goal function, transformation, constraints, and search method specified in that paper. This object can then be used just like any other attack; for example, by calling ``attack.attack_dataset``.

TextAttack supports the following attack recipes (each recipe's documentation contains a link to the corresponding paper):

.. contents:: :local:


Attacks on classification models
#################################


1. Alzantot Genetic Algorithm (Generating Natural Language Adversarial Examples)
2. Faster Alzantot Genetic Algorithm (Certified Robustness to Adversarial Word Substitutions)
3. BAE (BAE: BERT-Based Adversarial Examples)
4. BERT-Attack: (BERT-Attack: Adversarial Attack Against BERT Using BERT)
5. CheckList: (Beyond Accuracy: Behavioral Testing of NLP models with CheckList)
6. DeepWordBug (Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)
7. HotFlip (HotFlip: White-Box Adversarial Examples for Text Classification)
8. Improved Genetic Algorithm (Natural Language Adversarial Attacks and Defenses in Word Level)
9. Input Reduction (Pathologies of Neural Models Make Interpretations Difficult)
10. Kuleshov (Adversarial Examples for Natural Language Classification Problems)
11. Particle Swarm Optimization (Word-level Textual Adversarial Attacking as Combinatorial Optimization)
12. PWWS (Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)
13. TextFooler (Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment)
14. TextBugger (TextBugger: Generating Adversarial Text Against Real-world Applications)



.. automodule:: textattack.attack_recipes.genetic_algorithm_alzantot_2018
   :members:
   :noindex:


.. automodule:: textattack.attack_recipes.faster_genetic_algorithm_jia_2019
   :members:
   :noindex:


.. automodule:: textattack.attack_recipes.bae_garg_2019
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.bert_attack_li_2020
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.checklist_ribeiro_2020
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.deepwordbug_gao_2018
   :members:
   :noindex:


.. automodule:: textattack.attack_recipes.hotflip_ebrahimi_2017
   :members:
   :noindex:


.. automodule:: textattack.attack_recipes.iga_wang_2019
   :members:
   :noindex:


.. automodule:: textattack.attack_recipes.input_reduction_feng_2018
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.kuleshov_2017
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.pso_zang_2020
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.pwws_ren_2019
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.textfooler_jin_2019
   :members:
   :noindex:



.. automodule:: textattack.attack_recipes.textbugger_li_2018
   :members:
   :noindex:


Attacks on sequence-to-sequence models
############################################

15. MORPHEUS (It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)
16. Seq2Sick (Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)


.. automodule:: textattack.attack_recipes.morpheus_tan_2020
   :members:
   :noindex:


.. automodule:: textattack.attack_recipes.seq2sick_cheng_2018_blackbox
   :members:
   :noindex:
