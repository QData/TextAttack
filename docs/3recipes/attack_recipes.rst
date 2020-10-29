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


Alzantot Genetic Algorithm (Generating Natural Language Adversarial Examples)
***********************************************************************************

.. warning::
    This attack uses a very slow language model. Consider using the ``fast-alzantot``
    recipe instead.


.. automodule:: textattack.attack_recipes.genetic_algorithm_alzantot_2018
   :members:
   :noindex:

Faster Alzantot Genetic Algorithm (Certified Robustness to Adversarial Word Substitutions)
**********************************************************************************************

.. automodule:: textattack.attack_recipes.faster_genetic_algorithm_jia_2019
   :members:
   :noindex:

BAE (BAE: BERT-Based Adversarial Examples)
*********************************************

.. automodule:: textattack.attack_recipes.bae_garg_2019
   :members:
   :noindex:

BERT-Attack: (BERT-Attack: Adversarial Attack Against BERT Using BERT)
*************************************************************************

.. automodule:: textattack.attack_recipes.bert_attack_li_2020
   :members:
   :noindex:


CheckList: (Beyond Accuracy: Behavioral Testing of NLP models with CheckList)
*******************************************************************************

.. automodule:: textattack.attack_recipes.checklist_ribeiro_2020
   :members:
   :noindex:

DeepWordBug (Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)
******************************************************************************************************

.. automodule:: textattack.attack_recipes.deepwordbug_gao_2018
   :members:
   :noindex:

HotFlip (HotFlip: White-Box Adversarial Examples for Text Classification)
******************************************************************************

.. automodule:: textattack.attack_recipes.hotflip_ebrahimi_2017
   :members:
   :noindex:

Improved Genetic Algorithm (Natural Language Adversarial Attacks and Defenses in Word Level)
*************************************************************************************************

.. automodule:: textattack.attack_recipes.iga_wang_2019
   :members:
   :noindex:

Input Reduction (Pathologies of Neural Models Make Interpretations Difficult)
************************************************************************************

.. automodule:: textattack.attack_recipes.input_reduction_feng_2018
   :members:
   :noindex:


Kuleshov (Adversarial Examples for Natural Language Classification Problems)
******************************************************************************

.. automodule:: textattack.attack_recipes.kuleshov_2017
   :members:
   :noindex:

Particle Swarm Optimization (Word-level Textual Adversarial Attacking as Combinatorial Optimization)
*****************************************************************************************************

.. automodule:: textattack.attack_recipes.pso_zang_2020
   :members:
   :noindex:

PWWS (Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)
***************************************************************************************************

.. automodule:: textattack.attack_recipes.pwws_ren_2019
   :members:
   :noindex:

TextFooler (Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment)
************************************************************************************************************************

.. automodule:: textattack.attack_recipes.textfooler_jin_2019
   :members:
   :noindex:


TextBugger (TextBugger: Generating Adversarial Text Against Real-world Applications)
****************************************************************************************

.. automodule:: textattack.attack_recipes.textbugger_li_2018
   :members:
   :noindex:

Attacks on sequence-to-sequence models
############################################

MORPHEUS (It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)
*****************************************************************************************************

.. automodule:: textattack.attack_recipes.morpheus_tan_2020
   :members:
   :noindex:

Seq2Sick (Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)
*********************************************************************************************************

.. automodule:: textattack.attack_recipes.seq2sick_cheng_2018_blackbox
   :members:
   :noindex:
