Attack Recipes
===============

We provide a number of pre-built attack recipes, which correspond to attacks from the literature. To run an attack recipe, run::

    textattack attack --recipe [recipe_name]


Attacks on classification models
#################################

    
Alzantot Genetic Algorithm (Generating Natural Language Adversarial Examples)
###################################################################################

.. warning::
    This attack uses a very slow language model. Consider using the ``fast-alzantot``
    recipe instead.


.. automodule:: textattack.attack_recipes.genetic_algorithm_alzantot_2018
   :members:
       
Faster Alzantot Genetic Algorithm (Certified Robustness to Adversarial Word Substitutions)
##############################################################################################

.. automodule:: textattack.attack_recipes.faster_genetic_algorithm_jia_2019
   :members:
   
BAE (BAE: BERT-Based Adversarial Examples)
#############################################

.. automodule:: textattack.attack_recipes.bae_garg_2019
   :members:

BERT-Attack: (BERT-Attack: Adversarial Attack Against BERT Using BERT)
#########################################################################

.. automodule:: textattack.attack_recipes.bert_attack_li_2020
   :members:

DeepWordBug (Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)
######################################################################################################

.. automodule:: textattack.attack_recipes.deepwordbug_gao_2018
   :members:

HotFlip (HotFlip: White-Box Adversarial Examples for Text Classification)
##############################################################################

.. automodule:: textattack.attack_recipes.hotflip_ebrahimi_2017
   :members:

Improved Genetic Algorithm (Natural Language Adversarial Attacks and Defenses in Word Level)
#################################################################################################

.. automodule:: textattack.attack_recipes.iga_wang_2019
   :members:


Input Reduction (Pathologies of Neural Models Make Interpretations Difficult)
####################################################################################

.. automodule:: textattack.attack_recipes.input_reduction_feng_2018
   :members:

Kuleshov (Adversarial Examples for Natural Language Classification Problems)
##############################################################################

.. automodule:: textattack.attack_recipes.kuleshov_2017
   :members:
   
Particle Swarm Optimization (Word-level Textual Adversarial Attacking as Combinatorial Optimization)
#####################################################################################################

.. automodule:: textattack.attack_recipes.pso_zang_2020
   :members:

PWWS (Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)
###################################################################################################

.. automodule:: textattack.attack_recipes.pwws_ren_2019
    :members:

TextFooler (Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment)
########################################################################################################################

.. automodule:: textattack.attack_recipes.textfooler_jin_2019
   :members:


TextBugger (TextBugger: Generating Adversarial Text Against Real-world Applications)
########################################################################################

.. automodule:: textattack.attack_recipes.textbugger_li_2018
   :members:

Attacks on sequence-to-sequence models
##########################################

MORPHEUS (It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)
#####################################################################################################

.. automodule:: textattack.attack_recipes.morpheus_tan_2020
   :members:

Seq2Sick (Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)
#########################################################################################################

.. automodule:: textattack.attack_recipes.seq2sick_cheng_2018_blackbox
   :members:



