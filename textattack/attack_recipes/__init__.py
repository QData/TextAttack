""".. _attack_recipes:

Attack Recipes:
======================

We provide a number of pre-built attack recipes, which correspond to attacks from the literature. To run an attack recipe from the command line, run::

    textattack attack --recipe [recipe_name]

To initialize an attack in Python script, use::

    <recipe name>.build(model_wrapper)

For example, ``attack = InputReductionFeng2018.build(model)`` creates `attack`, an object of type ``Attack`` with the goal function, transformation, constraints, and search method specified in that paper. This object can then be used just like any other attack; for example, by calling ``attack.attack_dataset``.

TextAttack supports the following attack recipes (each recipe's documentation contains a link to the corresponding paper):

.. contents:: :local:
"""

from .attack_recipe import AttackRecipe

from .bae_garg_2019 import BAEGarg2019
from .bert_attack_li_2020 import BERTAttackLi2020
from .genetic_algorithm_alzantot_2018 import GeneticAlgorithmAlzantot2018
from .faster_genetic_algorithm_jia_2019 import FasterGeneticAlgorithmJia2019
from .deepwordbug_gao_2018 import DeepWordBugGao2018
from .hotflip_ebrahimi_2017 import HotFlipEbrahimi2017
from .input_reduction_feng_2018 import InputReductionFeng2018
from .kuleshov_2017 import Kuleshov2017
from .morpheus_tan_2020 import MorpheusTan2020
from .seq2sick_cheng_2018_blackbox import Seq2SickCheng2018BlackBox
from .textbugger_li_2018 import TextBuggerLi2018
from .textfooler_jin_2019 import TextFoolerJin2019
from .pwws_ren_2019 import PWWSRen2019
from .iga_wang_2019 import IGAWang2019
from .pruthi_2019 import Pruthi2019
from .pso_zang_2020 import PSOZang2020
from .checklist_ribeiro_2020 import CheckList2020
