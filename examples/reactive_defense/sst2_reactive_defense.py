# import os

from textattack import Attacker
from textattack.attack_recipes import (
    # PWWSRen2019,
    # BAEGarg2019,
    # TextFoolerJin2019,
    # BERTAttackLi2020,
    # GeneticAlgorithmAlzantot2018,
    # CLARE2020,
    # FasterGeneticAlgorithmJia2019,
    DeepWordBugGao2018,
    # PSOZang2020,
)
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import TADModelWrapper
from textattack.reactive_defense.tad_reactive_defender import TADReactiveDefender

dataset = HuggingFaceDataset("glue", subset="sst2", split="validation")

# init reactive_defender to post fix attacked result
reactive_defender = TADReactiveDefender("taddeberta-sst2")

# use the based tad_classifier (without defense) to test
target_model = reactive_defender.tad_classifier

model_wrapper = TADModelWrapper(target_model)

# recipe = PWWSRen2019.build(model_wrapper)
# +-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 31     |
# | Number of failed attacks:     | 801    |
# | Number of skipped attacks:    | 40     |
# | Original accuracy:            | 95.41% |
# | Accuracy under attack:        | 91.86% |
# | Attack success rate:          | 3.73%  |
# | Average perturbed word %:     | 7.06%  |
# | Average num. words per input: | 17.4   |
# | Avg num queries:              | 144.1  |
# +-------------------------------+--------+

# recipe = BAEGarg2019.build(model_wrapper)
# +-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 18     |
# | Number of failed attacks:     | 79     |
# | Number of skipped attacks:    | 3      |
# | Original accuracy:            | 97.0%  |
# | Accuracy under attack:        | 79.0%  |
# | Attack success rate:          | 18.56% |
# | Average perturbed word %:     | 12.8%  |
# | Average num. words per input: | 16.92  |
# | Avg num queries:              | 55.45  |
# +-------------------------------+--------+

# recipe = TextFoolerJin2019.build(model_wrapper)
# #+-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 6      |
# | Number of failed attacks:     | 91     |
# | Number of skipped attacks:    | 3      |
# | Original accuracy:            | 97.0%  |
# | Accuracy under attack:        | 91.0%  |
# | Attack success rate:          | 6.19%  |
# | Average perturbed word %:     | 16.28% |
# | Average num. words per input: | 16.92  |
# | Avg num queries:              | 124.56 |
# +-------------------------------+--------+

# recipe = GeneticAlgorithmAlzantot2018.build(model_wrapper)
#
# recipe = BERTAttackLi2020.build(model_wrapper)
#
# recipe = FasterGeneticAlgorithmJia2019.build(model_wrapper)
#
recipe = DeepWordBugGao2018.build(model_wrapper)
# +-------------------------------+--------+
# | Attack Results                |        |
# +-------------------------------+--------+
# | Number of successful attacks: | 14     |
# | Number of failed attacks:     | 83     |
# | Number of skipped attacks:    | 3      |
# | Original accuracy:            | 97.0%  |
# | Accuracy under attack:        | 83.0%  |
# | Attack success rate:          | 14.43% |
# | Average perturbed word %:     | 24.32% |
# | Average num. words per input: | 16.92  |
# | Avg num queries:              | 33.13  |
# +-------------------------------+--------+

# recipe = CLARE2020.build(model_wrapper)
# +-------------------------------+---------+
# | Attack Results                |         |
# +-------------------------------+---------+
# | Number of successful attacks: | 50      |
# | Number of failed attacks:     | 47      |
# | Number of skipped attacks:    | 3       |
# | Original accuracy:            | 97.0%   |
# | Accuracy under attack:        | 47.0%   |
# | Attack success rate:          | 51.55%  |
# | Average perturbed word %:     | 30.37%  |
# | Average num. words per input: | 16.92   |
# | Avg num queries:              | 1771.19 |
# +-------------------------------+---------+

# recipe = PSOZang2020.build(model_wrapper)
# +-------------------------------+---------+
# | Attack Results                |         |
# +-------------------------------+---------+
# | Number of successful attacks: | 10      |
# | Number of failed attacks:     | 87      |
# | Number of skipped attacks:    | 3       |
# | Original accuracy:            | 97.0%   |
# | Accuracy under attack:        | 87.0%   |
# | Attack success rate:          | 10.31%  |
# | Average perturbed word %:     | 18.98%  |
# | Average num. words per input: | 16.92   |
# | Avg num queries:              | 6497.32 |
# +-------------------------------+---------+


attacker = Attacker(recipe, dataset)

# install pyabsa for this example
attacker.attack_args.num_examples = 100
# results = attacker.attack_dataset()
results = attacker.attack_dataset(reactive_defender=reactive_defender)

# Online Reactive Adversarial Defense:
# https://huggingface.co/spaces/yangheng/TAD

# Ref repo:

# https://github.com/yangheng95/PyABSA
# https://github.com/yangheng95/TextAttack
