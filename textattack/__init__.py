"""Welcome to the API references for TextAttack!

What is TextAttack?

`TextAttack <https://github.com/QData/TextAttack>`__
is a Python framework for adversarial attacks, adversarial training, and data augmentation in NLP.

TextAttack makes experimenting with the robustness of NLP models seamless, fast, and easy. It's also useful for NLP model training, adversarial training, and data augmentation.

TextAttack provides components for common NLP tasks like sentence encoding, grammar-checking, and word replacement that can be used on their own.
"""
from .attack_args import AttackArgs, CommandLineAttackArgs
from .augment_args import AugmenterArgs
from .dataset_args import DatasetArgs
from .model_args import ModelArgs
from .training_args import TrainingArgs, CommandLineTrainingArgs
from .attack import Attack
from .attacker import Attacker
from .trainer import Trainer
from .metrics import Metric

from . import (
    attack_recipes,
    attack_results,
    augmentation,
    commands,
    constraints,
    datasets,
    goal_function_results,
    goal_functions,
    loggers,
    metrics,
    models,
    search_methods,
    shared,
    transformations,
)


name = "textattack"
