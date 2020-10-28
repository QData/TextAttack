"""Welcome to the API references for TextAttack!

What is TextAttack?

`TextAttack <https://github.com/QData/TextAttack>`__ is a Python framework for adversarial attacks, adversarial training, and data augmentation in NLP.

TextAttack makes experimenting with the robustness of NLP models seamless, fast, and easy. It's also useful for NLP model training, adversarial training, and data augmentation.

TextAttack provides components for common NLP tasks like sentence encoding, grammar-checking, and word replacement that can be used on their own.
"""

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
    models,
    search_methods,
    shared,
    transformations,
)

name = "textattack"
