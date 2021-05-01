"""
Combating Adversarial Misspellings with Robust Word Recognition
=================================================================

"""
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.shared.attack import Attack
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from .attack_recipe import AttackRecipe


class Pruthi2019(AttackRecipe):
    """An implementation of the attack used in "Combating Adversarial
    Misspellings with Robust Word Recognition", Pruthi et al., 2019.

    This attack focuses on a small number of character-level changes that simulate common typos. It combines:
        - Swapping neighboring characters
        - Deleting characters
        - Inserting characters
        - Swapping characters for adjacent keys on a QWERTY keyboard.

    https://arxiv.org/abs/1905.11268

    :param model: Model to attack.
    :param max_num_word_swaps: Maximum number of modifications to allow.
    """

    @staticmethod
    def build(model, max_num_word_swaps=1):
        # a combination of 4 different character-based transforms
        # ignore the first and last letter of each word, as in the paper
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )
        # only edit words of length >= 4, edit max_num_word_swaps words.
        # note that we also are not editing the same word twice, so
        # max_num_word_swaps is really the max number of character
        # changes that can be made. The paper looks at 1 and 2 char attacks.
        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=max_num_word_swaps),
            RepeatModification(),
        ]
        # untargeted attack
        goal_function = UntargetedClassification(model)
        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)
