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


def Pruthi2019(model, max_num_word_swaps=1):
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
            WordSwapQWERTY(random_one=False, skip_first_char=True, skip_last_char=True),
        ]
    )
    constraints = [
        MinWordLength(min_length=4),
        StopwordModification(),
        MaxWordsPerturbed(max_num_words=max_num_word_swaps),
        RepeatModification(),
    ]
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)
