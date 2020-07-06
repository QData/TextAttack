

from textattack.transformations import (
        WordSwapNeighboringCharacterSwap, 
        WordSwapRandomCharacterDeletion, 
        WordSwapRandomCharacterInsertion, 
        WordSwapQWERTY, 
        CompositeTransformation,
)
from textattack.constraints.pre_transformation import StopwordModification, MinWordLength, RepeatModification
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.search_methods import GreedySearch
from textattack.goal_functions import UntargetedClassification
from textattack.shared.attack import Attack


def Pruthi2019(model, swaps=1):
    transformation = CompositeTransformation([
                        WordSwapNeighboringCharacterSwap(random_one=False, skip_first_char=True, skip_last_char=True),
                        WordSwapRandomCharacterDeletion(random_one=False, skip_first_char=True, skip_last_char=True),
                        WordSwapRandomCharacterInsertion(random_one=False, skip_first_char=True, skip_last_char=True),
                        WordSwapQWERTY(random_one=False, skip_first_char=True, skip_last_char=True),
                        ])
    constraints = [MinWordLength(min_length=4), StopwordModification(), MaxWordsPerturbed(swaps), RepeatModification()]
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)



