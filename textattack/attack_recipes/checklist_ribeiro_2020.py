from textattack.constraints.pre_transformation import RepeatModification
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.shared.attack import Attack
from textattack.transformations import (
    CompositeTransformation,
    WordSwapContract,
    WordSwapExtend,
)


def Checklist2020(model, max_num_word_swaps=1):
    """An implementation of the attack used in "Beyond Accuracy: Behavioral
     Testing of NLP models with CheckList", Ribeiro et al., 2020."

    This attack focuses on a number of attacks used in the Invariance Testing
    Method:
        - Contracting
        - Extending
        - Changing Names
        - Possibly Negation...

    https://arxiv.org/abs/2005.04118

    :param model: Model to attack.
    :param max_num_word_swaps: Maximum number of modifications to allow.
    """

    transformation = CompositeTransformation([WordSwapExtend(), WordSwapContract()])

    # we need this constraint, or extend and contract start to modify each other's changes and form infinite loop
    constraints = [RepeatModification()]
    # untargeted attack
    goal_function = UntargetedClassification(model)
    search_method = GreedySearch()
    return Attack(goal_function, constraints, transformation, search_method)
