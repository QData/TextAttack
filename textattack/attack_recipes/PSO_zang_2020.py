from textattack.constraints.pre_transformation import (  # InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import PSOAlgorithm
from textattack.shared.attack import Attack
from textattack.transformations import WordSwapEmbedding, WordSwapHowNet


def PSOZang2020(model):
    """
        Zang, Y., Yang, C., Qi, F., Liu, Z., Zhang, M., Liu, Q., & Sun, M. (2019).
        
        Word-level Textual Adversarial Attacking as Combinatorial Optimization.
        
        https://www.aclweb.org/anthology/2020.acl-main.540.pdf
    """
    #
    # Swap words with their synonyms extracted based on the HowNet.
    #
    transformation = WordSwapHowNet()
    #
    # Don't modify the same word twice or stopwords
    #
    constraints = [RepeatModification(), StopwordModification()]
    #
    #
    # During entailment, we should only edit the hypothesis - keep the premise
    # the same.
    #
    # input_column_modification = InputColumnModification(
    #     ["premise", "hypothesis"], {"premise"}
    # )
    # constraints.append(input_column_modification)
    #
    # Goal is untargeted classification
    #
    goal_function = UntargetedClassification(model)
    #
    # Perform word substitution with a genetic algorithm.
    #
    search_method = PSOAlgorithm(pop_size=60, max_iters=20)

    return Attack(goal_function, constraints, transformation, search_method)
