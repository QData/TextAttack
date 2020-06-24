from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.transformations import WordSwapWordNet, WordSwapNamedEntity, CompositeTransformation
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared.attack import Attack

def PWWSRen2019(model):
    transformation = CompositeTransformation([WordSwapWordNet(), WordSwapNamedEntity()])
    constraints = [RepeatModification(), StopwordModification()]
    goal_function = UntargetedClassification(model)
    search_method = GreedyWordSwapWIR('pwws', ascending=False)
    return Attack(goal_function, constraints, transformation, search_method)

