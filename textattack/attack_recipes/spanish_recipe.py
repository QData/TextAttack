from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapWordNet,
)

from .attack_recipe import AttackRecipe


class SpanishRecipe(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        transformation = CompositeTransformation(
            [
                WordSwapWordNet(language="esp"),
                WordSwapChangeLocation(language="esp"),
                WordSwapChangeName(language="esp"),
            ]
        )
        constraints = [RepeatModification(), StopwordModification("spanish")]
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR()
        return Attack(goal_function, constraints, transformation, search_method)
