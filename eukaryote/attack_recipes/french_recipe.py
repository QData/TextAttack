from eukaryote import Attack
from eukaryote.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from eukaryote.goal_functions import UntargetedClassification
from eukaryote.search_methods import GreedyWordSwapWIR
from eukaryote.transformations import (
    CompositeTransformation,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapWordNet,
)

from .attack_recipe import AttackRecipe


class FrenchRecipe(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        transformation = CompositeTransformation(
            [
                WordSwapWordNet(language="fra"),
                WordSwapChangeLocation(language="fra"),
                WordSwapChangeName(language="fra"),
            ]
        )
        constraints = [RepeatModification(), StopwordModification("french")]
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR()
        return Attack(goal_function, constraints, transformation, search_method)
