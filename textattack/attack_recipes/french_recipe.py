from textattack import Attack
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapWordNet, CompositeTransformation, WordSwapChangeLocation, WordSwapChangeName
from .attack_recipe import AttackRecipe

class French_Recipe(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapWordNet('fra')
        transformation.language= 'fra'
        transformation = CompositeTransformation([transformation, WordSwapChangeLocation(language="fra"),WordSwapChangeName(language="fra")])
        constraints = [RepeatModification(), StopwordModification("french")]
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR()
        return Attack(goal_function, constraints, transformation, search_method)