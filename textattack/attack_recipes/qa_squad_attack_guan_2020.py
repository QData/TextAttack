"""
QASquadAttack2020
===============
(Question Answering on Adversarial SQuAD 2.0)

"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import HalfOverlappingOutput
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwapInflections
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class QASquadAttackGuan2020(AttackRecipe):

    @staticmethod
    def build(model_wrapper):
        
        goal_function = HalfOverlappingOutput(model_wrapper)
        transformation = WordSwapEmbedding(max_candidates=50)
        use_constraint = UniversalSentenceEncoder(threshold=0.840845057, metric="angular",
            compare_against_original=False, window_size=15, skip_text_shorter_than_window=True)
        constraints = [RepeatModification(), StopwordModification(), PartOfSpeech(), use_constraint]
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)