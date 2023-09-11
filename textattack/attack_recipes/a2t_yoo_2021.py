"""
A2T (A2T: Attack for Adversarial Training Recipe)
==================================================

"""

from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    MaxModificationRate,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import BERT
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding, WordSwapMaskedLM

from .attack_recipe import AttackRecipe


class A2TYoo2021(AttackRecipe):
    """Towards Improving Adversarial Training of NLP Models.

    (Yoo et al., 2021)

    https://arxiv.org/abs/2109.00544
    """

    @staticmethod
    def build(model_wrapper, mlm=False):
        """Build attack recipe.

        Args:
            model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
                Model wrapper containing both the model and the tokenizer.
            mlm (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`True`, load `A2T-MLM` attack. Otherwise, load regular `A2T` attack.

        Returns:
            :class:`~textattack.Attack`: A2T attack.
        """
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
        constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))
        sent_encoder = BERT(
            model_name="stsb-distilbert-base", threshold=0.9, metric="cosine"
        )
        constraints.append(sent_encoder)

        if mlm:
            transformation = transformation = WordSwapMaskedLM(
                method="bae", max_candidates=20, min_confidence=0.0, batch_size=16
            )
        else:
            transformation = WordSwapEmbedding(max_candidates=20)
            constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))

        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper, model_batch_size=32)
        #
        # Greedily swap words with "Word Importance Ranking".
        #

        max_len = getattr(model_wrapper, "max_length", None) or min(
            1024,
            model_wrapper.tokenizer.model_max_length,
            model_wrapper.model.config.max_position_embeddings - 2,
        )
        search_method = GreedyWordSwapWIR(
            wir_method="gradient", truncate_words_to=max_len
        )

        return Attack(goal_function, constraints, transformation, search_method)
