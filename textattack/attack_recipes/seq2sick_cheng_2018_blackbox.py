"""

Seq2Sick
================================================
(Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with Adversarial Examples)
"""

from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import NonOverlappingOutput
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapEmbedding

from .attack_recipe import AttackRecipe


class Seq2SickCheng2018BlackBox(AttackRecipe):
    """Cheng, Minhao, et al.

    Seq2Sick: Evaluating the Robustness of Sequence-to-Sequence Models with
    Adversarial Examples

    https://arxiv.org/abs/1803.01128

    This is a greedy re-implementation of the seq2sick attack method. It does
    not use gradient descent.

    Works against any encoder-decoder generation model loaded via
    :class:`~textattack.models.wrappers.HuggingFaceModelWrapper` (e.g. a raw
    ``transformers.BartForConditionalGeneration`` or
    ``transformers.T5ForConditionalGeneration``), not just TextAttack's own
    ``T5ForTextToText`` helper. See
    https://github.com/QData/TextAttack/issues/771 and
    https://github.com/QData/TextAttack/issues/772.

    Example, attacking a BART summarization model directly from
    ``transformers``::

        import transformers
        from textattack.attack_recipes import Seq2SickCheng2018BlackBox
        from textattack.models.wrappers import HuggingFaceModelWrapper

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

        attack = Seq2SickCheng2018BlackBox.build(model_wrapper)
        result = attack.attack(input_text, original_summary)
        print(result.__str__(color_method="ansi"))

    This also works against translation models (e.g. attacking BART/MarianMT
    on an en-de translation task): pass the model's own unperturbed
    translation as the second argument to ``.attack()``, since
    ``NonOverlappingOutput`` (unlike ``MinimizeBleu``, used by
    :class:`~textattack.attack_recipes.MorpheusTan2020`) compares against
    that rather than a ground-truth reference translation. See
    :class:`~textattack.attack_recipes.MorpheusTan2020` for a worked
    translation example and https://github.com/QData/TextAttack/issues/725.
    """

    @staticmethod
    def build(model_wrapper, goal_function="non_overlapping"):
        #
        # Goal is non-overlapping output.
        #
        goal_function = NonOverlappingOutput(model_wrapper)
        transformation = WordSwapEmbedding(max_candidates=50)
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(30))
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)
