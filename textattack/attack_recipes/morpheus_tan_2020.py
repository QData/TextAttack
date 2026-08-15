"""
MORPHEUS2020
===============
(It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations)


"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import MinimizeBleu
from textattack.search_methods import GreedySearch
from textattack.transformations import WordSwapInflections

from .attack_recipe import AttackRecipe


class MorpheusTan2020(AttackRecipe):
    """Samson Tan, Shafiq Joty, Min-Yen Kan, Richard Socher.

    It’s Morphin’ Time! Combating Linguistic Discrimination with
    Inflectional Perturbations

    https://www.aclweb.org/anthology/2020.acl-main.263/

    Like :class:`~textattack.attack_recipes.Seq2SickCheng2018BlackBox`, this
    works against any encoder-decoder generation model loaded via
    :class:`~textattack.models.wrappers.HuggingFaceModelWrapper`, so it can
    attack machine-translation checkpoints (e.g. MarianMT, mBART, a BART
    checkpoint fine-tuned for translation) the same way it attacks the
    text-classification models used elsewhere in this codebase's examples.
    ``MinimizeBleu``'s ``ground_truth_output`` is the *reference*
    translation (not the model's own unperturbed output, unlike
    ``NonOverlappingOutput``/seq2sick), so pass the target-language
    reference sentence as the second argument to ``.attack()``. See
    https://github.com/QData/TextAttack/issues/725.

    Example, attacking an English-to-German translation model (any
    encoder-decoder checkpoint works the same way, including BART/MarianMT/
    mBART translation checkpoints -- swap in the ``model``/``tokenizer``
    below for one of those)::

        import transformers
        from textattack.attack_recipes import MorpheusTan2020
        from textattack.models.wrappers import HuggingFaceModelWrapper

        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small")
        # `max_length` avoids `transformers`' 20-token generation default,
        # which truncates sentence-length translations.
        model_wrapper = HuggingFaceModelWrapper(model, tokenizer, max_length=200)

        attack = MorpheusTan2020.build(model_wrapper)
        # t5-small needs the task prefix; other translation checkpoints
        # (BART/MarianMT/mBART) typically don't.
        input_text = "translate English to German: The quick brown fox jumps over the lazy dog."
        reference_translation = "Der schnelle braune Fuchs springt über den faulen Hund."
        result = attack.attack(input_text, reference_translation)
        print(result.__str__(color_method="ansi"))
    """

    @staticmethod
    def build(model_wrapper):
        #
        # Goal is to minimize BLEU score between the model output given for the
        # perturbed input sequence and the reference translation
        #
        goal_function = MinimizeBleu(model_wrapper)

        # Swap words with their inflections
        transformation = WordSwapInflections()

        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]

        #
        # Greedily swap words (see pseudocode, Algorithm 1 of the paper).
        #
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)
