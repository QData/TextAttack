"""
Augmenter Recipes:
===================

Transformations and constraints can be used for simple NLP data augmentations. Here is a list of recipes for NLP data augmentations

"""
import random

from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)

from . import Augmenter

DEFAULT_CONSTRAINTS = [RepeatModification(), StopwordModification()]


class EasyDataAugmenter(Augmenter):
    """An implementation of Easy Data Augmentation, which combines:

    - WordNet synonym replacement
        - Randomly replace words with their synonyms.
    - Word deletion
        - Randomly remove words from the sentence.
    - Word order swaps
        - Randomly swap the position of words in the sentence.
    - Random synonym insertion
        - Insert a random synonym of a random word at a random location.

    in one augmentation method.

    "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks" (Wei and Zou, 2019)
    https://arxiv.org/abs/1901.11196
    """

    def __init__(self, pct_words_to_swap=0.1, transformations_per_example=4):
        assert (
            pct_words_to_swap >= 0.0 and pct_words_to_swap <= 1.0
        ), "pct_words_to_swap must be in [0., 1.]"
        assert (
            transformations_per_example > 0
        ), "transformations_per_example must be a positive integer"
        self.pct_words_to_swap = pct_words_to_swap
        self.transformations_per_example = transformations_per_example
        n_aug_each = max(transformations_per_example // 4, 1)

        self.synonym_replacement = WordNetAugmenter(
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=n_aug_each,
        )
        self.random_deletion = DeletionAugmenter(
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=n_aug_each,
        )
        self.random_swap = SwapAugmenter(
            pct_words_to_swap=pct_words_to_swap,
            transformations_per_example=n_aug_each,
        )
        self.random_insertion = SynonymInsertionAugmenter(
            pct_words_to_swap=pct_words_to_swap, transformations_per_example=n_aug_each
        )

    def augment(self, text):
        augmented_text = []
        augmented_text += self.synonym_replacement.augment(text)
        augmented_text += self.random_deletion.augment(text)
        augmented_text += self.random_swap.augment(text)
        augmented_text += self.random_insertion.augment(text)
        random.shuffle(augmented_text)
        return augmented_text[: self.transformations_per_example]

    def __repr__(self):
        return "EasyDataAugmenter"


class SwapAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import WordInnerSwapRandom

        transformation = WordInnerSwapRandom()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class SynonymInsertionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import WordInsertionRandomSynonym

        transformation = WordInsertionRandomSynonym()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class WordNetAugmenter(Augmenter):
    """Augments text by replacing with synonyms from the WordNet thesaurus."""

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapWordNet

        transformation = WordSwapWordNet()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class DeletionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import WordDeletion

        transformation = WordDeletion()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class EmbeddingAugmenter(Augmenter):
    """Augments text by transforming words with their embeddings."""

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapEmbedding

        transformation = WordSwapEmbedding(max_candidates=50)
        from textattack.constraints.semantics import WordEmbeddingDistance

        constraints = DEFAULT_CONSTRAINTS + [WordEmbeddingDistance(min_cos_sim=0.8)]
        super().__init__(transformation, constraints=constraints, **kwargs)


class CharSwapAugmenter(Augmenter):
    """Augments words by swapping characters out for other characters."""

    def __init__(self, **kwargs):
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapNeighboringCharacterSwap,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
            WordSwapRandomCharacterSubstitution,
        )

        transformation = CompositeTransformation(
            [
                # (1) Swap: Swap two adjacent letters in the word.
                WordSwapNeighboringCharacterSwap(),
                # (2) Substitution: Substitute a letter in the word with a random letter.
                WordSwapRandomCharacterSubstitution(),
                # (3) Deletion: Delete a random letter from the word.
                WordSwapRandomCharacterDeletion(),
                # (4) Insertion: Insert a random letter in the word.
                WordSwapRandomCharacterInsertion(),
            ]
        )
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class CheckListAugmenter(Augmenter):
    """Augments words by using the transformation methods provided by CheckList
    INV testing, which combines:

    - Name Replacement
    - Location Replacement
    - Number Alteration
    - Contraction/Extension

    "Beyond Accuracy: Behavioral Testing of NLP models with CheckList" (Ribeiro et al., 2020)
    https://arxiv.org/abs/2005.04118
    """

    def __init__(self, **kwargs):
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapChangeLocation,
            WordSwapChangeName,
            WordSwapChangeNumber,
            WordSwapContract,
            WordSwapExtend,
        )

        transformation = CompositeTransformation(
            [
                WordSwapChangeNumber(),
                WordSwapChangeLocation(),
                WordSwapChangeName(),
                WordSwapExtend(),
                WordSwapContract(),
            ]
        )

        constraints = [DEFAULT_CONSTRAINTS[0]]

        super().__init__(transformation, constraints=constraints, **kwargs)
