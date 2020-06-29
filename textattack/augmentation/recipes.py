import random

import textattack

from . import Augmenter

DEFAULT_CONSTRAINTS = [
    textattack.constraints.pre_transformation.RepeatModification(),
    textattack.constraints.pre_transformation.StopwordModification(),
]

class EasyDataAugmenter(Augmenter):
    def __init__(self, alpha, n_aug):
        from textattack.transformations import CompositeTransformation, \
                WordSwapWordNet, RandomSynonymInsertion, RandomSwap, \
                WordDeletion

        self.alpha = alpha
        self.n_aug = n_aug

        n_aug_each = int(n_aug/4) + 1
        transformation = CompositeTransformation([
                            WordSwapWordNet(),
                            RandomSynonymInsertion(),
                            RandomSwap(),
                            WordDeletion()
                        ])
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, transformations_per_example=n_aug)

    def augment(self, text):
        attacked_text = textattack.shared.AttackedText(text)
        num_words_to_swap = max(1, int(self.alpha*len(attacked_text.words)))
        self.num_words_to_swap = num_words_to_swap
        augmented_text = super().augment(text)
        return augmented_text



class WordNetAugmenter(Augmenter):
    """ Augments text by replacing with synonyms from the WordNet thesaurus. """

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapWordNet

        transformation = WordSwapWordNet()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)

class RandomDeletionAugmenter(Augmenter):
    def __init__(self, **kwargs):
        from textattack.transformations import WordDeletion
        transformation = WordDeletion()
        super().__init__(transformation, constraints=DEFAULT_CONSTRAINTS, **kwargs)


class EmbeddingAugmenter(Augmenter):
    """ Augments text by transforming words with their embeddings. """

    def __init__(self, **kwargs):
        from textattack.transformations import WordSwapEmbedding

        transformation = WordSwapEmbedding(
            max_candidates=50, embedding_type="paragramcf"
        )
        from textattack.constraints.semantics import WordEmbeddingDistance

        constraints = DEFAULT_CONSTRAINTS + [WordEmbeddingDistance(min_cos_sim=0.8)]
        super().__init__(transformation, constraints=constraints, **kwargs)


class CharSwapAugmenter(Augmenter):
    """ Augments words by swapping characters out for other characters. """

    def __init__(self, **kwargs):
        from textattack.transformations import CompositeTransformation
        from textattack.transformations import (
            WordSwapNeighboringCharacterSwap,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
            WordSwapRandomCharacterSubstitution,
            WordSwapNeighboringCharacterSwap,
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
