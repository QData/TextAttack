"""
Word Swap by Extension
-------------------------------
"""

from textattack.shared.data import EXTENSION_MAP

from .word_swap import WordSwap


class WordSwapExtend(WordSwap):
    """Transforms an input by performing extension on recognized
    combinations."""

    def _get_transformations(self, current_text, indices_to_modify):
        """Return all possible transformed sentences, each with one extension.

        >>> from textattack.transformations import WordSwapExtend
        >>> from textattack.augmentation import Augmenter

        >>> transformation = WordSwapExtend()
        >>> augmenter = Augmenter(transformation=transformation)
        >>> s = '''I'm fabulous'''
        >>> augmenter.augment(s)
        """
        transformed_texts = []
        words = current_text.words
        for idx in indices_to_modify:
            word = words[idx]
            # expend when word in map
            if word in EXTENSION_MAP:
                expanded = EXTENSION_MAP[word]
                transformed_text = current_text.replace_word_at_index(idx, expanded)
                transformed_texts.append(transformed_text)

        return transformed_texts
