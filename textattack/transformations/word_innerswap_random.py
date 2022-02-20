"""
Word Swap Transformation by swapping the order of words
==========================================================
"""


import random

from textattack.transformations import Transformation


class WordInnerSwapRandom(Transformation):
    """Transformation that randomly swaps the order of words in a sequence."""

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        words = current_text.words
        for idx in indices_to_modify:
            word = words[idx]
            swap_idxs = list(set(range(len(words))) - {idx})
            if swap_idxs:
                swap_idx = random.choice(swap_idxs)
                swapped_text = current_text.replace_word_at_index(
                    idx, words[swap_idx]
                ).replace_word_at_index(swap_idx, word)
                transformed_texts.append(swapped_text)
        return transformed_texts

    @property
    def deterministic(self):
        return False
