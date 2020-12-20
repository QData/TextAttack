"""
Word Swap by Contraction
============================================
"""

from textattack.shared.data import EXTENSION_MAP

from .word_swap import WordSwap


class WordSwapContract(WordSwap):
    """Transforms an input by performing contraction on recognized
    combinations."""

    reverse_contraction_map = {v: k for k, v in EXTENSION_MAP.items()}

    def _get_transformations(self, current_text, indices_to_modify):
        """Return all possible transformed sentences, each with one
        contraction."""
        transformed_texts = []

        words = current_text.words
        indices_to_modify = sorted(indices_to_modify)

        # search for every 2-words combination in reverse_contraction_map
        for idx, word_idx in enumerate(indices_to_modify[:-1]):
            next_idx = indices_to_modify[idx + 1]
            if (idx + 1) != next_idx:
                continue
            word = words[word_idx]
            next_word = words[next_idx]

            # generating the words to search for
            key = " ".join([word, next_word])

            # when a possible contraction is found in map, contract the current text
            if key in self.reverse_contraction_map:
                transformed_text = current_text.replace_word_at_index(
                    idx, self.reverse_contraction_map[key]
                )
                transformed_text = transformed_text.delete_word_at_index(next_idx)
                transformed_texts.append(transformed_text)

        return transformed_texts
