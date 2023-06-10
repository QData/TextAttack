from textattack.shared.data import MORPHONYM_LS

from . import WordSwap


class ChineseMorphonymCharacterSwap(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by a
    morphonym dictionary."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a morphonym."""
        word = list(word)
        candidate_words = set()
        for i in range(len(word)):
            character = word[i]
            for char_morpho_ls in MORPHONYM_LS:
                if character in char_morpho_ls:
                    for new_char in char_morpho_ls:
                        temp_word = word
                        temp_word[i] = new_char
                        candidate_words.add("".join(temp_word))
        return list(candidate_words)
