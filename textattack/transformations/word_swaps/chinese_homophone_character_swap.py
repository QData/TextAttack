import pinyin
import pandas as pd
from .word_swap import WordSwap

class ChineseHomophoneCharacterSwap(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    a homophone dictionary."""

    def __init__(self):
        homophone_dict = pd.read_csv('chinese_homophone_char.txt', header=None, sep='\n')

        homophone_dict = homophone_dict[0].str.split('\t', expand=True)

        self.homophone_dict = homophone_dict

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with 1 character
        replaced by a homophne."""
        homophones = []

        word = pinyin.get(word, format="strip", delimiter=" ")
        if word in self.homophone_dict.values:
            for row in range(self.homophone_dict.shape[0]):  # df is the DataFrame
                for col in range(0, 1):
                    if self.homophone_dict._get_value(row, col) == word:
                        for i in range(1, 4):
                            homophones.append(self.homophone_dict[col + i][row])
            return homophones
        else:
            return []