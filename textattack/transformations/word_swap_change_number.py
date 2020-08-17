import numpy as np
from textattack.transformations.word_swap import WordSwap


class WordSwapChangeNumber(WordSwap):

    """
    Future implementations:
    detect alphabetical numbers as well
    """

    def __init__(
        self, max_change=1, n=10, **kwargs
    ):
        super().__init__(**kwargs)
        self.max_change = max_change
        self.n = n

    def _get_replacement_words(self, word):

        if word.isdigit() and word != "2" and word != "4":
            num = int(word)
            change = int(num * self.max_change) + 1
            num_list = np.random.randint(num - change, num + change, self.n)
            print(num_list)
            return num_list.astype(str)



        return []


