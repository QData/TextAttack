"""
Word Swap by chinese hownet
-------------------------------------
"""

import OpenHowNet

from . import WordSwap


class ChineseWordSwapHowNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    OpenHownet http://nlp.csai.tsinghua.edu.cn/."""

    def __init__(self, topk=5):
        self.hownet_dict = OpenHowNet.HowNetDict(init_sim=True)
        self.topk = topk
        self.wordCache = {}

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with N characters
        replaced by a homoglyph."""
        if word in self.wordCache:
            return self.wordCache[word]
        results = self.hownet_dict.get_nearest_words(word, language="zh", K=self.topk)
        synonyms = []
        if results:
            for key, value in results.items():
                synonyms = synonyms + value[1:]
                self.wordCache[word] = synonyms.copy()
                break
            return synonyms
        else:
            return []
