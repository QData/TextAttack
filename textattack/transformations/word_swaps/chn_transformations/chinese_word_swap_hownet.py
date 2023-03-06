import OpenHowNet

from . import WordSwap


class ChineseWordSwapHowNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    OpenHownet http://nlp.csai.tsinghua.edu.cn/."""

    def __init__(self, topk=5):
        self.hownet_dict = OpenHowNet.HowNetDict(init_sim=True)
        self.topk = topk

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with N characters
        replaced by a homoglyph."""
        results = self.hownet_dict.get_nearest_words(word, language="zh", K=self.topk)
        synonyms = []
        if results:
            for key, value in results.items():
                for w in value:
                    synonyms.append(w)
            return synonyms
        else:
            return []
