import OpenHowNet

from .word_swap import WordSwap


class ChineseWordSwapHowNet(WordSwap):
    """Transforms an input by replacing its words with synonyms provided by
    WordNet."""

    def __init__(self):
        self.hownet_dict = OpenHowNet.HowNetDict(use_sim=True)
        self.topk = 10

    def _get_replacement_words(self, word):
        """Returns a list containing all possible words with N characters
        replaced by a homoglyph."""
        if self.hownet_dict.get(word):
            results = self.hownet_dict.get_nearest_words_via_sememes(word, self.topk)
            synonyms = [
                w["word"] for r in results for w in r["synset"] if w["word"] != word
            ]
            return synonyms
        else:
            return []
