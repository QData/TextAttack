"""
Word Swap by Embedding
-------------------------------

Based on paper: `<arxiv.org/abs/1603.00892>`_

Paper title: Counter-fitting Word Vectors to Linguistic Constraints

"""
from textattack.shared import AbstractWordEmbedding, WordEmbedding

from .word_swap import WordSwap


class WordSwapEmbedding(WordSwap):
    """Transforms an input by replacing its words with synonyms in the word
    embedding space.

    Args:
        max_candidates (int): maximum number of synonyms to pick
        embedding (textattack.shared.AbstractWordEmbedding): Wrapper for word embedding
    >>> from textattack.transformations import WordSwapEmbedding
    >>> from textattack.augmentation import Augmenter

    >>> transformation = WordSwapEmbedding()
    >>> augmenter = Augmenter(transformation=transformation)
    >>> s = 'I am fabulous.'
    >>> augmenter.augment(s)
    """

    def __init__(self, max_candidates=15, embedding=None, **kwargs):
        super().__init__(**kwargs)
        if embedding is None:
            embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        self.max_candidates = max_candidates
        if not isinstance(embedding, AbstractWordEmbedding):
            raise ValueError(
                "`embedding` object must be of type `textattack.shared.AbstractWordEmbedding`."
            )
        self.embedding = embedding

    def _get_replacement_words(self, word):
        """Returns a list of possible 'candidate words' to replace a word in a
        sentence or phrase.

        Based on nearest neighbors selected word embeddings.
        """
        try:
            word_id = self.embedding.word2index(word.lower())
            nnids = self.embedding.nearest_neighbours(word_id, self.max_candidates)
            candidate_words = []
            for i, nbr_id in enumerate(nnids):
                nbr_word = self.embedding.index2word(nbr_id)
                candidate_words.append(recover_word_case(nbr_word, word))
            return candidate_words
        except KeyError:
            # This word is not in our word embedding database, so return an empty list.
            return []

    def extra_repr_keys(self):
        return ["max_candidates", "embedding"]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
