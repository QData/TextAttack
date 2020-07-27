from flair.data import Sentence
from flair.models import SequenceTagger
import lemminflect

from textattack.transformations.word_swap import WordSwap


class WordSwapInflections(WordSwap):
    """Transforms an input by replacing its words with their inflections.

    For example, the inflections of 'schedule' are {'schedule', 'schedules',
    'scheduling'}.

    Base on ``It’s Morphin’ Time! Combating Linguistic Discrimination with
    Inflectional Perturbations".
    (https://www.aclweb.org/anthology/2020.acl-main.263.pdf)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._flair_pos_tagger = SequenceTagger.load("pos-fast")
        self._flair_to_lemminflect_pos_map = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJ"}

    def _get_replacement_words(self, word, word_part_of_speech):
        if word_part_of_speech not in self._flair_to_lemminflect_pos_map:
            # Only nouns, verbs, and adjectives have proper inflections.
            return []
        replacement_inflections_dict = lemminflect.getAllLemmas(word)
        # `lemminflect.getAllLemmas` returns a dict mapping part-of-speech
        # to available inflections. First, map part-of-speech from flair
        # POS tag to lemminflect.
        lemminflect_pos = self._flair_to_lemminflect_pos_map[word_part_of_speech]
        return replacement_inflections_dict.get(lemminflect_pos, None)

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        sentence = Sentence(" ".join(words))
        self._flair_pos_tagger.predict(sentence)
        word_list, pos_list = zip_flair_result(sentence)

        assert len(words) == len(
            word_list
        ), "Part-of-speech tagger returned incorrect number of tags"

        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            word_to_replace_pos = pos_list[i][:2]  # get the root POS
            replacement_words = (
                self._get_replacement_words(word_to_replace, word_to_replace_pos) or []
            )
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))

        return transformed_texts


def zip_flair_result(pred):
    """Parse the output from the FLAIR POS tagger."""
    if not isinstance(pred, Sentence):
        raise TypeError("Result from Flair POS tagger must be a `Sentence` object.")

    tokens = pred.tokens
    word_list = []
    pos_list = []
    for token in tokens:
        word_list.append(token.text)
        pos_list.append(token.annotation_layers["pos"][0]._value)

    return word_list, pos_list
