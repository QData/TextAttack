"""
Word Swap by inflections
-------------------------------


"""

import random

import lemminflect

from .word_swap import WordSwap


class WordSwapInflections(WordSwap):
    """Transforms an input by replacing its words with their inflections.

    For example, the inflections of 'schedule' are {'schedule', 'schedules', 'scheduling'}.

    Base on ``It’s Morphin’ Time! Combating Linguistic Discrimination with Inflectional Perturbations``.

    `Paper URL`_

    .. _Paper URL: https://www.aclweb.org/anthology/2020.acl-main.263.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # `AttackedText.pos_of_word_index` now returns flair's "upos-fast" tags
        # directly (Universal Dependencies UPOS, e.g. "NOUN", "VERB", "PROPN"),
        # not the old fine-grained en-ptb tags this dict's keys used to assume
        # (see https://github.com/QData/TextAttack/issues/713,
        # https://github.com/QData/TextAttack/issues/727). Without the plain
        # "NOUN"/"VERB"/"ADJ" entries below, this lookup always misses and the
        # transformation silently returns no candidates for ordinary words.
        # Kept the legacy en-ptb keys too in case a caller wires up a
        # PTB-style tagger.
        # (mapping info: https://universaldependencies.org/u/pos/,
        # https://github.com/slavpetrov/universal-pos-tags)
        self._enptb_to_universal = {
            # current flair "upos-fast" (Universal Dependencies) tags
            "NOUN": "NOUN",
            "PROPN": "NOUN",
            "VERB": "VERB",
            "AUX": "VERB",
            "ADJ": "ADJ",
            # legacy fine-grained en-ptb tags
            "JJRJR": "ADJ",
            "VBN": "VERB",
            "VBP": "VERB",
            "JJ": "ADJ",
            "VBZ": "VERB",
            "VBG": "VERB",
            "NN": "NOUN",
            "VBD": "VERB",
            "NP": "NOUN",
            "NNP": "NOUN",
            "VB": "VERB",
            "NNS": "NOUN",
            "VP": "VERB",
            "TO": "VERB",
            "SYM": "NOUN",
            "MD": "VERB",
            "NNPS": "NOUN",
            "JJS": "ADJ",
            "JJR": "ADJ",
            "RB": "ADJ",
        }

    def _get_replacement_words(self, word, word_part_of_speech):
        # only nouns, verbs, and adjectives are considered for replacement
        if word_part_of_speech not in self._enptb_to_universal:
            return []

        # gets a dict that maps part-of-speech (POS) to available lemmas
        replacement_inflections_dict = lemminflect.getAllLemmas(word)

        # if dict is empty, there are no replacements for this word
        if not replacement_inflections_dict:
            return []

        # map the fine-grained POS to a universal POS
        lemminflect_pos = self._enptb_to_universal[word_part_of_speech]

        # choose lemma with same POS, if ones exists; otherwise, choose lemma randomly
        if lemminflect_pos in replacement_inflections_dict:
            lemma = replacement_inflections_dict[lemminflect_pos][0]
        else:
            lemma = random.choice(list(replacement_inflections_dict.values()))[0]

        # get the available inflections for chosen lemma
        inflections = lemminflect.getAllInflections(
            lemma, upos=lemminflect_pos
        ).values()

        # merge tuples, remove duplicates, remove copy of the original word
        replacement_words = list(set([infl for tup in inflections for infl in tup]))
        replacement_words = [r for r in replacement_words if r != word]

        return replacement_words

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        for i in indices_to_modify:
            word_to_replace = current_text.words[i]
            word_to_replace_pos = current_text.pos_of_word_index(i)
            replacement_words = (
                self._get_replacement_words(word_to_replace, word_to_replace_pos) or []
            )
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))

        return transformed_texts
