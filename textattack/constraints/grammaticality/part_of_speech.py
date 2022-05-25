"""
Part of Speech Constraint
--------------------------
"""
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import lru
import nltk

import textattack
from textattack.constraints import Constraint
from textattack.shared.utils import LazyLoader, device
from textattack.shared.validators import transformation_consists_of_word_swaps

# Set global flair device to be TextAttack's current device
flair.device = device

stanza = LazyLoader("stanza", globals(), "stanza")


class PartOfSpeech(Constraint):
    """Constraints word swaps to only swap words with the same part of speech.
    Uses the NLTK universal part-of-speech tagger by default. An implementation
    of `<https://arxiv.org/abs/1907.11932>`_ adapted from
    `<https://github.com/jind11/TextFooler>`_.

    POS taggers from Flair `<https://github.com/flairNLP/flair>`_ and
    Stanza `<https://github.com/stanfordnlp/stanza>`_ are also available

    Args:
        tagger_type (str): Name of the tagger to use (available choices: "nltk", "flair", "stanza").
        tagset (str): tagset to use for POS tagging (e.g. "universal")
        allow_verb_noun_swap (bool): If `True`, allow verbs to be swapped with nouns and vice versa.
        compare_against_original (bool): If `True`, compare against the original text.
            Otherwise, compare against the most recent text.
        language_nltk: Language to be used for nltk POS-Tagger
            (available choices: "eng", "rus")
        language_stanza: Language to be used for stanza POS-Tagger
            (available choices: https://stanfordnlp.github.io/stanza/available_models.html)
    """

    def __init__(
        self,
        tagger_type="nltk",
        tagset="universal",
        allow_verb_noun_swap=True,
        compare_against_original=True,
        language_nltk="eng",
        language_stanza="en",
    ):
        super().__init__(compare_against_original)
        self.tagger_type = tagger_type
        self.tagset = tagset
        self.allow_verb_noun_swap = allow_verb_noun_swap
        self.language_nltk = language_nltk
        self.language_stanza = language_stanza

        self._pos_tag_cache = lru.LRU(2**14)
        if tagger_type == "flair":
            if tagset == "universal":
                self._flair_pos_tagger = SequenceTagger.load("upos-fast")
            else:
                self._flair_pos_tagger = SequenceTagger.load("pos-fast")

        if tagger_type == "stanza":
            self._stanza_pos_tagger = stanza.Pipeline(
                lang=self.language_stanza,
                processors="tokenize, pos",
                tokenize_pretokenized=True,
            )

    def clear_cache(self):
        self._pos_tag_cache.clear()

    def _can_replace_pos(self, pos_a, pos_b):
        return (pos_a == pos_b) or (
            self.allow_verb_noun_swap and set([pos_a, pos_b]) <= set(["NOUN", "VERB"])
        )

    def _get_pos(self, before_ctx, word, after_ctx):
        context_words = before_ctx + [word] + after_ctx
        context_key = " ".join(context_words)
        if context_key in self._pos_tag_cache:
            word_list, pos_list = self._pos_tag_cache[context_key]
        else:
            if self.tagger_type == "nltk":
                word_list, pos_list = zip(
                    *nltk.pos_tag(
                        context_words, tagset=self.tagset, lang=self.language_nltk
                    )
                )

            if self.tagger_type == "flair":
                context_key_sentence = Sentence(
                    context_key,
                    use_tokenizer=textattack.shared.utils.TextAttackFlairTokenizer(),
                )
                self._flair_pos_tagger.predict(context_key_sentence)
                word_list, pos_list = textattack.shared.utils.zip_flair_result(
                    context_key_sentence
                )

            if self.tagger_type == "stanza":
                word_list, pos_list = textattack.shared.utils.zip_stanza_result(
                    self._stanza_pos_tagger(context_key), tagset=self.tagset
                )

            self._pos_tag_cache[context_key] = (word_list, pos_list)

        # idx of `word` in `context_words`
        assert word in word_list, "POS list not matched with original word list."
        word_idx = word_list.index(word)
        return pos_list[word_idx]

    def _check_constraint(self, transformed_text, reference_text):
        try:
            indices = transformed_text.attack_attrs["newly_modified_indices"]
        except KeyError:
            raise KeyError(
                "Cannot apply part-of-speech constraint without `newly_modified_indices`"
            )

        for i in indices:
            reference_word = reference_text.words[i]
            transformed_word = transformed_text.words[i]
            before_ctx = reference_text.words[max(i - 4, 0) : i]
            after_ctx = reference_text.words[
                i + 1 : min(i + 4, len(reference_text.words))
            ]
            ref_pos = self._get_pos(before_ctx, reference_word, after_ctx)
            replace_pos = self._get_pos(before_ctx, transformed_word, after_ctx)
            if not self._can_replace_pos(ref_pos, replace_pos):
                return False

        return True

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

    def extra_repr_keys(self):
        return [
            "tagger_type",
            "tagset",
            "allow_verb_noun_swap",
        ] + super().extra_repr_keys()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pos_tag_cache"] = self._pos_tag_cache.get_size()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._pos_tag_cache = lru.LRU(state["_pos_tag_cache"])
