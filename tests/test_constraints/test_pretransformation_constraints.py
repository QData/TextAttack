import collections

import pytest

import textattack

# a sentence with 23 words
sentence = "south korea's inflation rate slowed slightly in august as oil and commodity prices showed signs of stabilising, the national statistical office said monday."

# a premise with 36 words
premise = "Among these are the red brick Royal Palace, which now houses the Patan Museum (Nepal's finest and most modern museum), and, facing the palace across the narrow brick plaza, eight temples of different styles and sizes."
# a hypothesis with 13 words
hypothesis = "The Patan Museum is down the street from the red brick Royal Palace."


@pytest.fixture
def sentence_attacked_text():
    return textattack.shared.AttackedText(sentence)


@pytest.fixture
def entailment_attacked_text():
    raw_text_pair = collections.OrderedDict(
        [("premise", premise), ("hypothesis", hypothesis)]
    )
    return textattack.shared.AttackedText(raw_text_pair)


class TestPretransformationConstraints:
    def test_input_column_modification_basic(
        self, sentence_attacked_text, entailment_attacked_text
    ):
        constraint = textattack.constraints.pre_transformation.InputColumnModification(
            ["text"], {}
        )
        assert constraint._get_modifiable_indices(sentence_attacked_text) == set(
            range(23)
        )

        assert constraint._get_modifiable_indices(entailment_attacked_text) == set(
            range(49)
        )

    def test_input_column_modification_premise(self, entailment_attacked_text):
        constraint = textattack.constraints.pre_transformation.InputColumnModification(
            ["premise", "hypothesis"],
            {"hypothesis"},  # don't modify 'hypothesis' column
        )
        assert constraint._get_modifiable_indices(entailment_attacked_text) == set(
            range(0, 36)
        )

    def test_input_column_modification_hypothesis(self, entailment_attacked_text):
        constraint = textattack.constraints.pre_transformation.InputColumnModification(
            ["premise", "hypothesis"], {"premise"}  # don't modify 'premise' column
        )
        assert constraint._get_modifiable_indices(entailment_attacked_text) == set(
            range(36, 49)
        )

    def test_max_word_index(self, sentence_attacked_text):
        short_constraint = (
            textattack.constraints.pre_transformation.MaxWordIndexModification(8)
        )
        assert short_constraint._get_modifiable_indices(sentence_attacked_text) == set(
            range(8)
        )

        long_constraint = (
            textattack.constraints.pre_transformation.MaxWordIndexModification(8000)
        )
        assert long_constraint._get_modifiable_indices(sentence_attacked_text) == set(
            range(len(sentence_attacked_text.words))
        )

    def test_repeat_modification(
        self, sentence_attacked_text, entailment_attacked_text
    ):
        constraint = textattack.constraints.pre_transformation.RepeatModification()
        assert constraint._get_modifiable_indices(sentence_attacked_text) == set(
            range(len(sentence_attacked_text.words))
        )
        assert constraint._get_modifiable_indices(entailment_attacked_text) == set(
            range(len(entailment_attacked_text.words))
        )
        sentence_attacked_text.attack_attrs["modified_indices"] = {0, 1, 2, 3}
        assert constraint._get_modifiable_indices(sentence_attacked_text) == (
            set(range(len(sentence_attacked_text.words))) - {0, 1, 2, 3}
        )
        entailment_attacked_text.attack_attrs["modified_indices"] = {1, 3, 11, 17, 23}
        assert constraint._get_modifiable_indices(entailment_attacked_text) == (
            set(range(len(entailment_attacked_text.words))) - {1, 3, 11, 17, 23}
        )

    def test_stopword_modification(
        self, sentence_attacked_text, entailment_attacked_text
    ):
        constraint = textattack.constraints.pre_transformation.StopwordModification()
        assert constraint._get_modifiable_indices(sentence_attacked_text) == (
            set(range(len(sentence_attacked_text.words))) - {6, 8, 10, 15, 17}
        )
        assert constraint._get_modifiable_indices(entailment_attacked_text) == (
            set(range(len(entailment_attacked_text.words)))
            - {1, 2, 3, 8, 9, 11, 16, 17, 20, 22, 25, 31, 34, 39, 40, 41, 43, 44}
        )
