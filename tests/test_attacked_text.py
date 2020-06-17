import collections

import pytest

import textattack

raw_text = "A person walks up stairs into a room and sees beer poured from a keg and people talking."


@pytest.fixture
def attacked_text():
    return textattack.shared.AttackedText(raw_text)


premise = "Among these are the red brick Royal Palace, which now houses the Patan Museum (Nepal's finest and most modern museum), and, facing the palace across the narrow brick plaza, eight temples of different styles and sizes."
hypothesis = "The Patan Museum is down the street from the red brick Royal Palace."
raw_text_pair = collections.OrderedDict(
    [("premise", premise), ("hypothesis", hypothesis)]
)


@pytest.fixture
def attacked_text_pair():
    return textattack.shared.AttackedText(raw_text_pair)


class TestAttackedText:
    def test_words(self, attacked_text):
        assert attacked_text.words == [
            "A",
            "person",
            "walks",
            "up",
            "stairs",
            "into",
            "a",
            "room",
            "and",
            "sees",
            "beer",
            "poured",
            "from",
            "a",
            "keg",
            "and",
            "people",
            "talking",
        ]

    def test_window_around_index(self, attacked_text):
        assert attacked_text.text_window_around_index(5, 1) == "into"
        assert attacked_text.text_window_around_index(5, 2) == "stairs into"
        assert attacked_text.text_window_around_index(5, 3) == "stairs into a"
        assert attacked_text.text_window_around_index(5, 4) == "up stairs into a"
        assert attacked_text.text_window_around_index(5, 5) == "up stairs into a room"

    def test_big_window_around_index(self, attacked_text):
        assert (
            attacked_text.text_window_around_index(0, 10 ** 5) + "."
        ) == attacked_text.text

    def test_window_around_index_start(self, attacked_text):
        assert attacked_text.text_window_around_index(0, 3) == "A person walks"

    def test_window_around_index_end(self, attacked_text):
        assert attacked_text.text_window_around_index(17, 3) == "and people talking"

    def test_text(self, attacked_text, attacked_text_pair):
        assert attacked_text.text == raw_text
        assert attacked_text_pair.text == "\n".join(raw_text_pair.values())

    def test_printable_text(self, attacked_text, attacked_text_pair):
        assert attacked_text.printable_text == raw_text
        desired_printed_pair_text = (
            "Premise: " + premise + "\n\n" + "Hypothesis: " + hypothesis
        )
        print("p =>", attacked_text_pair.printable_text)
        print("d =>", desired_printed_pair_text)
        assert attacked_text_pair.printable_text == desired_printed_pair_text

    def test_tokenizer_input(self, attacked_text, attacked_text_pair):
        assert attacked_text.tokenizer_input == (raw_text,)
        assert attacked_text_pair.tokenizer_input == (premise, hypothesis)

    def test_word_replacement(self, attacked_text):
        assert (
            attacked_text.replace_word_at_index(3, "down").text
            == "A person walks down stairs into a room and sees beer poured from a keg and people talking."
        )
        assert (
            attacked_text.replace_word_at_index(10, "wine").text
            == "A person walks up stairs into a room and sees wine poured from a keg and people talking."
        )

    def test_multi_word_replacement(self, attacked_text):
        new_text = attacked_text.replace_words_at_indices(
            (0, 3, 10, 14, 17), ("The", "down", "wine", "bottle", "sitting")
        )
        assert (
            new_text.text
            == "The person walks down stairs into a room and sees wine poured from a bottle and people sitting."
        )

    def test_word_deletion(self, attacked_text):
        new_text = attacked_text.delete_word_at_index(4).delete_word_at_index(16)
        assert (
            new_text.text
            == "A person walks up into a room and sees beer poured from a keg and people."
        )
