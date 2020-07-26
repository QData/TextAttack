import collections

import pytest

import textattack

raw_text = "A person walks up stairs into a room and sees beer poured from a keg and people talking."


@pytest.fixture
def attacked_text():
    return textattack.shared.AttackedText(raw_text)


raw_pokemon_text = "the threat implied in the title pokémon 4ever is terrifying  like locusts in a horde these things will keep coming ."


@pytest.fixture
def pokemon_attacked_text():
    return textattack.shared.AttackedText(raw_pokemon_text)


premise = "Among these are the red brick Royal Palace, which now houses the Patan Museum (Nepal's finest and most modern museum), and, facing the palace across the narrow brick plaza, eight temples of different styles and sizes."
hypothesis = "The Patan Museum is down the street from the red brick Royal Palace."
raw_text_pair = collections.OrderedDict(
    [("premise", premise), ("hypothesis", hypothesis)]
)

raw_hyphenated_text = "It's a run-of-the-mill kind of farmer's tan."


@pytest.fixture
def hyphenated_text():
    return textattack.shared.AttackedText(raw_hyphenated_text)


@pytest.fixture
def attacked_text_pair():
    return textattack.shared.AttackedText(raw_text_pair)


class TestAttackedText:
    def test_words(self, attacked_text, pokemon_attacked_text):
        # fmt: off
        assert attacked_text.words == [
            "A", "person", "walks", "up", "stairs", "into", "a", "room", "and", "sees", "beer", "poured", "from", "a", "keg", "and", "people", "talking",
        ]
        assert pokemon_attacked_text.words == ['the', 'threat', 'implied', 'in', 'the', 'title', 'pokémon', '4ever', 'is', 'terrifying', 'like', 'locusts', 'in', 'a', 'horde', 'these', 'things', 'will', 'keep', 'coming']
        # fmt: on

    def test_window_around_index(self, attacked_text):
        assert attacked_text.text_window_around_index(5, 1) == "into"
        assert attacked_text.text_window_around_index(5, 2) == "stairs into"
        assert attacked_text.text_window_around_index(5, 3) == "stairs into a"
        assert attacked_text.text_window_around_index(5, 4) == "up stairs into a"
        assert attacked_text.text_window_around_index(5, 5) == "up stairs into a room"
        assert (
            attacked_text.text_window_around_index(5, float("inf"))
            == "A person walks up stairs into a room and sees beer poured from a keg and people talking"
        )

    def test_big_window_around_index(self, attacked_text):
        assert (
            attacked_text.text_window_around_index(0, 10 ** 5) + "."
        ) == attacked_text.text

    def test_window_around_index_start(self, attacked_text):
        assert attacked_text.text_window_around_index(0, 3) == "A person walks"

    def test_window_around_index_end(self, attacked_text):
        assert attacked_text.text_window_around_index(17, 3) == "and people talking"

    def test_text(self, attacked_text, pokemon_attacked_text, attacked_text_pair):
        assert attacked_text.text == raw_text
        assert pokemon_attacked_text.text == raw_pokemon_text
        assert attacked_text_pair.text == "\n".join(raw_text_pair.values())

    def test_printable_text(self, attacked_text, attacked_text_pair):
        assert attacked_text.printable_text() == raw_text
        desired_printed_pair_text = (
            "Premise: " + premise + "\n" + "Hypothesis: " + hypothesis
        )
        print("p =>", attacked_text_pair.printable_text())
        print("d =>", desired_printed_pair_text)
        assert attacked_text_pair.printable_text() == desired_printed_pair_text

    def test_tokenizer_input(self, attacked_text, attacked_text_pair):
        assert attacked_text.tokenizer_input == raw_text
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
        new_text = (
            new_text.delete_word_at_index(0)
            .delete_word_at_index(0)
            .delete_word_at_index(0)
        )
        assert (
            new_text.text
            == "up into a room and sees beer poured from a keg and people."
        )

    def test_word_insertion(self, attacked_text):
        new_text = attacked_text.insert_text_before_word_index(3, "a long way")
        assert (
            new_text.text
            == "A person walks a long way up stairs into a room and sees beer poured from a keg and people talking."
        )
        new_text = new_text.insert_text_after_word_index(19, "on the couch")
        assert (
            new_text.text
            == "A person walks a long way up stairs into a room and sees beer poured from a keg and people on the couch talking."
        )

    def test_pair_word_insertion(self, attacked_text_pair):
        new_text = attacked_text_pair.insert_text_after_word_index(3, "old decrepit")
        assert new_text.text == (
            "Among these are the old decrepit red brick Royal Palace, which now houses the Patan Museum (Nepal's finest and most modern museum), and, facing the palace across the narrow brick plaza, eight temples of different styles and sizes."
            + "\n"
            + "The Patan Museum is down the street from the red brick Royal Palace."
        )
        new_text = new_text.insert_text_after_word_index(37, "and shapes")
        assert new_text.text == (
            "Among these are the old decrepit red brick Royal Palace, which now houses the Patan Museum (Nepal's finest and most modern museum), and, facing the palace across the narrow brick plaza, eight temples of different styles and sizes and shapes."
            + "\n"
            + "The Patan Museum is down the street from the red brick Royal Palace."
        )
        new_text = new_text.insert_text_after_word_index(40, "The")
        assert new_text.text == (
            "Among these are the old decrepit red brick Royal Palace, which now houses the Patan Museum (Nepal's finest and most modern museum), and, facing the palace across the narrow brick plaza, eight temples of different styles and sizes and shapes."
            + "\n"
            + "The The Patan Museum is down the street from the red brick Royal Palace."
        )

    def test_modified_indices(self, attacked_text):
        new_text = attacked_text.insert_text_after_word_index(
            2, "a very long way"
        ).insert_text_after_word_index(20, "on the couch")
        assert (
            new_text.text
            == "A person walks a very long way up stairs into a room and sees beer poured from a keg and people on the couch talking."
        )
        for old_idx, new_idx in enumerate(new_text.attack_attrs["original_index_map"]):
            assert (attacked_text.words[old_idx] == new_text.words[new_idx]) or (
                new_idx == -1
            )
        new_text = (
            new_text.delete_word_at_index(0)
            .delete_word_at_index(15)
            .delete_word_at_index(15)
            .delete_word_at_index(15)
            .delete_word_at_index(20)
        )
        for old_idx, new_idx in enumerate(new_text.attack_attrs["original_index_map"]):
            assert (attacked_text.words[old_idx] == new_text.words[new_idx]) or (
                new_idx == -1
            )
        assert (
            new_text.text
            == "person walks a very long way up stairs into a room and sees beer poured and people on the couch."
        )

    def test_hyphen_apostrophe_words(self, hyphenated_text):
        assert hyphenated_text.words == [
            "It's",
            "a",
            "run-of-the-mill",
            "kind",
            "of",
            "farmer's",
            "tan",
        ]
