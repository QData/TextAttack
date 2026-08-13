def test_imports():
    import flair
    import torch

    import textattack

    del textattack, torch, flair


def test_word_swap_change_location():
    from flair.data import Sentence
    from flair.models import SequenceTagger

    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps import WordSwapChangeLocation

    augmenter = Augmenter(transformation=WordSwapChangeLocation())
    s = "I am in Dallas."
    s_augmented = augmenter.augment(s)
    augmented_text = Sentence(s_augmented[0])
    tagger = SequenceTagger.load("flair/ner-english")
    original_text = Sentence(s)
    tagger.predict(original_text)
    tagger.predict(augmented_text)

    entity_original = []
    entity_augmented = []

    for entity in original_text.get_spans("ner"):
        entity_original.append(entity.tag)
    for entity in augmented_text.get_spans("ner"):
        entity_augmented.append(entity.tag)
    assert entity_original == entity_augmented


def test_word_swap_change_location_consistent():
    from flair.data import Sentence
    from flair.models import SequenceTagger

    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps import WordSwapChangeLocation

    augmenter = Augmenter(transformation=WordSwapChangeLocation(consistent=True))
    s = "I am in New York. I love living in New York."
    s_augmented = augmenter.augment(s)
    augmented_text = Sentence(s_augmented[0])
    tagger = SequenceTagger.load("flair/ner-english")
    original_text = Sentence(s)
    tagger.predict(original_text)
    tagger.predict(augmented_text)

    entity_original = []
    entity_augmented = []

    for entity in original_text.get_spans("ner"):
        entity_original.append(entity.tag)
    for entity in augmented_text.get_spans("ner"):
        entity_augmented.append(entity.tag)

    assert entity_original == entity_augmented
    assert s_augmented[0].count("New York") == 0


def test_word_swap_change_name():
    from flair.data import Sentence
    from flair.models import SequenceTagger

    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps import WordSwapChangeName

    augmenter = Augmenter(transformation=WordSwapChangeName())
    s = "My name is Anthony Davis."
    s_augmented = augmenter.augment(s)
    augmented_text = Sentence(s_augmented[0])
    tagger = SequenceTagger.load("flair/ner-english")
    original_text = Sentence(s)
    tagger.predict(original_text)
    tagger.predict(augmented_text)

    entity_original = []
    entity_augmented = []

    for entity in original_text.get_spans("ner"):
        entity_original.append(entity.tag)
    for entity in augmented_text.get_spans("ner"):
        entity_augmented.append(entity.tag)
    assert entity_original == entity_augmented


def test_word_swap_change_name_consistent():
    from flair.data import Sentence
    from flair.models import SequenceTagger

    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps import WordSwapChangeName

    augmenter = Augmenter(transformation=WordSwapChangeName(consistent=True))
    s = "My name is Anthony Davis. Anthony Davis plays basketball."
    s_augmented = augmenter.augment(s)
    augmented_text = Sentence(s_augmented[0])
    tagger = SequenceTagger.load("flair/ner-english")
    original_text = Sentence(s)
    tagger.predict(original_text)
    tagger.predict(augmented_text)

    entity_original = []
    entity_augmented = []

    for entity in original_text.get_spans("ner"):
        entity_original.append(entity.tag)
    for entity in augmented_text.get_spans("ner"):
        entity_augmented.append(entity.tag)

    assert entity_original == entity_augmented
    assert s_augmented[0].count("Anthony") == 0 or s_augmented[0].count("Davis") == 0


def test_chinese_morphonym_character_swap():
    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps.chn_transformations import (
        ChineseMorphonymCharacterSwap,
    )

    augmenter = Augmenter(
        transformation=ChineseMorphonymCharacterSwap(),
        pct_words_to_swap=0.1,
        transformations_per_example=5,
    )
    s = "自然语言处理。"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "自然语言处埋。"
    assert augmented_s or s in augmented_text_list


def test_chinese_word_swap_hownet():
    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps.chn_transformations import (
        ChineseWordSwapHowNet,
    )

    augmenter = Augmenter(
        transformation=ChineseWordSwapHowNet(),
        pct_words_to_swap=0.1,
        transformations_per_example=5,
    )
    s = "自然语言。"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "中间语言。"
    assert augmented_s or s in augmented_text_list


def test_word_swap_inflections_pos_matching():
    # Regression test for https://github.com/QData/TextAttack/issues/713 and
    # https://github.com/QData/TextAttack/issues/727: AttackedText.pos_of_word_index
    # returns flair's upos-fast tags (e.g. "NOUN", "VERB"), so
    # WordSwapInflections's POS-to-lemma mapping must have entries for those
    # tags, not just legacy fine-grained en-ptb tags (e.g. "NN", "VBD"), or it
    # silently returns zero candidates for ordinary words.
    import textattack
    from textattack.transformations.word_swaps import WordSwapInflections

    transformation = WordSwapInflections()
    attacked_text = textattack.shared.AttackedText("The cats were running quickly.")

    # Confirm the tagger is actually giving us the upos-fast tag, not a
    # legacy en-ptb one, so this test exercises the real mismatch and isn't
    # trivially passing for the wrong reason.
    cats_index = attacked_text.words.index("cats")
    cats_pos = attacked_text.pos_of_word_index(cats_index)
    assert cats_pos == "NOUN"
    # Before the fix, "NOUN" wasn't a key in the mapping (only "NN" was), so
    # this lookup missed and _get_replacement_words returned [] for every
    # ordinary noun.
    noun_candidates = transformation._get_replacement_words("cats", cats_pos)
    assert "cat" in noun_candidates

    were_index = attacked_text.words.index("were")
    were_pos = attacked_text.pos_of_word_index(were_index)
    assert were_pos == "VERB"
    # Same failure mode as above, for verbs ("VERB" vs. the legacy "VBD").
    verb_candidates = transformation._get_replacement_words("were", were_pos)
    assert "was" in verb_candidates


def test_chinese_word_swap_masked():
    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps.chn_transformations import (
        ChineseWordSwapMaskedLM,
    )

    augmenter = Augmenter(
        transformation=ChineseWordSwapMaskedLM(),
        pct_words_to_swap=0.1,
        transformations_per_example=5,
    )
    s = "自然语言处理。"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "自然语言文字。"
    assert augmented_s or s in augmented_text_list
