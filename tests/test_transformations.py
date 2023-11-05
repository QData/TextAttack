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
