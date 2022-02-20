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
