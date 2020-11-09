def test_imports():
    import torch

    import textattack

    del textattack, torch


def test_embedding_augmenter():
    from textattack.augmentation import EmbeddingAugmenter

    augmenter = EmbeddingAugmenter(
        pct_words_to_swap=0.01, transformations_per_example=64
    )
    s = "There is nothing either good or bad, but thinking makes it so."
    augmented_text_list = augmenter.augment(s)
    augmented_s = (
        "There is nothing either good or unfavourable, but thinking makes it so."
    )
    assert augmented_s in augmented_text_list


def test_checklist_augmenter():
    from textattack.augmentation import CheckListAugmenter

    augmenter = CheckListAugmenter(
        pct_words_to_swap=0.01, transformations_per_example=64
    )
    s = "I'll be happy to assist you."
    augmented_text_list = augmenter.augment(s)
    augmented_s = "I will be happy to assist you."
    assert augmented_s in augmented_text_list

    s = "I will be happy to assist you."
    augmented_text_list = augmenter.augment(s)
    augmented_s = "I'll be happy to assist you."
    assert augmented_s in augmented_text_list


def test_charwap_augmenter():
    from textattack.augmentation import CharSwapAugmenter

    augmenter = CharSwapAugmenter(
        pct_words_to_swap=0.01, transformations_per_example=64
    )
    s = "To be or not to be"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "T be or not to be"
    assert augmented_s in augmented_text_list


def test_easydata_augmenter():
    from textattack.augmentation import EasyDataAugmenter

    augmenter = EasyDataAugmenter(
        pct_words_to_swap=0.01, transformations_per_example=64
    )
    s = "Hakuna Montana"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "Montana Hakuna"
    assert augmented_s in augmented_text_list


def test_wordnet_augmenter():
    from textattack.augmentation import WordNetAugmenter

    augmenter = WordNetAugmenter(pct_words_to_swap=0.01, transformations_per_example=64)
    s = "The Dragon warrior is a panda"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "The firedrake warrior is a panda"
    assert augmented_s in augmented_text_list
