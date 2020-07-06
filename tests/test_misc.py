def test_imports():
    import torch

    import textattack

    del textattack, torch


def test_embedding_augmenter():
    from textattack.augmentation import EmbeddingAugmenter

    augmenter = EmbeddingAugmenter(transformations_per_example=64)
    s = "There is nothing either good or bad, but thinking makes it so."
    augmented_text_list = augmenter.augment(s)
    augmented_s = (
        "There is nothing either good or unfavourable, but thinking makes it so."
    )
    assert augmented_s in augmented_text_list
