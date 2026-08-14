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


def test_easydata_augmenter2():
    from textattack.augmentation import EasyDataAugmenter

    augmenter = EasyDataAugmenter(
        pct_words_to_swap=0.01, transformations_per_example=64
    )
    s = "hello hello hello derek"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "derek hello hello hello"
    assert augmented_s in augmented_text_list


def test_wordnet_augmenter():
    from textattack.augmentation import WordNetAugmenter

    augmenter = WordNetAugmenter(pct_words_to_swap=0.01, transformations_per_example=64)
    s = "The Dragon warrior is a panda"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "The firedrake warrior is a panda"
    assert augmented_s in augmented_text_list


def test_deletion_augmenter():
    from textattack.augmentation import DeletionAugmenter

    augmenter = DeletionAugmenter(pct_words_to_swap=0.1, transformations_per_example=10)
    s = "The United States"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "United States"
    assert augmented_s in augmented_text_list


def test_high_yield_scales_with_transformations_per_example():
    # Regression test: the retry-bound fix for issue #800 (stop a
    # low-diversity transformation from silently returning fewer than
    # `transformations_per_example` unique augmentations) had an
    # intermediate version whose outer loop exited as soon as the target
    # count was reached. In `high_yield=True` mode a single outer
    # iteration can add many results to the set at once, so that made
    # output plateau around the same size regardless of
    # `transformations_per_example` instead of scaling with it (~4-13x
    # fewer results, verified against pre-regression output for the same
    # input/seed range).
    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps import WordSwapWordNet

    s = "A person walks up stairs into a room and sees beer poured from a keg and people talking."

    def unique_count(n):
        augmenter = Augmenter(
            transformation=WordSwapWordNet(),
            pct_words_to_swap=0.15,
            transformations_per_example=n,
            high_yield=True,
        )
        return len(set(augmenter.augment(s)))

    small = unique_count(5)
    large = unique_count(20)
    # Loose bound (transformation output is stochastic): `large` should be
    # meaningfully bigger than `small`, not roughly flat.
    assert large > small * 2


def test_augment_dedup_sample_from_set_no_crash():
    # Regression test: the final downsampling step called
    # `random.sample(all_transformed_texts, n)` where
    # `all_transformed_texts` is a `set`. Python 3.11+ raises
    # `TypeError: Population must be a sequence` for a set argument, since
    # `random.sample` stopped accepting arbitrary sized iterables/sets.
    # `fast_augment=True, high_yield=False` is what exercises this
    # particular downsampling branch.
    from textattack.augmentation import Augmenter
    from textattack.transformations.word_swaps import WordSwapWordNet

    augmenter = Augmenter(
        transformation=WordSwapWordNet(),
        pct_words_to_swap=0.2,
        transformations_per_example=3,
        high_yield=False,
        fast_augment=True,
    )
    s = "A person walks up stairs into a room and sees beer poured from a keg and people talking."
    augmented_text_list = augmenter.augment(s)
    assert len(augmented_text_list) <= 3


def test_high_yield_fast_augment():
    from textattack.augmentation import WordNetAugmenter

    augmenter_hy = WordNetAugmenter(
        pct_words_to_swap=0.1, transformations_per_example=2, high_yield=True
    )
    augmenter_fa = WordNetAugmenter(
        pct_words_to_swap=0.1,
        transformations_per_example=2,
        high_yield=True,
        fast_augment=True,
    )
    augmenter = WordNetAugmenter(pct_words_to_swap=0.1, transformations_per_example=2)
    s = "The dragon warrior is a panda"
    augmented_text_list_hy = augmenter_hy.augment(s)
    augmented_text_list_fa = augmenter_fa.augment(s)
    augmented_text_list = augmenter.augment(s)

    check1 = (
        len(augmented_text_list_hy)
        >= len(augmented_text_list_fa)
        >= len(augmented_text_list)
    )
    check2 = True
    for augmented_text in augmented_text_list:
        if augmented_text not in augmented_text_list_hy:
            check2 = False
            break

    assert check1 and check2


def test_back_translation():
    from textattack.augmentation import Augmenter
    from textattack.transformations.sentence_transformations import BackTranslation

    augmenter = Augmenter(transformation=BackTranslation())
    s = "What on earth are you doing?"
    augmented_text_list = augmenter.augment(s)
    augmented_s = "What the hell are you doing?"
    assert augmented_s in augmented_text_list


def test_back_transcription():
    from textattack.augmentation import Augmenter
    from textattack.transformations.sentence_transformations import BackTranscription

    try:
        augmenter = Augmenter(transformation=BackTranscription())
    except ModuleNotFoundError:
        print(
            "To use BackTranscription augmenter, install `fairseq`, `g2p_en` and `librosa` libraries"
        )
    else:
        s = "What on earth are you doing?"
        augmented_text_list = augmenter.augment(s)
        assert augmented_text_list
