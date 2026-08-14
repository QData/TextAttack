def _build_attack(model, tokenizer, transformation):
    from textattack import Attack
    from textattack.constraints.pre_transformation import (
        RepeatModification,
        StopwordModification,
    )
    from textattack.goal_functions import UntargetedClassification
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from textattack.search_methods import GreedyWordSwapWIR

    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    goal_function = UntargetedClassification(wrapper)
    return Attack(
        goal_function,
        [RepeatModification(), StopwordModification()],
        transformation,
        GreedyWordSwapWIR(),
    )


def _model_and_tokenizer():
    import transformers

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "hf-internal-testing/tiny-random-bert"
    )
    return model, tokenizer


def test_cuda_skips_hf_device_map_model_but_moves_unrelated_module():
    # Regression test: the `hf_device_map` skip in `Attack.cuda_`/`to_cuda`
    # used to apply to any `torch.nn.Module` with a truthy `hf_device_map`
    # attribute, not just `transformers.PreTrainedModel` instances. Since
    # this visitor also traverses non-HuggingFace modules reachable from a
    # Constraint/GoalFunction/Transformation, a coincidental attribute name
    # collision would silently skip moving that module to the configured
    # device. Confirm the HF model is still (correctly) skipped, while an
    # unrelated module with the same attribute name is not.
    from unittest.mock import patch

    import torch

    from textattack.transformations import WordSwapRandomCharacterDeletion

    model, tokenizer = _model_and_tokenizer()
    model.hf_device_map = {"": "cpu"}

    transformation = WordSwapRandomCharacterDeletion()
    transformation.some_unrelated_module = torch.nn.Linear(2, 2)
    transformation.some_unrelated_module.hf_device_map = {"": "cpu"}

    attack = _build_attack(model, tokenizer, transformation)

    with (
        patch.object(model, "to", wraps=model.to) as model_to_spy,
        patch.object(
            transformation.some_unrelated_module,
            "to",
            wraps=transformation.some_unrelated_module.to,
        ) as marker_to_spy,
    ):
        attack.cuda_()

    assert model_to_spy.called is False
    assert marker_to_spy.called is True


def test_cpu_skips_hf_device_map_model_but_moves_unrelated_module():
    # Same guard as cuda_/to_cuda, added separately to cpu_/to_cpu.
    from unittest.mock import patch

    import torch

    from textattack.transformations import WordSwapRandomCharacterDeletion

    model, tokenizer = _model_and_tokenizer()
    model.hf_device_map = {"": "cpu"}

    transformation = WordSwapRandomCharacterDeletion()
    transformation.some_unrelated_module = torch.nn.Linear(2, 2)
    transformation.some_unrelated_module.hf_device_map = {"": "cpu"}

    attack = _build_attack(model, tokenizer, transformation)

    with (
        patch.object(model, "cpu", wraps=model.cpu) as model_cpu_spy,
        patch.object(
            transformation.some_unrelated_module,
            "cpu",
            wraps=transformation.some_unrelated_module.cpu,
        ) as marker_cpu_spy,
    ):
        attack.cpu_()

    assert model_cpu_spy.called is False
    assert marker_cpu_spy.called is True


def test_cuda_moves_model_without_device_map():
    from unittest.mock import patch

    from textattack.transformations import WordSwapRandomCharacterDeletion

    model, tokenizer = _model_and_tokenizer()
    transformation = WordSwapRandomCharacterDeletion()
    attack = _build_attack(model, tokenizer, transformation)

    with patch.object(model, "to", wraps=model.to) as model_to_spy:
        attack.cuda_()

    assert model_to_spy.called is True
