import pytest


@pytest.mark.parametrize(
    "goal_function_name,model_module_path,expected_match",
    [
        # ForSequenceClassification: both the pre-4.x and current
        # transformers module layouts should match (#722).
        (
            "UntargetedClassification",
            "transformers.modeling_bert.BertForSequenceClassification",
            True,
        ),
        (
            "UntargetedClassification",
            "transformers.models.bert.modeling_bert.BertForSequenceClassification",
            True,
        ),
        (
            "UntargetedClassification",
            "transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification",
            True,
        ),
        # ForConditionalGeneration: raw transformers seq2seq generation
        # models, not just TextAttack's own T5ForTextToText helper (#771).
        (
            "NonOverlappingOutput",
            "transformers.models.t5.modeling_t5.T5ForConditionalGeneration",
            True,
        ),
        (
            "MinimizeBleu",
            "transformers.models.bart.modeling_bart.BartForConditionalGeneration",
            True,
        ),
        (
            "NonOverlappingOutput",
            "textattack.models.helpers.t5_for_text_to_text.T5ForTextToText",
            True,
        ),
        # A classification-headed model on an encoder-decoder backbone
        # should NOT match the generation entry.
        (
            "NonOverlappingOutput",
            "transformers.models.bart.modeling_bart.BartForSequenceClassification",
            False,
        ),
    ],
)
def test_model_goal_function_compatibility_regexes(
    goal_function_name, model_module_path, expected_match
):
    import re

    from textattack.goal_functions import (
        MinimizeBleu,
        NonOverlappingOutput,
        UntargetedClassification,
    )
    from textattack.shared.validators import MODELS_BY_GOAL_FUNCTION

    goal_function = {
        "UntargetedClassification": UntargetedClassification,
        "NonOverlappingOutput": NonOverlappingOutput,
        "MinimizeBleu": MinimizeBleu,
    }[goal_function_name]

    globs = MODELS_BY_GOAL_FUNCTION[goal_function]
    matched = any(re.match(glob, model_module_path) for glob in globs)
    assert matched == expected_match


def test_validate_model_goal_function_compatibility_no_warning_for_generation_model():
    # Regression test: every attack wrapping a raw transformers
    # encoder-decoder generation model (the capability
    # HuggingFaceModelWrapper's .generate() path (#771) was added for)
    # used to print a spurious "Unknown if model ... compatible" warning
    # for NonOverlappingOutput/MinimizeBleu, since MODELS_BY_GOAL_FUNCTIONS
    # only recognized TextAttack's own T5ForTextToText helper.
    import logging

    import transformers

    from textattack.goal_functions import NonOverlappingOutput
    from textattack.shared.utils import logger as ta_logger
    from textattack.shared.validators import validate_model_goal_function_compatibility

    # textattack's logger sets propagate=False (see
    # textattack/shared/utils/install.py) and uses its own StreamHandler,
    # so pytest's `caplog` fixture (which relies on propagation reaching a
    # handler on the root logger) never sees its records. Attach a handler
    # directly to the logger object instead, which works regardless of
    # propagation since it only affects bubbling to ancestor loggers, not
    # the logger's own handlers.
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = ListHandler()
    ta_logger.addHandler(handler)
    try:
        validate_model_goal_function_compatibility(
            NonOverlappingOutput, transformers.T5ForConditionalGeneration
        )
    finally:
        ta_logger.removeHandler(handler)

    assert not any("Unknown if model" in m for m in records)
