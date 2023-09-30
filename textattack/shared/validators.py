"""
Misc Validators
=================
Validators ensure compatibility between search methods, transformations, constraints, and goal functions.

"""

import re

import textattack
from textattack.goal_functions import (
    InputReduction,
    MinimizeBleu,
    NonOverlappingOutput,
    TargetedClassification,
    UntargetedClassification,
)

from . import logger

# A list of goal functions and the corresponding available models.
MODELS_BY_GOAL_FUNCTIONS = {
    (TargetedClassification, UntargetedClassification, InputReduction): [
        r"^textattack.models.helpers.lstm_for_classification.*",
        r"^textattack.models.helpers.word_cnn_for_classification.*",
        r"^transformers.modeling_\w*\.\w*ForSequenceClassification$",
    ],
    (
        NonOverlappingOutput,
        MinimizeBleu,
    ): [
        r"^textattack.models.helpers.t5_for_text_to_text.*",
    ],
}

# Unroll the `MODELS_BY_GOAL_FUNCTIONS` dictionary into a dictionary that has
# a key for each goal function. (Note the plurality here that distinguishes
# the two variables from one another.)
MODELS_BY_GOAL_FUNCTION = {}
for goal_functions, matching_model_globs in MODELS_BY_GOAL_FUNCTIONS.items():
    for goal_function in goal_functions:
        MODELS_BY_GOAL_FUNCTION[goal_function] = matching_model_globs


def validate_model_goal_function_compatibility(goal_function_class, model_class):
    """Determines if ``model_class`` is task-compatible with
    ``goal_function_class``.

    For example, a text-generative model like one intended for
    translation or summarization would not be compatible with a goal
    function that requires probability scores, like the
    UntargetedGoalFunction.
    """
    # Verify that this is a valid goal function.
    try:
        matching_model_globs = MODELS_BY_GOAL_FUNCTION[goal_function_class]
    except KeyError:
        matching_model_globs = []
        logger.warn(f"No entry found for goal function {goal_function_class}.")
    # Get options for this goal function.
    # model_module = model_class.__module__
    model_module_path = ".".join((model_class.__module__, model_class.__name__))
    # Ensure the model matches one of these options.
    for glob in matching_model_globs:
        if re.match(glob, model_module_path):
            logger.info(
                f"Goal function {goal_function_class} compatible with model {model_class.__name__}."
            )
            return
    # If we got here, the model does not match the intended goal function.
    for goal_functions, globs in MODELS_BY_GOAL_FUNCTIONS.items():
        for glob in globs:
            if re.match(glob, model_module_path):
                logger.warn(
                    f"Unknown if model {model_class.__name__} compatible with provided goal function {goal_function_class}."
                    f" Found match with other goal functions: {goal_functions}."
                )
                return
    # If it matches another goal function, warn user.

    # Otherwise, this is an unknown model–perhaps user-provided, or we forgot to
    # update the corresponding dictionary. Warn user and return.
    logger.warn(
        f"Unknown if model of class {model_class} compatible with goal function {goal_function_class}."
    )


def validate_model_gradient_word_swap_compatibility(model):
    """Determines if ``model`` is task-compatible with
    ``GradientBasedWordSwap``.

    We can only take the gradient with respect to an individual word if
    the model uses a word-based tokenizer.
    """
    if isinstance(model, textattack.models.helpers.LSTMForClassification):
        return True
    else:
        raise ValueError(f"Cannot perform GradientBasedWordSwap on model {model}.")


def transformation_consists_of(transformation, transformation_classes):
    """Determines if ``transformation`` is or consists only of instances of a
    class in ``transformation_classes``"""
    from textattack.transformations import CompositeTransformation

    if isinstance(transformation, CompositeTransformation):
        for t in transformation.transformations:
            if not transformation_consists_of(t, transformation_classes):
                return False
        return True
    else:
        for transformation_class in transformation_classes:
            if isinstance(transformation, transformation_class):
                return True
        return False


def transformation_consists_of_word_swaps(transformation):
    """Determines if ``transformation`` is a word swap or consists of only word
    swaps."""
    from textattack.transformations import WordSwap, WordSwapGradientBased

    return transformation_consists_of(transformation, [WordSwap, WordSwapGradientBased])


def transformation_consists_of_word_swaps_and_deletions(transformation):
    """Determines if ``transformation`` is a word swap or consists of only word
    swaps and deletions."""
    from textattack.transformations import WordDeletion, WordSwap, WordSwapGradientBased

    return transformation_consists_of(
        transformation, [WordDeletion, WordSwap, WordSwapGradientBased]
    )
