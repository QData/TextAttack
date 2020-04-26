from textattack.goal_functions import *

from .utils import get_logger

logger = get_logger()

GOAL_FUNCTIONS_BY_MODEL = {
    (TargetedClassification, UntargetedClassification): [
            'models.classification.*',
            'models.entailment.*',
        ],
    (NonOverlappingOutput): [ # @todo add TargetedKeywordsOutput
            'models.translation.*',
            'models.summarization.*',
        ],
    
}

def validate_model_goal_function_compatibility(model, goal_function):
    """
        Determines if `model` is task-compatible with `goal_function`. 
        
        For example, a text-generative model like one intended for translation
            or summarization would not be compatible with a goal function
            that requires probability scores, like the UntargetedGoalFunction.
    """
    import pdb; pdb.set_trace() # TODO figure out how to do glob matching with model class
    logger.warn(f'Unknown if model {model} compatible with goal function {goal_function}.')
    return True