"""
Imperceptible Perturbations Algorithm
======================================

"""

from .attack_recipe import AttackRecipe
from textattack.goal_functions import TargetedClassification, TargetedStrict, NamedEntityRecognition, LogitSum, MaximizeLevenshtein, MinimizeBleu
from textattack.transformations import WordSwapInvisibleCharacters, WordSwapHomoglyphSwap, WordSwapDeletions, WordSwapReorderings
from textattack.search_methods import DifferentialEvolution
from textattack import Attack


class BadCharacters2021(AttackRecipe):

    """
    Imperceptible Perturbations Attack Recipe
    =======================================================

    Implements imperceptible adversarial attacks on NLP models as outlined in the Bad Characters paper
    https://arxiv.org/abs/2106.09898.

    This recipe combines imperceptible transformations with the Differential Evolution 
    search method. It supports a variety of goal functions (targeted, untargeted, 
    NER, translation) and several types of character-level perturbations.

    Transformations supported:
    - WordSwapInvisibleCharacters: injects invisible Unicode characters
    - WordSwapHomoglyphSwap: replaces characters with homoglyphs
    - WordSwapDeletions: inserts deletion control characters
    - WordSwapReorderings: inserts reordering control characters

    Goal functions supported:
    - Targeted classification (probability output)
    - Strict targeted classification (probability output)
    - Named Entity Recognition (list of entity dicts output)
    - Logit sum (for logits-based classifiers like toxic comment detection)
    - Translation BLEU score minimization
    - Translation Levenshtein distance maximization

    All transformations are compatible with all goal functions.

    Note: This recipe assumes the model wrapper is compatible with the goal function 
    chosen. For example, a Named Entity Recognition goal function expects a model wrapper 
    that outputs a list of dictionaries per input, while classification goals expect 
    probability or logit arrays.
    """

    @staticmethod
    def build(model_wrapper, goal_function_type: str = None, perturbation_type: str = None, allow_skip: bool = False, perturbs=1, popsize=32, maxiter=10, **goal_function_kwargs):
        """
        Builds an imperceptible attack instance.

        Args:
            model_wrapper: A TextAttack model wrapper compatible with the selected goal function.
            goal_function_type (str, optional): One of:
                - "targeted_classification": targeted attack on a classification model (default).
                - "targeted_strict": stricter targeted attack on a classification model.
                - "named_entity_recognition": token-level targeted attack on a NER model.
                - "logit_sum": untargeted attack minimizing total logits.
                - "minimize_bleu": attack minimizing BLEU score between original and perturbed translations.
                - "maximize_levenshtein": attack maximizing Levenshtein distance between original and perturbed translations.
            perturbation_type (str, optional): One of:
                - "homoglyphs" (default)
                - "invisible"
                - "deletions"
                - "reorderings"
            allow_skip (bool): If set to False, the attack will continue even if attacking the unperturbed input string already completes the goal. Set to False in the paper.
            perturbs (int): Maximum number of perturbations allowed per input string. Values from 1 to 5 were used in the paper.
            popsize (int): Population size for differential evolution. Set to 32 in the paper.
            maxiter (int): Maximum number of generations for differential evolution. Set to 10 in the paper.
            **goal_function_kwargs: Additional arguments passed to the goal function.

        Returns:
            textattack.Attack: Configured Attack instance.
        """

        if goal_function_type is None:
            """
            Defaults to TargetedClassification
            **goal_function_kwargs:
            - target_class: int = 0
            """
            goal_function = TargetedClassification(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs) 
        elif goal_function_type == "targeted_strict":
            """
            Pass in a model wrapper that returns an array of probabilities
            **goal_function_kwargs:
            - target_class: int = 0
            """
            goal_function = TargetedStrict(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
        elif goal_function_type == "targeted_classification":
            """
            Pass in a model wrapper that returns an array of probabilities
            **goal_function_kwargs:
            - target_class: int = 0
            """
            goal_function = TargetedClassification(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
        elif goal_function_type == "named_entity_recognition":
            """
            Pass in a model wrapper that returns a list of dictionaries each containing 'entity' and 'score' keys
            **goal_function_kwargs:
            - target_suffix: str (no default value; must specify)
            """
            goal_function = NamedEntityRecognition(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
        elif goal_function_type == "logit_sum":
            """
            Pass in a model wrapper that returns an array of logits
            **goal_function_kwargs:
            - target_logit_sum=None
            - first_element_threshold=None
            Error if both are specified. If neither is specified, first_element_threshold is set to 0.5.
            """
            goal_function = LogitSum(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
        elif goal_function_type == "minimize_bleu":
            """
            Pass in a model wrapper that returns a string
            **goal_function_kwargs:
            - target_bleu: float=0.0
            """
            goal_function = MinimizeBleu(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
        elif goal_function_type == "maximize_levenshtein":
            """
            Pass in a model wrapper that returns a string
            **goal_function_kwargs:
            - target_distance: float=None
            """
            goal_function = MaximizeLevenshtein(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
        else:
            raise ValueError("Invalid goal_function_type!")
        
        if perturbation_type is None:
            # Default to homoglyphs
            transformation = WordSwapHomoglyphSwap()
        elif perturbation_type == "homoglyphs":
            transformation = WordSwapHomoglyphSwap()
        elif perturbation_type == "invisible":
            transformation = WordSwapInvisibleCharacters()
        elif perturbation_type == "deletions":
            transformation = WordSwapDeletions()
        elif perturbation_type == "reorderings":
            transformation = WordSwapReorderings()
        else:
            raise ValueError("Invalid perturbation_type!")
        
        
        search_method = DifferentialEvolution(
            popsize=popsize, 
            maxiter=maxiter, 
            verbose=False,
            max_perturbs=perturbs
        )

        constraints = []

        return Attack(goal_function, constraints, transformation, search_method)

        
            
