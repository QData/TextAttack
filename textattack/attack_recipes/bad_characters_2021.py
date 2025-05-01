"""
Imperceptible Perturbations Algorithm
======================================

"""

from .attack_recipe import AttackRecipe
from textattack.goal_functions import TargetedClassification, TargetedStrict, TargetedBonus, NamedEntityRecognition, LogitSum, MaximizeLevenshtein, MinimizeBleu
from textattack.transformations import WordSwapInvisibleCharacters, WordSwapHomoglyphSwap, WordSwapDeletions, WordSwapReorderings
from textattack.search_methods import DifferentialEvolution
from textattack import Attack


class BadCharacters2021(AttackRecipe):

    """
    Imperceptible Perturbations Attack Recipe
    =========================================

    Implements imperceptible adversarial attacks on NLP models as outlined in the
    `Bad Characters paper <https://arxiv.org/abs/2106.09898>`_.

    This recipe combines imperceptible transformations with the Differential Evolution
    search method. It supports a variety of goal functions (targeted, untargeted,
    NER, translation) and several types of character-level perturbations.

    **Transformations supported:**

    - ``WordSwapInvisibleCharacters``: injects invisible Unicode characters
    - ``WordSwapHomoglyphSwap``: replaces characters with homoglyphs
    - ``WordSwapDeletions``: inserts deletion control characters
    - ``WordSwapReorderings``: inserts reordering control characters

    **Goal functions supported:**

    - ``TargetedClassification`` 
    - ``TargetedStrict`` 
    - ``TargetedBonus``
    - ``LogitSum`` (for logits-based classifiers like toxic comment detection)
    - ``MinimizeBleu`` (translation BLEU score minimization)
    - ``MaximizeLevenshtein`` (translation Levenshtein distance maximization)

    All transformations are compatible with all goal functions.

    Note:
    This recipe assumes the model wrapper is compatible with the goal function chosen.
    For example, a ``NamedEntityRecognition`` goal function expects a model wrapper
    that outputs a list of dictionaries per input, while ``LogitSum`` expects an array of logits.
    """

    @staticmethod
    def build(model_wrapper, goal_function_type: str, perturbation_type: str = None, allow_skip: bool = False, perturbs=1, popsize=32, maxiter=10, **goal_function_kwargs):
        """
        Builds an imperceptible attack instance.

        Parameters
        ----------
        model_wrapper : ModelWrapper
            A TextAttack model wrapper compatible with the selected goal function.
        goal_function_type : str, optional
            Goal function type. One of:
            
            - ``"targeted_classification"``: targeted attack on a classification model (default).
            - ``"targeted_strict"``: stricter targeted attack.
            - ``"targeted_bonus"``: bonus if prediction for target class is highest.
            - ``"named_entity_recognition"``: token-level NER attack.
            - ``"logit_sum"``: untargeted attack minimizing total logits.
            - ``"minimize_bleu"``: translation attack minimizing BLEU.
            - ``"maximize_levenshtein"``: translation attack maximizing Levenshtein distance.
        perturbation_type : str, optional
            Type of character-level perturbation. One of:

            - ``"homoglyphs"`` (default)
            - ``"invisible"``
            - ``"deletions"``
            - ``"reorderings"``
        allow_skip : bool
            If False, the attack will continue even if the goal is already satisfied.
        perturbs : int
            Maximum number of perturbations allowed per input string.
        popsize : int
            Population size for differential evolution. Typically 32.
        maxiter : int
            Maximum number of generations for differential evolution. Typically 10.
        **goal_function_kwargs : dict
            Additional arguments passed to the goal function.

        Returns
        -------
        textattack.Attack
            Configured Attack instance.
        """

        if goal_function_type == "targeted_classification":
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
        elif goal_function_type == "targeted_bonus":
            """
            Pass in a model wrapper that returns an array of probabilities
            **goal_function_kwargs:
            - target_class: int = 0
            """
            goal_function = TargetedBonus(model_wrapper, allow_skip = allow_skip, **goal_function_kwargs)
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

        
            
