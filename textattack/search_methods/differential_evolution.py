from textattack.search_methods import SearchMethod
from scipy.optimize import differential_evolution
from typing import List
import numpy as np

from textattack.shared import AttackedText
from textattack.goal_function_results import GoalFunctionResult

from textattack.shared.validators import transformation_consists_of_word_swaps_differential_evolution

class DifferentialEvolution(SearchMethod):
    """
    A black-box adversarial search method using Differential Evolution (DE).

    This method searches for adversarial text examples by evolving a population
    of perturbation vectors and applying them to the input text. Only works with
    transformations that extend :class:`~textattack.transformations.word_swaps.WordSwapDifferentialEvolution`.
    """
    
    def __init__(self, popsize=3, maxiter=5, verbose=False, max_perturbs=1):
        """
        A black-box adversarial search method that uses Differential Evolution
        to find perturbations that are imperceptible but fool a model.
        """
        self.popsize = popsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.max_perturbs = max_perturbs

    def perform_search(self, initial_result: GoalFunctionResult) -> GoalFunctionResult:
        """
        Runs the DE optimization to find a successful adversarial attack.

        Args:
            initial_result (GoalFunctionResult): The starting point for the attack.

        Returns:
            GoalFunctionResult: The best adversarial candidate found (or original if no improvement).
        """
        initial_text = initial_result.attacked_text
        bounds_and_precomputed = self.get_bounds_and_precomputed(initial_text, self.max_perturbs)
        bounds = bounds_and_precomputed[0]
        precomputed = bounds_and_precomputed[1]

        best_score = np.inf
        best_result_found = None

        def obj(perturbation_vector): 
            nonlocal best_score, best_result_found
            cand: AttackedText = self.apply_perturbation(initial_text, perturbation_vector, precomputed) 
            if (len(self.filter_transformations([cand], initial_text, initial_text)) == 0):
                return np.inf
            result = self.get_goal_results([cand])[0][0]
            cur_score = -result.score
            if (cur_score <= best_score):
                best_result_found = result
            return cur_score

        _ = differential_evolution(obj, bounds, disp=self.verbose, maxiter=self.maxiter, popsize=self.popsize) # minimises obj
        
        if best_result_found is None:
            return initial_result
        return best_result_found

    def check_transformation_compatibility(self, transformation):
        if transformation_consists_of_word_swaps_differential_evolution(transformation):
            self.apply_perturbation = transformation.apply_perturbation
            self.get_bounds_and_precomputed = transformation.get_bounds_and_precomputed
            return True
        
        return False

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["popsize", "maxiter", "max_perturbs", "verbose"]
