import numpy as np
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from scipy.optimize import differential_evolution
from typing import List

from textattack.shared.validators import transformation_is_imperceptible

class ImperceptibleDE(SearchMethod):
    
    def __init__(self, popsize=3, maxiter=5, verbose=True, max_perturbs=1):
        """
        A black-box adversarial search method that uses Differential Evolution
        to find perturbations that are imperceptible but fool a model.
        """
        self.popsize = popsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.max_perturbs = max_perturbs

    def _obj(self, perturbation_vector: List[float], initial_result, glyph_map=None):
        """
        Objective function for DE. Takes a perturbation vector and returns the attack score.
        """
        if (self.transformation_type == "homoglyphs"):
            cand = self._candidate(perturbation_vector, initial_result, glyph_map)
        else:
            cand = self._candidate(perturbation_vector, initial_result)
        result, _ = self.get_goal_result(cand)
        return result.score

    def _candidate(self, perturbation_vector: List[float], initial_result, glyph_map=None):
        """
        Applies the perturbation vector to the initial text and returns the candidate text.
        """
        if (self.transformation_type == "homoglyphs"):
            ret = self.apply_perturbation(initial_result.attacked_text, perturbation_vector, glyph_map)
        else:
            ret = self.apply_perturbation(initial_result.attacked_text, perturbation_vector)
        return ret

    def perform_search(self, initial_result):
        """
        Runs the DE optimization to find a successful adversarial attack.

        Args:
            initial_result: The starting point for the attack.
        
        Returns:
            A GoalFunctionResult representing the best adversarial candidate found.
        """
        print("Starting differential evolution...")
        use_glyph_map = (self.transformation_type == "homoglyphs")
        if (use_glyph_map):
            glyph_map = self.get_glyph_map(initial_result.attacked_text)
        def obj(perturbation_vector):
            if (use_glyph_map):
                return self._obj(perturbation_vector, initial_result, glyph_map)
            else:
                return self._obj(perturbation_vector, initial_result)

        # Get bounds for perturbation based on the number of allowed changes
        _bounds = self.bounds(initial_result.attacked_text, max_perturbs=self.max_perturbs)

        # Run the DE optimizer
        result = differential_evolution(obj, _bounds, disp=self.verbose, maxiter=self.maxiter, popsize=self.popsize)
        if (use_glyph_map):
            cand = self._candidate(result.x, initial_result, glyph_map)
        else:
            cand = self._candidate(result.x, initial_result)
        ret, _ = self.get_goal_result(cand)
        return ret

    def check_transformation_compatibility(self, transformation):
        """
        Ensures that the given transformation is compatible with imperceptible attacks.
        Sets the internal transformation type accordingly.
        Differential evolution works with word swaps, deletions, and insertions.
        """
        is_imperceptible, *category = transformation_is_imperceptible(transformation)
        if (is_imperceptible):
            self.transformation_type = category[0]
        return is_imperceptible

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["population_size", "max_iterations", "differential_weight", "crossover_probability"]
