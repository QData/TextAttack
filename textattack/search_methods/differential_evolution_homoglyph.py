import numpy as np
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from scipy.optimize import differential_evolution
from typing import List

class DifferentialEvolutionSearchHomoglyph(SearchMethod):
    
    def __init__(self, popsize=3, maxiter=5, verbose=True, max_perturbs=1):
        self.popsize = popsize
        self.maxiter = maxiter
        self.verbose = verbose
        self.max_perturbs = max_perturbs

    def _obj(self, perturbation_vector: List[float], initial_result, glyph_map):
        cand = self._candidate(perturbation_vector, initial_result, glyph_map)
        result, _ = self.get_goal_result(cand)
        return result.score

    def _candidate(self, perturbation_vector: List[float], initial_result, glyph_map):
        ret = self.apply_perturbation(initial_result.attacked_text, perturbation_vector, glyph_map)
        return ret


    def perform_search(self, initial_result):
        """Performs the differential evolution search."""
        print("Starting differential evolution (homoglyphs)...")
        glyph_map = self.get_glyph_map(initial_result.attacked_text)
        def obj(perturbation_vector):
            return self._obj(perturbation_vector, initial_result, glyph_map)
        _bounds = self.bounds(initial_result.attacked_text, max_perturbs=self.max_perturbs)
        result = differential_evolution(obj, _bounds, disp=self.verbose, maxiter=self.maxiter, popsize=self.popsize)
        cand = self._candidate(result.x, initial_result, glyph_map)
        ret, _ = self.get_goal_result(cand)
        return ret

    def check_transformation_compatibility(self, transformation):
        """Differential evolution works with word swaps, deletions, and insertions."""
        return True

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["population_size", "max_iterations", "differential_weight", "crossover_probability"]
