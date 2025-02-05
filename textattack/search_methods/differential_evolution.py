import numpy as np
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from scipy.optimize import differential_evolution
from typing import List

class DifferentialEvolutionSearch(SearchMethod):
    """
    Differential Evolution Search
    =============================

    An implementation of differential evolution for generating adversarial 
    examples in NLP tasks. This approach aims to maximise a fitness function 
    over a population of perturbed texts.

    Args:
        population_size (int): The number of candidate solutions in the population.
        max_iterations (int): The maximum number of evolution iterations.
        differential_weight (float): The scaling factor for mutation (F).
        crossover_probability (float): The probability of crossover (CR).
    """

    def __init__(self, popsize=3, maxiter=5, verbose=True):
        self.popsize = popsize
        self.maxiter = maxiter
        self.verbose = verbose

    def _obj(self, perturbation_vector: List[float], initial_result, glyph_map):
        cand = self._candidate(perturbation_vector, initial_result, glyph_map)
        result, _ = self.get_goal_result(cand)
        return -result.score

    def _candidate(self, perturbation_vector: List[float], initial_result, glyph_map):
        ret = self.apply_perturbation(initial_result.attacked_text, perturbation_vector, glyph_map)
        return ret


    def perform_search(self, initial_result):
        """Performs the differential evolution search."""
        print("Starting differential evolution...")
        glyph_map = self.get_glyph_map(initial_result.attacked_text)
        def obj(perturbation_vector):
            return self._obj(perturbation_vector, initial_result, glyph_map)
        _bounds = self.bounds(initial_result.attacked_text, max_perturbs=1)
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
