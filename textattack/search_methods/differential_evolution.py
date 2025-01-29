import numpy as np
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from scipy.optimize import differential_evolution

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

    def __init__(self, population_size=20, max_iterations=10):
        self.population_size = population_size
        self.max_iterations = max_iterations

    def perform_search(self, initial_result):
        """Performs the differential evolution search."""
        print("Starting differential evolution...")
        def obj():
            def _obj(perturbation_vector):
                candidate = candidate(perturbation_vector)
                result, _ = self.get_result(candidate)
                return result.score
            return _obj
        def bounds():
            return self.bounds(initial_result.attacked_text, max_perturbs=3)
        def candidate(perturbation_vector):
            ret = self.apply_perturbation(initial_result.attacked_text, perturbation_vector)
            return ret
        result = differential_evolution(obj(), bounds(), disp=verbose, maxiter=self.max_iterations, popsize=self.population_size)
        candidate = candidate(result.x)
        ret = self.get_result(candidate)
        return ret

    def check_transformation_compatibility(self, transformation):
        """Differential evolution works with word swaps, deletions, and insertions."""
        return True

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["population_size", "max_iterations", "differential_weight", "crossover_probability"]
