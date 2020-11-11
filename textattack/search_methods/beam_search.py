"""
Beam Search
===============

"""
import numpy as np

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class BeamSearch(SearchMethod):
    """An attack that maintinas a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation (Transformation): The type of transformation.
        beam_width (int): the number of candidates to retain at each step
    """

    def __init__(self, beam_width=8):
        self.beam_width = beam_width

    def _perform_search(self, initial_result):
        beam = [initial_result.attacked_text]
        best_result = initial_result
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            potential_next_beam = []
            for text in beam:
                transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text
                )
                for next_text in transformations:
                    potential_next_beam.append(next_text)
            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(potential_next_beam)
            scores = np.array([r.score for r in results])
            best_result = results[scores.argmax()]
            if search_over:
                return best_result

            # Refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = (-scores).argsort()[: self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]
        return best_result

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["beam_width"]
