from .attack import Attack
from textattack.attack_results import FailedAttackResult, SuccessfulAttackResult
import numpy as np

class BeamSearch(Attack):
    """ 
    An attack that greedily chooses from a list of possible 
    perturbations.
    
    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation (Transformation): The type of transformation.
        beam_width (int): the number of candidates to retain at each step
        max_words_changed (:obj:`int`, optional): The maximum number of words 
            to change.
        
    """
    def __init__(self, goal_function, transformation, constraints=[], beam_width=8, 
            max_words_changed=32):
        super().__init__(goal_function, transformation, constraints=constraints)
        self.beam_width = beam_width
        self.max_words_changed = max_words_changed
        
    def attack_one(self, original_tokenized_text, correct_output):
        max_words_changed = min(
            self.max_words_changed, 
            len(original_tokenized_text.words)
        )
        original_result = self.goal_function.get_results([original_tokenized_text], correct_output)[0]
        default_unswapped_word_indices = list(range(len(original_tokenized_text.words)))
        beam = [(original_tokenized_text, default_unswapped_word_indices)]
        num_words_changed = 0
        best_result = None
        while num_words_changed < max_words_changed:
            num_words_changed += 1
            potential_next_beam = []
            for text, unswapped_word_indices in beam:
                transformations = self.get_transformations(
                        text, indices_to_replace=unswapped_word_indices,
                        original_text=original_tokenized_text
                )
                for next_text in transformations:
                    new_unswapped_word_indices = unswapped_word_indices.copy()
                    modified_word_index = next_text.attack_attrs['modified_word_index']
                    new_unswapped_word_indices.remove(modified_word_index)
                    potential_next_beam.append((next_text, new_unswapped_word_indices))
            if len(potential_next_beam) == 0:
                # If we did not find any possible perturbations, give up.
                return FailedAttackResult(original_result)
            transformed_text_candidates = [text for (text,_) in potential_next_beam]
            results = self.goal_function.get_results(transformed_text_candidates, correct_output)
            scores = np.array([r.score for r in results])
            # If we succeeded, break
            best_result = results[scores.argmax()]
            if best_result.succeeded:
                break
            # Otherwise, refill the beam. This works by sorting the scores
            # in descending order and filling the beam from there.
            best_indices = -scores.argsort()[:self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]
        
        if best_result is None:
            return FailedAttackResult(original_result, best_result)
        else:
            return SuccessfulAttackResult(original_result, best_result)
