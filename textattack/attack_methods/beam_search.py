from .attack import Attack
from textattack.attack_results import AttackResult, FailedAttackResult

class BeamSearch(Attack):
    """ 
    An attack that greedily chooses from a list of possible 
    perturbations.
    Args:
        model (nn.Module): The model to attack.
        transformation (Transformation): The type of transformation.
        beam_width (int): the number of candidates to retain at each step
        max_words_changed (:obj:`int`, optional): The maximum number of words 
            to change.
        
    """
    def __init__(self, model, transformation, constraints=[], beam_width=8, 
            max_words_changed=32):
        super().__init__(model, transformation, constraints=constraints)
        self.beam_width = beam_width
        self.max_words_changed = max_words_changed
        
    def attack_one(self, original_label, original_tokenized_text):
        max_words_changed = min(
            self.max_words_changed, 
            len(original_tokenized_text.words)
        )
        original_prob = self._call_model([original_tokenized_text]).max()
        default_unswapped_word_indices = list(range(len(original_tokenized_text.words)))
        beam = [(original_tokenized_text, default_unswapped_word_indices)]
        num_words_changed = 0
        new_text_label = original_label
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
                return FailedAttackResult(original_tokenized_text, original_label)
            transformed_text_candidates = [text for (text,_) in potential_next_beam]
            scores = self._call_model(transformed_text_candidates)
            # The best choice is the one that minimizes the original class label.
            best_index = scores[:, original_label].argmin()
            new_tokenized_text = transformed_text_candidates[best_index]
            # If we changed the label, break.
            new_text_label = scores[best_index].argmax().item()
            if new_text_label != original_label:
                new_prob = scores[best_index].max()
                break
            # Otherwise, refill the beam. This works by sorting the scores from
            # the original label in ascending order and filling the beam from
            # there.
            best_indices = scores[:, original_label].argsort()[:self.beam_width]
            beam = [potential_next_beam[i] for i in best_indices]
           
        
        if original_label == new_text_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult( 
                original_tokenized_text, 
                new_tokenized_text, 
                original_label,
                new_text_label,
                float(original_prob),
                float(new_prob)
            )
