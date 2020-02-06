from .attack import Attack
from textattack.attack_results import AttackResult, FailedAttackResult

class BeamSearch(Attack):
    """ 
    An attack that greedily chooses from a list of possible 
    perturbations.
    Args:
        model: The model to attack.
        transformation: The type of transformation.
        max_words_changed (:obj:`int`, optional): The maximum number of words to change. Defaults to 32. 
        
    """
    def __init__(self, model, transformation, constraints=[], beam_width=8, max_words_changed=32):
        super().__init__(model, transformation, constraints=constraints)
        self.beam_width = beam_width
        self.max_words_changed = max_words_changed
        
    def attack_one(self, original_label, tokenized_text):
        original_tokenized_text = tokenized_text
        original_prob = self._call_model([tokenized_text]).squeeze().max()
        num_words_changed = 0
        unswapped_word_indices = list(range(len(tokenized_text.words)))
        new_tokenized_text = None
        new_text_label = None
        beam = [(tokenized_text, unswapped_word_indices)]
        self.max_words_changed = min(self.max_words_changed, len(tokenized_text.words))
        while (self.max_words_changed is not None) and num_words_changed < self.max_words_changed:
            num_words_changed += 1
            potential_next_beam = []
            for text, unswapped_word_indices in beam:
                transformations = self.get_transformations(
                        text, indices_to_replace=unswapped_word_indices
                )
                for next_text in transformations:
                    new_unswapped_word_indices = unswapped_word_indices[:]
                    try:
                        new_unswapped_word_indices.remove(next_text.attack_attrs['modified_word_index'])
                    except:
                        import pdb; pdb.set_trace()
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
            # Otherwise, remove this word from list of words to change, fill beam,
            # and iterate.
            best_indices = scores.argmax(dim=0)[:self.beam_width]
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
