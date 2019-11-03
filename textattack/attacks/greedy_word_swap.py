from attack import Attack, AttackResult

class GreedyWordSwap(Attack):
    """ An attack that greedily chooses from a list of possible 
        perturbations.
    """
    def __init__(self, model, perturbation,  max_depth=32):
        super().__init__(model, perturbation, "Greedy Word Swap")
        self.max_depth = max_depth
        
    def _attack_one(self, original_label, tokenized_text):
        original_tokenized_text = tokenized_text
        num_words_changed = 0
        unswapped_word_indices = list(range(len(tokenized_text.words())))
        new_tokenized_text = None
        new_text_label = None
        while num_words_changed <= self.max_depth and len(unswapped_word_indices):
            num_words_changed += 1
            perturbed_text_candidates = self.perturbation.perturb(tokenized_text,
                indices_to_replace=unswapped_word_indices)
            if len(perturbed_text_candidates) == 0:
                # If we did not find any possible perturbations, give up.
                return None
            scores = self._call_model(perturbed_text_candidates)
            # The best choice is the one that minimizes the original class label.
            best_index = scores[:, original_label].argmin()
            new_tokenized_text = perturbed_text_candidates[best_index]
            # If we changed the label, break.
            new_text_label = scores[best_index].argmax().item()
            if new_text_label != original_label:
                break
            # Otherwise, remove this word from list of words to change and
            # iterate.
            word_swap_loc = tokenized_text.first_word_diff(new_tokenized_text)
            tokenized_text = new_tokenized_text
            unswapped_word_indices.remove(word_swap_loc)
            
        return AttackResult(
            original_tokenized_text,
            new_tokenized_text,
            original_label,
            new_text_label
        )