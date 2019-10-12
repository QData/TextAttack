from attack import Attack

class GreedyWordSwap(Attack):
    """ An attack that greedily chooses from a list of possible 
        perturbations.
    """
    def __init__(self, model, perturbation,  max_depth=32):
        super().__init__(model, perturbation)
        self.max_depth = max_depth
        
    def _attack_one(self, label, tokenized_text):
        original_text = tokenized_text
        num_words_changed = 0
        unswapped_word_indices = list(range(len(tokenized_text.words())))
        while num_words_changed <= self.max_depth and len(unswapped_word_indices):
            num_words_changed += 1
            perturbed_text_candidates = self.perturbation.perturb(tokenized_text,
                indices_to_replace=unswapped_word_indices)
            if not perturbed_text_candidates:
                # If we did not find any possible perturbations, give up.
                return None
            # @TODO filter candidates by constraints here
            print('# perturbations:', len(perturbed_text_candidates))
            scores = self._call_model(perturbed_text_candidates)
            # The best choice is the one that minimizes the original class label.
            best_index = scores[:, label].argmin()
            new_tokenized_text = perturbed_text_candidates[best_index]
            # If we changed the label, break.
            if scores[best_index].argmin() != label:
                break
            # Otherwise, remove this word from list of words to change and
            # iterate.
            word_swap_loc = tokenized_text.first_word_diff(new_tokenized_text)
            print('Best word swap:', word_swap_loc)
            print("new_tokenized_text:", new_tokenized_text.text)
            tokenized_text = new_tokenized_text
            unswapped_word_indices.remove(word_swap_loc)
            
        return AttackResult(
            original_class,
            tokenized_text,
            original_class,
            new_class
        )