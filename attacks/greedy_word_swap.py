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
        unswapped_word_indices = list(range(len(tokenized_text.ids)))
        while num_words_changed <= self.max_depth and len(unswapped_word_indices):
            num_words_changed += 1
            candidate_tokenized_texts = self.perturbation.perturb(tokenized_text)
            # @TODO filter candidates by constraints here
            new_scores = self._call_model(candidate_tokenized_texts)
            # @TODO should this really be argmax tho?
                # and break if prediction changes
            best_index = candidate_tokenized_texts.argmax()
            new_tokenized_text = candidate_tokenized_texts[best_index]
            word_swap_loc = tokenized_text.first_word_diff(new_tokenized_text)
            tokenized_text = new_tokenized_text
            unswapped_word_indices.remove(word_swap_loc)
        return AttackResult(
            original_class,
            tokenized_text,
            original_class,
            new_class
        )