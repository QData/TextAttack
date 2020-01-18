from .search import Search

from textattack.attacks import AttackResult, FailedAttackResult
from textattack.search import BlackBoxAttack

class GreedyWordSwap(Search):
    """ 
    An attack that greedily chooses from a list of possible 
    perturbations.

    Args:
        model: The PyTorch NLP model to attack.
        transformation: The type of transformation.
        max_depth (:obj:`int`, optional): The maximum number of words to change. Defaults to 32. 
        
    """
    def __init__(self, get_transformations, max_depth=32):
        self.get_transformations = get_transformations
        self.max_depth = max_depth
        
    def __call__(self, original_label, tokenized_text):
        original_tokenized_text = tokenized_text
        original_prob = self._call_model([tokenized_text]).squeeze().max()
        num_words_changed = 0
        unswapped_word_indices = list(range(len(tokenized_text.words)))
        new_tokenized_text = None
        new_text_label = None
        while num_words_changed <= self.max_depth and len(unswapped_word_indices):
            num_words_changed += 1
            transformed_text_candidates = self.get_transformations(
                tokenized_text, indices_to_replace=unswapped_word_indices)
            if len(transformed_text_candidates) == 0:
                # If we did not find any possible perturbations, give up.
                break
            scores = self._call_model(transformed_text_candidates)
            # The best choice is the one that minimizes the original class label.
            best_index = scores[:, original_label].argmin()
            new_tokenized_text = transformed_text_candidates[best_index]
            # If we changed the label, break.
            new_text_label = scores[best_index].argmax().item()
            if new_text_label != original_label:
                new_prob = scores[best_index].max()
                break
            # Otherwise, remove this word from list of words to change and
            # iterate.
            word_swap_loc = tokenized_text.first_word_diff_index(new_tokenized_text)
            tokenized_text = new_tokenized_text
            unswapped_word_indices.remove(word_swap_loc)
           
        
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
