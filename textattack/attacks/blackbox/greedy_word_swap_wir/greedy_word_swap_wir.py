import torch

from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.blackbox import BlackBoxAttack

class GreedyWordSwapWIR(BlackBoxAttack):
    """
    An attack that greedily chooses from a list of possible 
    perturbations for each index, after ranking indices by importance.
    Reimplementation of paper:
    Is BERT Really Robust? A Strong Baseline for Natural Language Attack on 
    Text Classification and Entailment by Jin et. al, 2019
    https://github.com/jind11/TextFooler 
    Args:
        model: The PyTorch NLP model to attack.
        transformation: The type of transformation.
        max_depth (:obj:`int`, optional): The maximum number of words to change. Defaults to 32. 
    """

    def __init__(self, model, transformations=[],  max_depth=32):
        super().__init__(model)
        self.transformation = transformations[0]
        self.max_depth = max_depth
        
    def _attack_one(self, original_label, tokenized_text):
        original_tokenized_text = tokenized_text
        num_words_changed = 0
       
        # Sort words by order of importance
        orig_probs = self._call_model([tokenized_text]).squeeze()
        orig_prob = orig_probs.max()
        len_text = len(tokenized_text.words)
        leave_one_texts = \
            [tokenized_text.replace_word_at_index(i,'[UNK]') for i in range(len_text)]
        leave_one_probs = self._call_model(leave_one_texts)
        leave_one_probs_argmax = leave_one_probs.argmax(dim=-1)
        importance_scores = (orig_prob - leave_one_probs[:, original_label] 
            + (leave_one_probs_argmax != original_label).float() *
            (leave_one_probs.max(dim=-1)[0]
            - torch.index_select(orig_probs, 0, leave_one_probs_argmax))).data.cpu().numpy()
        index_order = (-importance_scores).argsort()

        new_tokenized_text = None
        new_text_label = None
        i = 0
        while ((self.max_depth is None) or num_words_changed <= self.max_depth) and i < len(index_order):
            transformed_text_candidates = self.get_transformations(
                self.transformation,
                tokenized_text,
                original_tokenized_text,
                indices_to_replace=[index_order[i]])
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            num_words_changed += 1
            scores = self._call_model(transformed_text_candidates)
            # The best choice is the one that minimizes the original class label.
            best_index = scores[:, original_label].argmin()
            # If we changed the label, return the index with best similarity.
            new_text_label = scores[best_index].argmax().item()
            if new_text_label != original_label:
                # @TODO: Use vectorwise operations
                new_tokenized_text = None
                max_similarity = -float('inf')
                for i in range(len(transformed_text_candidates)):
                    if scores[i].argmax().item() == new_text_label:
                        candidate = transformed_text_candidates[i]
                        try:
                            similarity_score = candidate.attack_attrs['similarity_score']
                        except KeyError:
                            # If the attack was run without any similarity metrics, 
                            # candidates won't have a similarity score. In this
                            # case, break and return the candidate that changed
                            # the original score the most.
                            new_tokenized_text = transformed_text_candidates[best_index]
                            break
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            new_tokenized_text = candidate
                return AttackResult( 
                    original_tokenized_text, 
                    new_tokenized_text, 
                    original_label,
                    new_text_label
                )
            tokenized_text = transformed_text_candidates[best_index]
        
        return FailedAttackResult(original_tokenized_text, original_label)
