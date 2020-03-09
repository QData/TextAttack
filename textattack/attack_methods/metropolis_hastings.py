import torch
import math
import numpy as np
from .attack import Attack
from textattack.shared import utils
from textattack.attack_results import AttackResult, FailedAttackResult
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class MetropolisHastingsSampling(Attack):
    """ 
    Uses Metropolis-Hastinings Sampling to generate adversarial samples.
    Based off paper "Generating Fluent Adversarial Examples for Natural Langauges" by Zhang, Zhou, Miao, Li (2019)
    N.B.: Only replacement of words are supported by TextAttack. No deletion or inseration 
    """
    def __init__(self, model, transformation, constraints=[], mmax_iter = 200, lm_type = "gpt-2"):
        super().__init__(model, transformation, constraints=constraints)
        self.max_iter = 500
        self.lm_type = lm_type
        if lm_type == "gpt-2":
            self.lm_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.language_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.language_model.to(utils.get_device())
            self.language_model.eval()

    def lm_score(self, text):
        """
        Args:
            text (str)
        Returns: 1/perplexity(x)
        """
        if self.lm_type == "gpt-2":
            
            input_ids = self.lm_tokenizer.encode(text, add_special_tokens=True)
            input_ids.append(self.lm_tokenizer.eos_token_id)

            input_ids = torch.tensor(input_ids).to(utils.get_device())
            loss = self.language_model(input_ids, labels=input_ids)[0].item()
            pp = math.exp(loss)

            return 1/pp
        
    def stationary_dist(self, x, original_label, target_label=None):
        """
        Stationary distribution that we want to sample from
        Args:
            x (TokenizedText)
            original_label
            target_label: If running targetted attack, set to output class. Else, none
        """
        prob = self._call_model([x])[0]
        lm_score = self.lm_score(x.text)
        if target_label:
            return lm_score * prob[target_label].item()
        else:
            return lm_score * torch.cat((prob[:original_label], prob[original_label:])).max().item()

    def stationary_dist_batch(self, x_list, original_label, target_label=None):
        """
        Stationary distribution that we want to sample from.
        Process item in batch and store new labels
        Args:
            x (TokenizedText)
            original_label
            target_label: If running targetted attack, set to output class. Else, none
        """
        probs = self._call_model(x_list)
        lm_scores = [self.lm_score(x.text) for x in x_list]

        if target_label:
            scores = [lm_scores[i] * probs[i][target_label].item() for i in range(len(x_list))]
        else:
            def cmax(x, original):
                values, indices = torch.topk(x, 2)
                if indices[0] == original:
                    return values[1].item()
                else:
                    return values[0].item()
            scores = [lm_scores[i] * cmax(probs[i], original_label) for i in range(len(x_list))]

        new_labels = [probs[i].argmax().item() for i in range(len(x_list))]

        return scores, new_labels, probs

    def attack_one(self, original_label, original_tokenized_text):

        original_prob = self._call_model([original_tokenized_text]).max()
        max_words_changed = min(
            self.max_iter, 
            len(original_tokenized_text.words)
        )
        current_text = original_tokenized_text
        new_label = original_label
        
        for n in range(self.max_iter):
            i = n % max_words_changed
            orig_stat_dist = self.stationary_dist(current_text, original_label)

            transformations = self.get_transformations(
                        current_text, indices_to_replace=[i],
                        original_text=original_tokenized_text
                )

            if not len(transformations):                
                continue

            scores, new_labels, probs = self.stationary_dist_batch(transformations, original_label) 
            norm_factor = sum(scores)
            proposal_prob = [s/norm_factor for s in scores]
            jump = np.random.choice(list(range(len(proposal_prob))), p=proposal_prob)

            # Now we have calculate probability of return proposal
            reverse_trans = self.get_transformations(
                        transformations[jump], indices_to_replace=[i],
                        original_text=original_tokenized_text
                )

            reverse_jump = -1
            for k in range(len(reverse_trans)):
                if reverse_trans[k].text == current_text.text:
                    reverse_jump = k
                    break
            if reverse_jump == -1:
                #print("Failed to find reverse jump.")
                continue
            dist, _, _ = self.stationary_dist_batch(reverse_trans, original_label)
            norm_factor = sum(dist)
            return_prob = dist[reverse_jump] / norm_factor

            acceptance_ratio = (scores[jump] * return_prob) / (orig_stat_dist * proposal_prob[jump])
            acceptance_ratio = min(1, acceptance_ratio)
            u = np.random.uniform(low=0.0, high=1.0)

            if acceptance_ratio <= u or new_labels[jump] != original_label:
                # Accept the proposed jump
                #print("Accept!")
                current_text = transformations[jump]
                new_label = new_labels[jump]

            if new_label != original_label:
                new_prob = probs[jump][new_label]
                break

        if original_label == new_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult( 
                original_tokenized_text, 
                current_text,
                original_label,
                new_label,
                float(original_prob),
                float(new_prob)
            )
