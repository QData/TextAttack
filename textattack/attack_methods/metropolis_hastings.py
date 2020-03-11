import torch
import math
import numpy as np
from .attack import Attack
from textattack.shared import utils
from textattack.attack_results import AttackResult, FailedAttackResult
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from textattack.goal_functions import UntargetedClassification

class MetropolisHastingsSampling(Attack):
    """ 
    Uses Metropolis-Hastings Sampling to generate adversarial samples.
    Based off paper "Generating Fluent Adversarial Examples for Natural Langauges" by Zhang, Zhou, Miao, Li (2019)
    N.B.: Only replacement of words are supported by TextAttack. No deletion or insertion 
    """
    def __init__(self, model, transformation, constraints=[], max_iter = 500, lm_type = "gpt-2"):
        super().__init__(model, transformation, constraints=constraints)
        self.max_iter = max_iter
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
        
    def stationary_dist(self, x, original_label):
        """
        Stationary distribution that we want to sample from
        Args:
            x (TokenizedText)
            original_label
        """
        result = self.goal_function.get_results([x], original_label)[0]
        lm_score = self.lm_score(x.text)
        if isinstance(self.goal_function, UntargetedClassification):
            # If untargetted, we care about P(y != y_correct).
            # We add b/c result.score is negative. 
            return lm_score * (1 + result.score)
        else:
            return lm_score * result.score

    def stationary_dist_batch(self, x_list, original_label):
        """
        Stationary distribution that we want to sample from.
        Process items in batch and store new labels and probability score 
        to avoid unecessary calls to model. 
        Args:
            x_list (list): List of TokenizedText
            original_label
        """
        results = self.goal_function.get_results(x_list, original_label)
        lm_scores = [self.lm_score(x.text) for x in x_list]

        if len(results) != len(lm_scores):
            raise ValueError("Unequal list length between GoalFunctionResults and LM results.")

        if isinstance(self.goal_function, UntargetedClassification):
            scores = [lm_scores[i] * (1 + results[i].score.item()) for i in range(len(results))]
        else:
            scores = [lm_scores[i] * results[i].score.item() for i in range(len(results))]


        new_labels = [results[i].output for i in range(len(results))]

        return results, scores, new_labels

    def attack_one(self, tokenized_text, correct_output):
        original_result = self.goal_function.get_results([tokenized_text], correct_output)[0]

        max_iter = max(
            self.max_iter, 
            len(tokenized_text.words)
        )
        text_len = len(tokenized_text.words)

        current_result = original_result
        current_text = tokenized_text
        current_label = correct_output
        
        for n in range(max_iter):
            # i-th word we want to transform
            i = n % text_len
            orig_stat_dist = self.stationary_dist(current_text, correct_output)

            transformations = self.get_transformations(
                        current_text, indices_to_replace=[i],
                        original_text=tokenized_text
                )

            if len(transformations) == 0:                
                continue

            results, scores, new_labels = self.stationary_dist_batch(transformations, correct_output) 
            norm_factor = sum(scores)
            proposal_dist = [s/norm_factor for s in scores]
            # Choose one transformation randomly according to proposal distribution
            jump = np.random.choice(list(range(len(transformations))), p=proposal_dist)

            # Now we have calculate probability of return proposal
            reverse_transformations = self.get_transformations(
                        transformations[jump], indices_to_replace=[i],
                        original_text=tokenized_text
                )

            reverse_jump = -1
            for k in range(len(reverse_transformations)):
                if reverse_transformations[k].text == current_text.text:
                    reverse_jump = k
                    break
            if reverse_jump == -1:
                return_prob = 0 
            else:
                _, ret_scores, _ = self.stationary_dist_batch(reverse_transformations, correct_output)
                norm_factor = sum(ret_scores)
                return_prob = ret_scores[reverse_jump] / norm_factor

            # According to M-H algorithm, let P(x) be probability of x according to target dist 
            # and g(x|x') be transition probability from x' to x. Then,  
            # acceptance ratio = min (1, (P(x')*g(x|x')) / (P(x) * g(x'|x)))
            acceptance_ratio = (scores[jump] * return_prob) / (orig_stat_dist * proposal_dist[jump])
            acceptance_ratio = min(1, acceptance_ratio)
            u = np.random.uniform(low=0.0, high=1.0)

            if acceptance_ratio <= u or new_labels[jump] != correct_output:
                # Accept the proposed jump
                current_result = results[jump]
                current_text = transformations[jump]
                current_label = new_labels[jump]

            if current_label != correct_output:
                break

        if correct_output == current_label:
            return FailedAttackResult(tokenized_text, correct_output)
        else:
             return AttackResult( 
                    tokenized_text, 
                    current_text, 
                    correct_output,
                    current_label,
                    float(original_result.score),
                    float(current_result.score)
                )
