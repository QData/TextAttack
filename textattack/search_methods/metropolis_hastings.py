import torch
import math
import warnings
import numpy as np
from .attack import Attack
from textattack.shared import utils
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
from transformers import AutoTokenizer, AutoModelWithLMHead
from textattack.goal_functions import UntargetedClassification, TargetedClassification

class MetropolisHastingsSampling(Attack):
    """ 
    Uses Metropolis-Hastings Sampling to generate adversarial samples.
    Based off paper "Generating Fluent Adversarial Examples for Natural Langauges" by Zhang, Zhou, Miao, Li (2019)

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation (Transformation): The type of transformation.
        max_iter (int): The maximum number of sampling to perform
        lm_type (str): The language model to use to estimate likelihood of text.
            Currently supported LM is "gpt-2"
    """

    def __init__(self, goal_function, transformation, constraints=[], max_iter = 500, lm_type = "gpt-2"):
        super().__init__(goal_function, transformation, constraints=constraints)

        if not (isinstance(self.goal_function, UntargetedClassification) \
            or isinstance(self.goal_function, TargetedClassification)):
            raise ValueError(f"Unsupported goal function: {type(self.goal_function).__name__}")

        self.max_iter = max_iter
        self.lm_type = lm_type

        self.lm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.language_model = AutoModelWithLMHead.from_pretrained("gpt2")

        try:
            # Try to use GPU, but sometimes we might have out-of-memory issue
            # Having the below line prevents CUBLAS error when we try to switch to CPU 
            torch.cuda.current_blas_handle()
            self.language_model = self.language_model.to(utils.get_device())
            self.use_gpu = True
        except RuntimeError as error:
            if "CUDA out of memory" in str(error):
                warnings.warn("CUDA out of memory. Running GPT-2 for Metropolis Hastings on CPU.")
                self.language_model = self.language_model.to("cpu")
                self.use_gpu = False
            else:
                raise error
        
        self.language_model.eval()

    def lm_score(self, text):
        """
        Assigns likelihood of a text as 1/perplexity(text)
        Args:
            text (str)
        Returns: 1/perplexity(text)
        """
        input_ids = self.lm_tokenizer.encode(text)
        input_ids.append(self.lm_tokenizer.eos_token_id)
        input_ids = torch.tensor(input_ids)
        if self.use_gpu:
            input_ids = input_ids.to(utils.get_device())
        
        with torch.no_grad():
            loss = self.language_model(input_ids, labels=input_ids)[0].item()
            del input_ids
        
        perplexity = math.exp(loss)
        return 1/perplexity
        
    def stationary_dist(self, x, original_label):
        """
        Unnormalized estimation of the distribution that we want to sample from.
        Args:
            x (TokenizedText)
            original_label
        Returns: lm_score(x) * prob_model(wrong|x)
        """
        model_result = self.goal_function.get_results([x], original_label)[0]
        lm_score = self.lm_score(x.text)
        if isinstance(self.goal_function, UntargetedClassification):
            # If untargetted, we care about P(y != y_correct).
            # We add b/c result.score is negative. 
            return lm_score * (1 + model_result.score)
        else:
            return lm_score * model_result.score

    def batch_stationary_dist(self, x_list, original_label):
        """
        Unnormalized estimation of the distribution that we want to sample from.
        Process items in batch to avoid unnecessary calls to model. 
        Args:
            x_list (list): List of TokenizedText
            original_label
        Returns: list of floats representing lm_score(x) * prob_model(wrong|x)
        """
        batch_size = len(x_list)
        model_results = self.goal_function.get_results(x_list, original_label)
        lm_scores = [self.lm_score(x.text) for x in x_list]

        if isinstance(self.goal_function, UntargetedClassification):
            scores = [lm_scores[i] * (1 + model_results[i].score) for i in range(batch_size)]
        else:
            scores = [lm_scores[i] * model_results[i].score for i in range(batch_size)]

        return scores, model_results

    def normalize(self, values):
        """
        Take list of values and normalize it into a probability distribution
        TODO Consider softmax?
        """
        s = sum(values)
        return [v/s for v in values]

    def attack_one(self, tokenized_text, correct_output):
        original_result = self.goal_function.get_results([tokenized_text], correct_output)[0]

        text_len = len(tokenized_text.words)
        max_iter = max(self.max_iter, text_len)

        current_result = original_result
        current_text = tokenized_text
        current_score = self.stationary_dist(current_text, correct_output)
        
        for n in range(max_iter):
            # i-th word we want to transform
            i = n % text_len

            transformations = self.get_transformations(
                        current_text, 
                        indices_to_replace=[i],
                        original_text=tokenized_text
                    )

            if len(transformations) == 0:                
                continue

            scores, model_results = self.batch_stationary_dist(transformations, correct_output) 
            proposal_dist = self.normalize(scores)
            # Choose one transformation randomly according to proposal distribution
            jump = np.random.choice(list(range(len(transformations))), p=proposal_dist)

            # Now we have calculate probability of return proposal g(x'|x)
            reverse_transformations = self.get_transformations(
                        transformations[jump], indices_to_replace=[i],
                        original_text=tokenized_text
                    )

            reverse_jump = -1
            for k in range(len(reverse_transformations)):
                if reverse_transformations[k].text == current_text.text:
                    # Transition x -> x' exists
                    reverse_jump = k
                    break
            if reverse_jump == -1:
                return_prob = 0
            else:
                ret_scores, _ = self.batch_stationary_dist(reverse_transformations, correct_output)
                return_prob = self.normalize(ret_scores)[reverse_jump]

            """
            According to Metropolis-Hastings algorithm
            let f(x) be value proportional to target distribution p(x)
            and g(x|x') be transition probability from x' to x.
            Then, acceptance ratio = min(1, (f(x')*g(x|x')) / (f(x) * g(x'|x)))
            """
            acceptance_ratio = (scores[jump] * return_prob) / (current_score * proposal_dist[jump])
            acceptance_ratio = min(1, acceptance_ratio)
            u = np.random.uniform(low=0.0, high=1.0)

            if acceptance_ratio <= u or model_results[jump].succeeded:
                # Accept the proposed jump
                current_result = model_results[jump]
                current_text = transformations[jump]
                current_score = scores[jump]

            if current_result.succeeded:
                break

        if correct_output == current_result.output:
            return FailedAttackResult(original_result, current_result)
        else:
             return SuccessfulAttackResult(original_result, current_result)
